#!/usr/bin/env python3
import os
import logging
import streamlit as st
from anthropic import Anthropic
import numpy as np
from typing import Dict, List
from pinecone import Pinecone
import time
import json
from datetime import datetime, timedelta
import html
import textwrap 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------------------
# VOCDatabaseQuerier Class Definition (unchanged) 
# ------------------------------------------------------------------------------
class VOCDatabaseQuerier:
    """
    This class queries:
    1) Maps the user's question to a question_type using a simple keyword matching approach.
    2) Fetches the offline summary for that question_type (e.g. "desc_community_impact_summary").
    3) Uses Anthropic to produce a final answer based on that summary and the full conversation history.
    """

    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str,
        anthropic_api_key: str
    ):
        logging.info("Initializing VOC Database Querier (Offline Summaries)...")
        
        # There were some initial issues with the API key, so we check if it's provided
        # This should already be integrated through GitHub Secrets
        if not anthropic_api_key:
            raise ValueError("Anthropic API key not provided")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key not provided")

        # Define Key     
        self.api_key = anthropic_api_key

        # Here we define the pinecone key, index name, and initialize the index
        logging.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.index = self.pc.Index(self.index_name)
        
        # Here we use describe_index_stats to check if the connection is successful
        try:
            stats = self.index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            logging.info(f"Connected to Pinecone index '{index_name}'. Vector count: {vector_count}")
        except Exception as e:
            logging.error(f"Error connecting to Pinecone: {e}")
            raise

        # Initialize Anthropic client
        logging.info("Initializing Anthropic client...")
        self.anthropic = Anthropic(api_key=self.api_key)
        logging.info("Anthropic client initialized successfully.")

        # These were the question types 
        self.question_types = {
            # Financial_challenges 1, 2, 3, and 4 
            "financial_challenges_1": {
                "context": "What specific challenges do you face in managing and forecasting your cash flow?",
                "keywords": ["cash flow", "manage cash", "forecast", "forecasting", "cashflow"]
            },
            "financial_challenges_2": {
                "context": "What specific financial tasks consume most of your time?",
                "keywords": ["time consuming", "financial tasks", "consume time", "time spent"]
            },
            "financial_challenges_3": {
                "context": "Tell us about a hard instance managing finances or getting a loan",
                "keywords": ["hard instance", "difficult", "challenge", "loan", "managing finances"]
            },
            "financial_challenges_4": {
                "context": "Challenges with applying for loans",
                "keywords": ["loan", "applying", "application", "credit", "approval"]
            },

            # Business Description
            "desc_business_brief": {
                "context": "A brief description of the business",
                "keywords": ["business description", "about business", "what business", "company"]
            },
            "desc_primary_products": {
                "context": "Primary products/services offered",
                "keywords": ["products", "services", "offerings", "what do you sell", "provide"]
            },
            "desc_community_impact": {
                "context": "Impact on the community",
                "keywords": ["community", "impact", "local", "society", "neighborhood"]
            },
            "desc_equity_inclusion": {
                "context": "Efforts to promote equity and inclusion",
                "keywords": ["equity", "inclusion", "diversity", "dei", "inclusive"]
            },

            # Business Goals and Growth
            "business_goals_1": {
                "context": "Achievements and business goals",
                "keywords": ["goals", "achievements", "milestones", "growth", "plan"]
            },
            "business_goals_2": {
                "context": "Daily tasks for a virtual CFO",
                "keywords": ["cfo", "financial officer", "daily tasks", "finance management"]
            },

            # Financial Tools and Advisory
            "financial_tool_needs": {
                "context": "Required features for financial management tool",
                "keywords": ["tool", "features", "financial management", "software", "app", "application"]
            },

            # Grant and Support
            "grant_usage": {
                "context": "How grant funds will be used",
                "keywords": ["grant", "funds", "money", "financial support", "use of"]
            },

            # Business Challenges
            "business_obstacles": {
                "context": "Major business obstacles and solutions",
                "keywords": ["obstacles", "challenges", "problems", "issues", "overcome"]
            },

            # Additional Context
            "additional_context": {
                "context": "Additional relevant information",
                "keywords": ["additional", "other", "more information", "context", "relevant"]
            },

            # Financial Advisor Questions
            "financial_advisor_questions": {
                "context": "Questions for financial advisor",
                "keywords": ["advisor", "advice", "financial advisor", "questions", "ask"]
            },

            # Financial assistance rationale 
            "reason_financial_assistance": {
                "context": "What is your main reason for seeking financial assistance for your business?",
                "keywords": ["financial assistance", "reason for funding", "seeking help", "business assistance", "funding need"]
            },

            # Planning responsibility 
            "financial_planning_responsible": {
                "context": "Who handles the financial planning and cash flow tracking at your business?",
                "keywords": ["financial planning", "cash flow tracking", "responsibility", "financial oversight", "management"]
            }
}

    def determine_question_type(self, user_query: str) -> str:
        """
        Purpose: Map the user's question to a question type using keyword matching.
        Input:
        - user_query: The user's input question.
        Output: The determined question type.
        """

        # The lower() function is used to make the keyword matching case-insensitive
        user_query = user_query.lower()
        
        # Here is a dictionary to store the scores for each question type
        scores = {}

        # For each question and info within the question_types dictionary, we check if the keywords are in the user query
        for qtype, info in self.question_types.items():
            score = 0
            for keyword in info["keywords"]:
                # If the keyword is in the user query, we increase the score
                if keyword.lower() in user_query:
                    score += 1
            scores[qtype] = score
        
        # Here we get the question that has the most points within the scores dictionary
        best_type = max(scores.items(), key=lambda x: x[1])
        
        # Worst case, we default to financial_challenges_1
        if best_type[1] == 0:
            logging.info("[determine_question_type] No keywords matched. Using default question type: financial_challenges_1")
            return "financial_challenges_1"
        
        logging.info(f"[determine_question_type] Query mapped to '{best_type[0]}' (score={best_type[1]})")
        return best_type[0]

    def get_offline_summary(self, question_type: str) -> str:
        """
        Purpose: Fetch the offline summary for the given question type from Pinecone.
        Input:
        - question_type: The question type for which to fetch the summary.
        Output: The summary text.
        """
        # Here we define the summary question type
        summary_qtype = f"{question_type}_summary"
        logging.info(f"Fetching offline summary for question_type='{summary_qtype}'")
        
        try:
            dummy_vector = [0.0] * 384  
            
            # Here we use the query() method to get the summary for the question type
            query_results = self.index.query(
                vector=dummy_vector,
                filter={"question_type": summary_qtype},
                top_k=1,
                include_metadata=True
            )
            
            # Check if there are any matches
            matches = getattr(query_results, 'matches', [])
            # If there are no matches, we log a warning and return an empty string
            if not matches:
                logging.warning(f"No summary found for {summary_qtype}")
                return ""
            
            # Extract the summary text from the metadata. We extract from the matches list
            summary_text = matches[0].metadata.get("text", "")
            if not summary_text:
                logging.warning(f"No text found in metadata for {summary_qtype}")
                return ""
                
            logging.info(f"Successfully retrieved summary for {summary_qtype}")
            # Return the summary text
            return summary_text
            
        except Exception as e:
            logging.error(f"Error fetching summary for {summary_qtype}: {e}")
            return ""

    def build_prompt_with_offline_summary(self, user_query: str, summary: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Purpose: Build the final prompt for Anthropic using the user query, summary, and conversation history.
        Input:
        - user_query: The user's input question.
        - summary: The offline summary text.
        - conversation_history: The full conversation history.
        Output: The final prompt string.
        """
        # Here we built the conversation thread that will be used in the prompt for context
        conversation_history_text = ""

        # For each message in conversation_history, we check if the role is user or assistant
        for msg in conversation_history:
            if msg["role"] == "user":
                conversation_history_text += f"User: {msg['content']}\n"
            else:
                conversation_history_text += f"Assistant: {msg['content']}\n"
                    
        prompt = f"""
            You are an expert data analyst assistant for **Breva**, a financial technology company focused on supporting small and medium-sized businesses (SMBs). You're working with the Thrive Grant program team to analyze application data.

            ### **CONTEXT AND PURPOSE**
            - You are exclusively serving Breva employees who need to extract insights from Thrive Grant applications
            - The Thrive Grant program provides financial assistance to SMBs facing various challenges
            - Your analysis will help Breva improve their products, services, and grant program
            - You're analyzing real responses from grant applicants about their business needs and challenges

            ### **CONVERSATION HISTORY**
            ---
            {conversation_history_text}
            ---

            ### **CURRENT QUERY**
            The user (a Breva employee) just asked: **"{user_query}"**

            ### **RELEVANT VOC DATA**
            Below is a summary of relevant Voice of Customer (VOC) data from our grant applications database:
            ---
            {summary}
            ---

            ### **RESPONSE REQUIREMENTS**

            1. **Analytical Approach**
            - Analyze patterns, trends, and outliers in the data
            - Identify key segments and how they differ (by business size, industry, etc. if available)
            - Provide quantitative breakdowns with percentages when possible
            - Highlight surprising or counterintuitive findings

            2. **Response Structure**
            - Start with a "Key Findings" section (3-5 bullet points of most important insights)
            - Use clear headings and subheadings to organize information
            - Include a "Business Implications" section
            - End with 1-2 suggested follow-up questions or areas for deeper investigation

            3. **Data Presentation**
            - Present statistics clearly (X% of respondents mentioned Y)
            - Use comparative language (more likely to, less frequently than, etc.)
            - Distinguish between facts from the data vs. your interpretations
            - Support insights with specific examples or quotes from the data when relevant

            4. **Tone and Focus**
            - Be objective, analytical, and business-focused
            - Avoid giving advice to SMBs directly
            - Frame everything as insights FOR Breva employees ABOUT applicant needs
            - Maintain a helpful, collaborative tone with the Breva team member

            Now, craft a concise, structured, data-driven response that helps the Breva employee understand the patterns and implications in this VOC data.
            """
        return prompt

    def generate_answer(self, user_query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Purpose: Generate an answer using Anthropic based on the user query, offline summary, and conversation history.
        Input:
        - user_query: The user's input question.
        - conversation_history: The full conversation history.
        Output: The generated answer string.
        """
        try:
            # First, we determine what type of question the user is asking
            qtype = self.determine_question_type(user_query)
            
            # For the given question type, we get the offline summary using the get_offline_summary method
            offline_summary = self.get_offline_summary(qtype)
            if not offline_summary:
                return (f"No offline summary found for question type '{qtype}'. "
                        f"Try a different approach or run offline summarization first.")
            
            # Build the final prompt including the conversation history
            final_prompt = self.build_prompt_with_offline_summary(user_query, offline_summary, conversation_history)
            logging.info("FINAL PROMPT constructed.")
    
            # Here we use the anthropic API to generate the answer
            response = self.anthropic.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=8192,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": final_prompt
                }]
            )
            
            # Extract the answer from the response
            if hasattr(response.content[0], 'text'):
                return response.content[0].text
            else:
                return str(response.content[0])
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return f"[Error generating answer: {str(e)}]"

# ------------------------------------------------------------------------------
# Helper functions for the Streamlit app
# ------------------------------------------------------------------------------
def initialize_session_state():
    """
    Purpose: Initialize session state variables for the Streamlit app.
    Input: None
    Output: None
    """

    # The following if statements check if the session state variables are already set
    # If not, they are initialized with default values
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
        
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
        
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
        
    if "temp_user_input" not in st.session_state:
        st.session_state.temp_user_input = ""

def initialize_querier():
    """
    Purpose: Initialize the VOC Database Querier if not already done.
    Input: None
    Output: None
    """

    # If the querier is not already initialized, we create a new instance
    if "querier" not in st.session_state:
        try:
            # Here we use a spinner to indicate loading. We also check if the API keys are set
            with st.spinner("Initializing chatbot..."):
                pinecone_api_key = st.secrets.get("pinecone_api_key")
                anthropic_api_key = st.secrets.get("anthropic_api_key")
                
                if not pinecone_api_key or not anthropic_api_key:
                    st.error("API keys not found in Streamlit secrets. Please add them in the Streamlit Cloud dashboard.")
                    st.info("Go to your app settings in Streamlit Cloud, navigate to 'Secrets', and add 'pinecone_api_key' and 'anthropic_api_key'.")
                    return False
                
                # Here we use the VOCDatabase Querier class to create a new instance
                st.session_state.querier = VOCDatabaseQuerier(
                    pinecone_api_key=pinecone_api_key,
                    index_name="voc-index-2025-q2",
                    anthropic_api_key=anthropic_api_key
                )
                return True
        except Exception as e:
            st.error(f"Error initializing the VOC Database Querier: {e}")
            return False
    return True

def download_chat_history():
    """
    Purpose: Download the chat history as a JSON file.
    Input: None
    Output: JSON string of chat history.
    """
    if not st.session_state.messages:
        return None
    # Here we create a JSON object with the session ID, timestamp, and messages
    chat_export = {"session_id": st.session_state.session_id, "timestamp": datetime.now().isoformat(), "messages": st.session_state.messages}
    return json.dumps(chat_export, indent=2)

def apply_custom_css():
    """
    Purpose: Apply custom CSS styles to the Streamlit app.
    Input: None
    Output: None
    """
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    
    <style>
    /* Breva Theme - Professional UI for Thrive Grant Insights */

    /* Base variables for the theme */
    :root {
        /* Primary color palette */
        --breva-primary: #4361EE;
        --breva-primary-light: #4895EF;
        --breva-primary-dark: #3A0CA3;
        
        /* Secondary color palette */
        --breva-secondary: #4CC9F0;
        --breva-accent: #F72585;
        
        /* Neutral colors */
        --breva-bg-dark: #111827;
        --breva-bg-medium: #1F2937;
        --breva-bg-light: #374151;
        --breva-text-light: #F9FAFB;
        --breva-text-medium: #D1D5DB;
        --breva-text-dark: #9CA3AF;
        --breva-border: #4B5563;
        
        /* Chat bubbles */
        --user-bubble-color: #4361EE;
        --assistant-bubble-color: #1F2937;
        
        /* UI components */
        --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --transition-speed: 0.3s;
    }

    /* Global styles */
    body {
        background-color: var(--breva-bg-dark);
        color: var(--breva-text-light);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        line-height: 1.6;
    }

    /* Improve layout and container styling */
    .main .block-container {
        max-width: 1200px;
        padding: 1rem 2rem;
    }

    /* Override Streamlit default styling */
    .stApp {
        background-color: var(--breva-bg-dark);
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--breva-text-light);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    h1 {
        font-size: 2rem;
        letter-spacing: -0.025em;
        border-bottom: 2px solid var(--breva-primary);
        padding-bottom: 0.5rem;
        display: inline-block;
    }

    h2 {
        font-size: 1.5rem;
        color: var(--breva-text-light);
    }

    h3 {
        font-size: 1.25rem;
        color: var(--breva-text-medium);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--breva-bg-medium);
        border-right: 1px solid var(--breva-border);
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] h2 {
        color: var(--breva-text-light);
        font-size: 1.4rem;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.1rem;
        color: var(--breva-primary-light);
        margin-bottom: 0.8rem;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--breva-text-medium);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    [data-testid="stSidebar"] ol, [data-testid="stSidebar"] ul {
        color: var(--breva-text-medium);
        padding-left: 1.2rem;
    }

    /* Custom divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(75, 85, 99, 0), rgba(75, 85, 99, 1) 50%, rgba(75, 85, 99, 0));
        margin: 1.5rem 0;
        border: none;
    }

    /* Button styling */
    .stButton button {
        background-color: var(--breva-primary) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.025em !important;
        transition: all var(--transition-speed) ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }

    .stButton button:hover {
        background-color: var(--breva-primary-light) !important;
        box-shadow: 0 4px 8px rgba(67, 97, 238, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    .stButton button:active {
        transform: translateY(1px) !important;
        box-shadow: 0 1px 2px rgba(67, 97, 238, 0.4) !important;
    }

    /* Download button styling */
    .stDownloadButton button {
        background-color: var(--breva-secondary) !important;
        color: var(--breva-bg-dark) !important;
        font-weight: 600 !important;
    }

    .stDownloadButton button:hover {
        background-color: #71D5F5 !important;
    }

    /* Status area styling */
    .status-container {
        background-color: var(--breva-bg-medium);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--breva-border);
        box-shadow: var(--card-shadow);
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
        color: var(--breva-primary-light) !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--breva-text-medium) !important;
    }

    /* Advanced chat styling */
    .chat-area {
        background-color: var(--breva-bg-medium);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--breva-border);
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }

    /* Enhanced message bubbles */
    .message-container {
        display: flex;
        width: 100%;
        margin-bottom: 16px;
        position: relative;
        animation: fadeIn 0.3s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-container {
        justify-content: flex-end;
    }

    .assistant-container {
        justify-content: flex-start;
    }

    .message-bubble {
        padding: 12px 16px;
        border-radius: 16px;
        max-width: 80%;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        position: relative;
        line-height: 1.5;
    }

    .user-bubble {
        background-color: var(--user-bubble-color);
        color: white;
        border-bottom-right-radius: 4px;
    }

    .assistant-bubble {
        background-color: var(--assistant-bubble-color);
        color: var(--breva-text-light);
        border-bottom-left-radius: 4px;
    }

    .message-bubble p {
        margin: 0 0 8px 0;
    }

    .message-bubble p:last-child {
        margin-bottom: 0;
    }

    .message-bubble ul, .message-bubble ol {
        margin: 8px 0;
        padding-left: 20px;
    }

    .message-bubble strong {
        color: var(--breva-secondary);
        font-weight: 600;
    }

    .message-time {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 4px;
        text-align: right;
    }

    /* Chat input styling */
    .stChatInputContainer {
        padding-bottom: 1rem !important;
        background-color: var(--breva-bg-dark) !important;
    }

    .stChatInputContainer > div {
        background-color: var(--breva-bg-medium) !important;
        border-radius: 12px !important;
        border: 1px solid var(--breva-border) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
        padding: 4px !important;
    }

    [data-testid="stChatInput"] {
        background-color: var(--breva-bg-light) !important;
        border-radius: 12px !important;
        border: 1px solid var(--breva-border) !important;
        color: var(--breva-text-light) !important;
        font-size: 1rem !important;
    }

    [data-testid="stChatButton"] {
        background-color: var(--breva-primary) !important;
        border-radius: 50% !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        transition: all var(--transition-speed) ease !important;
    }

    [data-testid="stChatButton"]:hover {
        background-color: var(--breva-primary-light) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
    }

    /* Custom styled components */
    .breva-card {
        background-color: var(--breva-bg-medium);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--breva-border);
        box-shadow: var(--card-shadow);
        transition: all var(--transition-speed) ease;
    }

    .breva-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    .breva-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        background-color: var(--breva-primary);
        color: white;
    }

    .breva-badge.secondary {
        background-color: var(--breva-secondary);
        color: var(--breva-bg-dark);
    }

    .breva-badge.accent {
        background-color: var(--breva-accent);
    }

    /* Footer styling */
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        padding-bottom: 1rem;
        border-top: 1px solid var(--breva-border);
        text-align: center;
        font-size: 0.8rem;
        color: var(--breva-text-dark);
    }

    /* Improve mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .message-bubble {
            max-width: 90%;
        }
        
        h1 {
            font-size: 1.5rem;
        }
        
        .stButton button {
            padding: 0.5rem 1rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def custom_chat_message(role: str, content: str, timestamp: str | None = None, is_html: bool = False):
    """
    Purpose: Display a custom chat message with enhanced styling.
    Input:
    - role: The role of the message sender (user or assistant).
    - content: The message content.
    - timestamp: The timestamp for the message (optional).
    - is_html: Whether the content is HTML (default: False).
    Output: None
    """
    # Here we set the current time if no timestamp is provided
    current_time = timestamp or datetime.now().strftime("%I:%M %p")
    
    # If the content is HTML, we use textwrap.dedent to remove leading whitespace
    # Otherwise, we escape the content to prevent HTML injection
    if is_html:
        content = textwrap.dedent(content).strip()
    else:
        content = html.escape(content)

    # If the role is user, we set the bubble to user-container
    if role == "user":
        bubble = f"""
        <div class="message-container user-container">
            <div class="message-bubble user-bubble">
                {content}
                <div class="message-time">{current_time}</div>
            </div>
        </div>"""
    else:
        bubble = f"""
        <div class="message-container assistant-container">
            <div class="message-bubble assistant-bubble">
                {content}
                <div class="message-time">{current_time}</div>
            </div>
        </div>"""

    # We use streamlit.markdown to display the message bubble
    st.markdown(textwrap.dedent(bubble), unsafe_allow_html=True)

def create_breva_card(title, content, icon=None):
    """
    Purpose: Create a custom Breva card with a title, content, and optional icon.
    Input:
    - title: The title of the card.
    - content: The content of the card.
    - icon: The Font Awesome icon name 
    Output: None
    """
    icon_html = f'<i class="fas fa-{icon}"></i> ' if icon else ''
    
    # Streamlit markdown for the card
    st.markdown(f"""
    <div class="breva-card">
        <h3>{icon_html}{title}</h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_header():
    """
    Purpose: Create a custom header with the app title and status.
    Input: None
    Output: None
    """

    # Create a header with two columns
    col1, col2 = st.columns([3, 1])
    
    # column 1: Title and badges
    with col1:
        st.title("Thrive Grant Applicant Insights")
        st.markdown('<div class="breva-badge">AI-Powered</div> <div class="breva-badge secondary">VOC Analytics</div>', unsafe_allow_html=True)
    
    # column 2: Status and date
    with col2:
        st.markdown(f"""
        <div style="text-align: right;">
            <p><strong>Status:</strong> <span style="color: #4CC9F0;">●</span> Active</p>
            <p><strong>Date:</strong> {datetime.now().strftime('%b %d, %Y')}</p>
        </div>
        """, unsafe_allow_html=True)

def create_sidebar():
    """
    Purpose: Create a custom sidebar with the app logo, title, and session stats.
    Input: None
    Output: None
    """

    # Define the sidebar layout
    with st.sidebar:
        # Breva image
        st.image("https://github.com/sribreva/Breva_VOC_Chat/raw/main/Breva.jpeg", width=200)

        st.title("Thrive Grant Insights")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # The about section 
        st.subheader("About this tool")
        st.markdown("""
        <i class="fas fa-chart-pie" style="color: var(--breva-primary);"></i> This tool helps Breva employees analyze Thrive Grant applications by providing data-driven insights from our Voice of Customer database.

        <div style="margin-top: 1rem;"><i class="fas fa-lightbulb" style="color: var(--breva-secondary);"></i> <strong>How to use:</strong></div>
        <ol>
            <li>Type your question about SMB grant applications</li>
            <li>Review the AI-generated insights</li>
            <li>Export conversations for reporting</li>
        </ol>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.subheader("Session Stats")
        col1, col2 = st.columns(2)
        
        # Display session stats
        with col1:
            st.metric("Questions", st.session_state.query_count)
        with col2:
            # Calculate session duration
            if "session_start_time" not in st.session_state:
                st.session_state.session_start_time = time.time()
            
            duration_mins = int((time.time() - st.session_state.session_start_time) / 60)
            st.metric("Duration", f"{duration_mins} min")
        
        st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Action buttons with improved styling
        st.subheader("Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                st.session_state.query_count = 0
                st.session_state.session_id = f"session_{int(time.time())}"
                st.session_state.session_start_time = time.time()
                st.rerun()
        
        with col2:
            if st.session_state.conversation_started:
                # We use the download_chat_history function to get the chat history
                chat_history = download_chat_history()
                if chat_history:
                    st.download_button(
                        label="💾 Export",
                        data=chat_history,
                        file_name=f"breva_chat_{st.session_state.session_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )


def display_messages():
    """
    Purpose: Display the chat messages in a custom chat area.
    Input: None
    Output: None
    """
    # Here we use the streamlit.container() to create a container for the chat area
    chat_container = st.container()
    
    # With teh chat_container, we create a custom chat area
    with chat_container:
        # We use the markdown to create a custom chat area
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        
        # For each index, message in the session state messages, we display the message
        for idx, message in enumerate(st.session_state.messages):
            # Time stamp for the message
            timestamp = (datetime.now() - timedelta(minutes=len(st.session_state.messages) - idx)).strftime("%I:%M %p")
            # Here we use the custom_chat_message function to display the message
            custom_chat_message(message["role"], message["content"], timestamp)
        
        st.markdown('</div>', unsafe_allow_html=True)


def display_welcome_message():
    """
    Purpose: Display a welcome message in the chat area.
    Input: None
    Output: None
    """

    # Here we use the streamlit textwrap.dedent to create a welcome message
    welcome_message = textwrap.dedent("""
    <h3 style="color: var(--breva-primary-light);">
        👋 Welcome to the Breva Thrive Grant Insights tool!
    </h3>

    <p>I can help you analyze applications by providing data‑driven insights on:</p>
    <ul>
        <li><strong style="color: var(--breva-secondary);">Financial challenges</strong> faced by applicants</li>
        <li><strong style="color: var(--breva-secondary);">Business goals</strong> and growth strategies</li>
        <li><strong style="color: var(--breva-secondary);">Funding needs</strong> and intended use of grants</li>
        <li><strong style="color: var(--breva-secondary);">Community impact</strong> of applicant businesses</li>
        <li><strong style="color: var(--breva-secondary);">Equity and inclusion</strong> efforts by applicants</li>
    </ul>

    <p>Ask me a question to get started with your data exploration!</p>
    """)

    # Here we use our custom
    custom_chat_message("assistant", welcome_message, is_html=True)

def create_status_area():
    """
    Purpose: Create a custom status area with database and AI model information.
    Input: None
    Output: None
    """

    # With the streamlit container, we create a custom status area
    with st.container():
        st.markdown('<div class="status-container">', unsafe_allow_html=True)
        
        cols = st.columns([2, 1, 1])
        
        with cols[0]:
            st.markdown("""
            <h3><i class="fas fa-database" style="color: var(--breva-primary);"></i> AI-powered VOC Analysis</h3>
            <p>Ask questions about financial challenges, business goals, or funding needs of grant applicants. The AI will analyze patterns and provide data-driven insights.</p>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background-color: var(--breva-bg-light); border-radius: 8px;">
                <div style="font-size: 0.8rem; color: var(--breva-text-medium);">DATABASE STATUS</div>
                <div style="font-size: 1.2rem; color: var(--breva-secondary);">CONNECTED</div>
                <div style="font-size: 0.8rem; color: var(--breva-text-medium);">Thrive Grant Q2 2025</div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; background-color: var(--breva-bg-light); border-radius: 8px;">
                <div style="font-size: 0.8rem; color: var(--breva-text-medium);">AI MODEL</div>
                <div style="font-size: 1.2rem; color: var(--breva-secondary);">CLAUDE 3.7</div>
                <div style="font-size: 0.8rem; color: var(--breva-text-medium);">Sonnet (2025)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_custom_chat_input():
    """
    Purpose: Create a custom chat input area with enhanced styling.
    Input: None
    Output: None
    """
    st.markdown("""
    <div class="search-container">
        <i class="fas fa-search search-icon"></i>
        <input type="text" id="custom-input" class="search-input" placeholder="Ask a question about Thrive Grant applicants...">
    </div>
    
    <script>
        const input = document.getElementById('custom-input');
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                // This would need to be connected to Streamlit
                console.log('Input value:', input.value);
                input.value = '';
            }
        });
    </script>
    """, unsafe_allow_html=True)

def create_footer():
    """
    Purpose: Create a custom footer with app information and copyright.
    Input: None
    Output: None
    """
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div>Breva Thrive Grant Insights Tool</div>
            <div>Internal Use Only</div>
            <div>© Breva 2025</div>
        </div>
        <div style="font-size: 0.7rem; opacity: 0.7;">
            Powered by Claude 3.7 | Version 2.4.1 | Last updated: April 15, 2025
        </div>
    </div>
    """, unsafe_allow_html=True)

def process_user_message(user_input):
    """
    Purpose: Process the user's input message and generate a response.
    Input: user_input: The user's input message.
    Output: None
    """
    # Here we use the add_user_message function to add the user message to the session state
    add_user_message(user_input)
    
    # Here we use the custom_chat_message function to display the user message
    custom_chat_message("user", user_input) 
    
    # Type indicator for the assistant when typing
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="message-container assistant-container">
        <div class="message-bubble assistant-bubble" style="display: inline-block;">
            <div class="typing-indicator">
                <span>●</span><span>●</span><span>●</span>
            </div>
        </div>
    </div>
    <style>
    .typing-indicator {
        display: flex;
        gap: 4px;
    }
    .typing-indicator span {
        animation: typing-bounce 1s infinite ease-in-out;
    }
    .typing-indicator span:nth-child(1) { animation-delay: 0s; }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Here we generate the response 
    try:
        # Simulate typing delay
        time.sleep(0.5)
        
        # Generate the answer using the querier
        answer = st.session_state.querier.generate_answer(user_input, st.session_state.messages[:-1])
        # Here we use the add_assistant_message function to add the assistant message to the session state
        add_assistant_message(answer)
        
        # Remove typing indicator and show the response
        typing_placeholder.empty()

        # Here we use the custom_chat_message function to display the assistant message
        custom_chat_message("assistant", answer)
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
        # Here we use the add_assistant_message function to add the error message to the session state
        add_assistant_message(error_msg)
        
        # Remove typing indicator and show the error
        typing_placeholder.empty()
        # Here we use the same custom_chat_message function to display the error message
        custom_chat_message("assistant", error_msg)



def add_user_message(user_input):
    """
    Purpose: Add a user message to the chat history.
    Input: user_input: The user's input message.
    Output: None
    """
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_started = True
    st.session_state.query_count += 1

def add_assistant_message(content):
    """
    Purpose: Add an assistant message to the chat history.
    Input: content: The assistant's response content.
    Output: None
    """
    st.session_state.messages.append({"role": "assistant", "content": content})


# ------------------------------------------------------------------------------
# Main Streamlit Application
# ------------------------------------------------------------------------------

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Breva Thrive Insights",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Here we initialize the session state
    initialize_session_state()

    # Apply custom CSS for dark theme
    apply_custom_css()
    
    # Sidebar configuration
    create_sidebar()

    # Main content area
    st.title("Thrive Grant Applicant Insights")
    
    # Status area
    create_status_area()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Initialize querier
    if not initialize_querier():
        st.stop()
    
    # Chat interface
    chat_container = st.container()

    # With the chat_container, we create a custom chat area
    with chat_container:
        if not st.session_state.conversation_started:
            # Here we use the display_welcome_message function to display the welcome message
            display_welcome_message()
        else:
            # Display existing messages with custom styling
            display_messages()
    
    # Chat input
    user_input = st.chat_input("Ask a question about Thrive Grant applicants...")

    # If there is a user input, we process the message
    if user_input:
        process_user_message(user_input)
    
    # Footer
    create_footer()

if __name__ == "__main__":
    main()
