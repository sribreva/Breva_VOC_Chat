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
from datetime import datetime

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
        
        # Validate API keys
        if not anthropic_api_key:
            raise ValueError("Anthropic API key not provided")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key not provided")
            
        self.api_key = anthropic_api_key

        # Connection to Pinecone - using updated API
        logging.info("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.index = self.pc.Index(self.index_name)
        
        # Get index stats to confirm connection
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

        # Question Types as defined in VOC_map_reduce.py
        self.question_types = {
            # Financial Challenges
            "financial_challenges_1": {
                "context": "What specific challenges do you face in managing and forecasting your cash flow?",
                "columns": ["What specific challenges do you face in managing and forecasting your cash flow?"]
            },
            "financial_challenges_2": {
                "context": "What specific financial tasks consume most of your time?",
                "columns": ["What specific financial tasks consume most of your time, and how do you feel these tasks impact your ability to focus on growing your business?"]
            },
            "financial_challenges_3": {
                "context": "Tell us about a hard instance managing finances or getting a loan",
                "columns": ["Please tell us about a recent instance where it was really hard for you to manage your finances, or to get financial help, such as a loan. What would have been the ideal solution?"]
            },
            "financial_challenges_4": {
                "context": "Challenges with applying for loans",
                "columns": ["What are the most significant challenges you face with applying for loans, and what do you wish you could improve?"]
            },

            # Business Description
            "desc_business_brief": {
                "context": "A brief description of the business",
                "columns": [
                    "Provide a brief description of your business",
                    "Provide a brief description of your business. Include a description of your products/services"
                ]
            },
            "desc_primary_products": {
                "context": "Primary products/services offered",
                "columns": ["Detail the primary products/services offered by your business"]
            },
            "desc_community_impact": {
                "context": "Impact on the community",
                "columns": ["Describe how your business positively impacts your community"]
            },
            "desc_equity_inclusion": {
                "context": "Efforts to promote equity and inclusion",
                "columns": ["Describe efforts made by your business to promote equity and inclusion in the workplace and community"]
            },

            # Business Goals and Growth
            "business_goals_1": {
                "context": "Achievements and business goals",
                "columns": [
                    "What significant achievements have you made in your business? What are your business goals for the coming year?",
                    "What significant achievements have you made in your business? What are your business goals for the next 12 months?"
                ]
            },
            "business_goals_2": {
                "context": "Daily tasks for a virtual CFO",
                "columns": ["If there were no constraints, what tasks would you want an advanced technology like a virtual Chief Financial Officer to handle for you daily?"]
            },

            # Financial Tools and Advisory
            "financial_tool_needs": {
                "context": "Required features for financial management tool",
                "columns": [
                    "What key features do you need in a tool to better manage your cash and build your business credit? What is (or would be) your budget for such a solution?",
                    "What key features do you need in a tool to better manage your cash and expenses? What is (or would be) your budget for such a solution?"
                ]
            },

            # Grant and Support
            "grant_usage": {
                "context": "How grant funds will be used",
                "columns": [
                    "Provide a brief statement detailing your financial need for this grant and how the funds will be used to enhance community impact",
                    "Provide a brief statement detailing how the funds will be used to enhance community impact"
                ]
            },

            # Business Challenges
            "business_obstacles": {
                "context": "Major business obstacles and solutions",
                "columns": ["Describe major obstacles your company encountered and how you resolved them"]
            },

            # Additional Context
            "additional_context": {
                "context": "Additional relevant information",
                "columns": ["Please include any relevant information or context that you believe would be helpful for the judges to consider when reviewing your application"]
            },

            # Financial Advisor Questions
            "financial_advisor_questions": {
                "context": "Questions for financial advisor",
                "columns": ["Please provide your top three (3) questions you would ask a financial advisor or business coach, about your business?"]
            },

            # Financial assistance rationale
            "reason_financial_assistance": {
                "context": "What is your main reason for seeking financial assistance for your business?",
                "columns": ["What is your main reason for seeking financial assistance for your business?"]
            },

            # Planning responsibility
            "financial_planning_responsible": {
                "context": "Who handles the financial planning and cash flow tracking at your business?",
                "columns": ["Who handles the financial planning and cash flow tracking at your business?"]
            }
        }

    def determine_question_type(self, user_query: str) -> str:
        """
        Determine the question type based on the user query using keyword matching.
        """
        user_query = user_query.lower()
        
        scores = {}
        for qtype, info in self.question_types.items():
            score = 0
            for keyword in info["keywords"]:
                if keyword.lower() in user_query:
                    score += 1
            scores[qtype] = score
        
        # Get the question type with the highest score
        best_type = max(scores.items(), key=lambda x: x[1])
        
        # If no keywords matched, default to financial_challenges_1
        if best_type[1] == 0:
            logging.info("[determine_question_type] No keywords matched. Using default question type: financial_challenges_1")
            return "financial_challenges_1"
        
        logging.info(f"[determine_question_type] Query mapped to '{best_type[0]}' (score={best_type[1]})")
        return best_type[0]

    def get_offline_summary(self, question_type: str) -> str:
        """
        Fetch the offline summary for a given question type from Pinecone.
        """
        summary_qtype = f"{question_type}_summary"
        logging.info(f"Fetching offline summary for question_type='{summary_qtype}'")
        
        try:
            # For Pinecone, we provide a dummy vector for the query.
            dummy_vector = [0.0] * 384  # Adjust dimension to your index if needed
            
            query_results = self.index.query(
                vector=dummy_vector,
                filter={"question_type": summary_qtype},
                top_k=1,
                include_metadata=True
            )
            
            matches = getattr(query_results, 'matches', [])
            if not matches:
                logging.warning(f"No summary found for {summary_qtype}")
                return ""
            
            summary_text = matches[0].metadata.get("text", "")
            if not summary_text:
                logging.warning(f"No text found in metadata for {summary_qtype}")
                return ""
                
            logging.info(f"Successfully retrieved summary for {summary_qtype}")
            return summary_text
            
        except Exception as e:
            logging.error(f"Error fetching summary for {summary_qtype}: {e}")
            return ""

def build_prompt_with_offline_summary(self, user_query: str, summary: str, conversation_history: List[Dict[str, str]]) -> str:
    """
    Build a prompt with the user query, offline summary, and conversation history.
    """
    # Build conversation history string
    conversation_history_text = ""
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
        Generate an answer using Anthropic based on the user query and the full conversation history.
        """
        try:
            # Determine the question type
            qtype = self.determine_question_type(user_query)
            
            # Fetch the offline summary for the question type
            offline_summary = self.get_offline_summary(qtype)
            if not offline_summary:
                return (f"No offline summary found for question type '{qtype}'. "
                        f"Try a different approach or run offline summarization first.")
            
            # Build the final prompt including the conversation history
            final_prompt = self.build_prompt_with_offline_summary(user_query, offline_summary, conversation_history)
            logging.info("FINAL PROMPT constructed.")
    
            # Call Anthropic to generate the final answer
            response = self.anthropic.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=8192,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": final_prompt
                }]
            )
            
            # Handle response content extraction
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
    """Initialize session state variables if they don't exist"""
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
    """Initialize the VOCDatabaseQuerier and store it in session state"""
    if "querier" not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                pinecone_api_key = st.secrets.get("pinecone_api_key")
                anthropic_api_key = st.secrets.get("anthropic_api_key")
                
                if not pinecone_api_key or not anthropic_api_key:
                    st.error("API keys not found in Streamlit secrets. Please add them in the Streamlit Cloud dashboard.")
                    st.info("Go to your app settings in Streamlit Cloud, navigate to 'Secrets', and add 'pinecone_api_key' and 'anthropic_api_key'.")
                    return False
                
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
    """Generate a downloadable file with the chat history"""
    if not st.session_state.messages:
        return None
    
    chat_export = {"session_id": st.session_state.session_id, "timestamp": datetime.now().isoformat(), "messages": st.session_state.messages}
    return json.dumps(chat_export, indent=2)

def apply_custom_css():
    """Apply custom CSS for the dark theme iMessage-style chat"""
    st.markdown("""
    <style>
        /* Dark theme base styling */
        :root {
            --background-color: #1E1E1E;
            --text-color: #E0E0E0;
            --accent-color: #6B46C1;
            --secondary-color: #2D2D2D;
            --border-color: #444444;
            --user-bubble-color: #145EAB;
            --assistant-bubble-color: #2D2D2D;
        }
        
        body {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Override Streamlit defaults */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1000px;
        }
        
        /* iMessage-style chat bubbles */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin-bottom: 12px !important;
        }
        
        /* Hide the default chat message icons */
        .stChatMessage [data-testid="chatAvatarIcon-user"],
        .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
            display: none !important;
        }
        
        /* Custom message bubble styling */
        .message-container {
            display: flex;
            width: 100%;
            margin-bottom: 16px;
        }
        
        .user-container {
            justify-content: flex-end;
        }
        
        .assistant-container {
            justify-content: flex-start;
        }
        
        .message-bubble {
            padding: 10px 16px;
            border-radius: 18px;
            max-width: 80%;
            margin: 0;
        }
        
        .user-bubble {
            background-color: var(--user-bubble-color);
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .assistant-bubble {
            background-color: var(--assistant-bubble-color);
            color: var(--text-color);
            border-bottom-left-radius: 5px;
        }
        
        /* Remove padding from message containers */
        .stChatMessage > div:first-child {
            padding: 0 !important;
        }

        /* Proper spacing for messages */
        .message-container + .message-container {
            margin-top: 8px;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--secondary-color);
            border-right: 1px solid var(--border-color);
        }
        
        /* Input area styling */
        [data-baseweb="input"] {
            border-radius: 20px !important;
            background-color: var(--secondary-color) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Button styling */
        .stButton button {
            background-color: var(--accent-color) !important;
            color: white !important;
            border-radius: 20px !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton button:hover {
            background-color: #805AD5 !important;
            box-shadow: 0 4px 8px rgba(107, 70, 193, 0.3) !important;
        }
        
        /* Status area styling */
        .status-area {
            background-color: var(--secondary-color);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-color);
        }
        
        /* Footer styling */
        .footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 0.8rem;
            color: #888;
        }
        
        /* Custom divider */
        .custom-divider {
            border-top: 1px solid var(--border-color);
            margin: 1.5rem 0;
        }
        
        /* Override Streamlit chat input styling */
        .stChatInputContainer {
            padding-bottom: 1rem !important;
            background-color: var(--background-color) !important;
        }
        
        .stChatInputContainer > div {
            background-color: var(--secondary-color) !important;
            border-radius: 20px !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Hide default chat message container styling */
        .stChatMessageContent {
            background-color: transparent !important;
            padding: 0 !important;
            border-radius: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def custom_chat_message(role, content):
    """Display a custom chat message with iMessage-style bubbles"""
    # Clean content by removing any HTML tags that might be in the content string
    
    if role == "user":
        st.markdown(f"""
        <div class="message-container user-container">
            <div class="message-bubble user-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # For assistant messages, wrap content in proper markdown formatting
        # This ensures markdown is rendered properly within the bubble
        content_div = f'<div class="message-bubble assistant-bubble">{content}</div>'
        st.markdown(f"""
        <div class="message-container assistant-container">
            {content_div}
        </div>
        """, unsafe_allow_html=True)

def display_messages():
    """Display all messages in the chat history with custom styling"""
    for message in st.session_state.messages:
        custom_chat_message(message["role"], message["content"])

def add_user_message(user_input):
    """Add a user message to the chat history"""
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_started = True
    st.session_state.query_count += 1

def add_assistant_message(content):
    """Add an assistant message to the chat history"""
    st.session_state.messages.append({"role": "assistant", "content": content})

def process_user_message(user_input):
    """Process a user message and generate a response"""
    # Add user message to chat history
    add_user_message(user_input)
    
    # Display user message with custom styling
    custom_chat_message("user", user_input)
    
    # Generate assistant response
    with st.spinner("Thinking..."):
        try:
            answer = st.session_state.querier.generate_answer(user_input, st.session_state.messages[:-1])
            add_assistant_message(answer)
            custom_chat_message("assistant", answer)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            add_assistant_message(error_msg)
            custom_chat_message("assistant", error_msg)

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
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS for dark theme
    apply_custom_css()
    
    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Breva", width=150)
        st.title("Thrive Grant Insights")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # App information and instructions
        st.subheader("About this tool")
        st.markdown("""
        This tool helps Breva employees analyze Thrive Grant applications by providing insights from our Voice of Customer database.

        **How to use:**
        1. Type your question about SMB grant applications
        2. Review the AI-generated insights
        3. Export conversations for reporting
        """)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Session stats
        st.subheader("Session Stats")
        st.metric("Questions Asked", st.session_state.query_count)
        st.metric("Session ID", st.session_state.session_id)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_started = False
                st.session_state.query_count = 0
                st.session_state.session_id = f"session_{int(time.time())}"
                st.rerun()
        
        with col2:
            if st.session_state.conversation_started:
                chat_history = download_chat_history()
                if chat_history:
                    st.download_button(
                        label="Export",
                        data=chat_history,
                        file_name=f"breva_chat_{st.session_state.session_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
    # Main content area
    st.title("Thrive Grant Applicant Insights")
    
    # Status area
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown("### AI-powered analysis of SMB grant applications")
            st.markdown("Ask questions about financial challenges, business goals, or funding needs of applicants.")
        with cols[1]:
            st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
            st.markdown(f"**Status:** {'Active' if initialize_querier() else 'Error'}")
            st.markdown(f"**Date:** {datetime.now().strftime('%b %d, %Y')}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Initialize querier
    if not initialize_querier():
        st.stop()
    
    # Chat interface
    chat_container = st.container()
    with chat_container:
        if not st.session_state.conversation_started:
            # Welcome message for new conversations
            welcome_message = """
👋 Welcome to the Breva Thrive Grant Insights tool!

I can help you analyze applications by providing insights on:

- **Financial challenges** faced by applicants
- **Business goals** and growth strategies
- **Funding needs** and intended use of grants
- **Community impact** of applicant businesses
- **Equity and inclusion** efforts by applicants

How can I assist you today?
            """
            custom_chat_message("assistant", welcome_message)
        else:
            # Display existing messages with custom styling
            display_messages()
    
    # Chat input
    user_input = st.chat_input("Ask a question about Thrive Grant applicants...")
    if user_input:
        process_user_message(user_input)
    
    # Footer
    st.markdown("""
    <div class="footer">
        Breva Thrive Grant Insights Tool | Internal Use Only | © Breva 2025
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
