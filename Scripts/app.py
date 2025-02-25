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
You are an internal assistant for **Breva**, a company focused on understanding and supporting small and medium-sized businesses (SMBs). This chatbot is used **exclusively by Breva employees** to extract insights from data collected via our **Thrive Grant application**. 

Your goal is to help Breva employees analyze and interpret customer responses, so they can better understand the financial challenges, funding needs, and business goals of SMBs. This is **not a customer-facing tool**—your responses should focus on helping Breva employees gain actionable insights from the collected data.

### **Contextual Information**
To ensure continuity, here's the conversation so far:
---
{conversation_history_text}
---

The user just asked: **"{user_query}"**

To assist them, I'm providing a relevant background summary extracted from our **Voice of Customer (VOC) database**, which contains real SMB responses regarding their financial challenges, funding concerns, and business strategies:
---
{summary}
---

### **Response Guidelines**
1. **Frame Your Answer for Breva Employees**  
   - Assume the user is a Breva employee analyzing customer responses, not an SMB owner seeking advice.  
   - Focus on **what insights can be drawn from the provided data** rather than providing direct guidance to the customer.  
   - Please provide statistics when available as this makes it more understandable. 

2. **Use the Background Information Thoughtfully**  
   - Incorporate key insights from the summary without directly repeating them.  
   - Structure your response using bullet points or subheadings where helpful.  
   - Clearly distinguish between **data-driven insights** and **potential interpretations or implications**.

Now, using the above information, please craft a structured, insightful response tailored for **internal Breva employees analyzing customer data.**
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
                    index_name="voc-index",
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
    if role == "user":
        st.markdown(f"""
        <div class="message-container user-container">
            <div class="message-bubble user-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-container assistant-container">
            <div class="message-bubble assistant-bubble">{content}</div>
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
            custom_chat_message("assistant", """
            👋 Welcome to the Breva Thrive Grant Insights tool! 
            
            I can help you analyze applications by providing insights on:
            
            - **Financial challenges** faced by applicants
            - **Business goals** and growth strategies
            - **Funding needs** and intended use of grants
            - **Community impact** of applicant businesses
            - **Equity and inclusion** efforts by applicants
            
            How can I assist you today?
            """)
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
