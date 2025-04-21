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
import pandas as pd
import altair as alt
from PIL import Image
import base64
import io
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------------------
# VOCDatabaseQuerier Class - Remains unchanged 
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

        # Question Types with added keywords for matching
        self.question_types = {
            # Financial Challenges
            "financial_challenges_1": {
                "context": "What specific challenges do you face in managing and forecasting your cash flow?",
                "columns": ["What specific challenges do you face in managing and forecasting your cash flow?"],
                "keywords": ["cash flow", "forecast", "managing cash", "forecasting", "cashflow", "cash management"]
            },
            "financial_challenges_2": {
                "context": "What specific financial tasks consume most of your time?",
                "columns": ["What specific financial tasks consume most of your time, and how do you feel these tasks impact your ability to focus on growing your business?"],
                "keywords": ["time consuming", "financial tasks", "time spent", "financial processes", "time-consuming"]
            },
            "financial_challenges_3": {
                "context": "Tell us about a hard instance managing finances or getting a loan",
                "columns": ["Please tell us about a recent instance where it was really hard for you to manage your finances, or to get financial help, such as a loan. What would have been the ideal solution?"],
                "keywords": ["loan challenges", "financing difficulty", "loan problem", "financial help", "hard to get loan"]
            },
            "financial_challenges_4": {
                "context": "Challenges with applying for loans",
                "columns": ["What are the most significant challenges you face with applying for loans, and what do you wish you could improve?"],
                "keywords": ["loan application", "loan process", "applying for loans", "loan paperwork", "borrowing money"]
            },

            # Business Description
            "desc_business_brief": {
                "context": "A brief description of the business",
                "columns": [
                    "Provide a brief description of your business",
                    "Provide a brief description of your business. Include a description of your products/services"
                ],
                "keywords": ["business type", "company overview", "what they do", "business description", "company description"]
            },
            "desc_primary_products": {
                "context": "Primary products/services offered",
                "columns": ["Detail the primary products/services offered by your business"],
                "keywords": ["products", "services", "offerings", "what they sell", "main product"]
            },
            "desc_community_impact": {
                "context": "Impact on the community",
                "columns": ["Describe how your business positively impacts your community"],
                "keywords": ["community", "impact", "local area", "social impact", "community engagement"]
            },
            "desc_equity_inclusion": {
                "context": "Efforts to promote equity and inclusion",
                "columns": ["Describe efforts made by your business to promote equity and inclusion in the workplace and community"],
                "keywords": ["diversity", "equity", "inclusion", "dei", "diverse", "inclusive"]
            },

            # Business Goals and Growth
            "business_goals_1": {
                "context": "Achievements and business goals",
                "columns": [
                    "What significant achievements have you made in your business? What are your business goals for the coming year?",
                    "What significant achievements have you made in your business? What are your business goals for the next 12 months?"
                ],
                "keywords": ["goals", "achievements", "milestones", "future plans", "objectives", "targets"]
            },
            "business_goals_2": {
                "context": "Daily tasks for a virtual CFO",
                "columns": ["If there were no constraints, what tasks would you want an advanced technology like a virtual Chief Financial Officer to handle for you daily?"],
                "keywords": ["cfo", "financial officer", "virtual cfo", "financial management", "finance automation"]
            },

            # Financial Tools and Advisory
            "financial_tool_needs": {
                "context": "Required features for financial management tool",
                "columns": [
                    "What key features do you need in a tool to better manage your cash and build your business credit? What is (or would be) your budget for such a solution?",
                    "What key features do you need in a tool to better manage your cash and expenses? What is (or would be) your budget for such a solution?"
                ],
                "keywords": ["tool features", "software needs", "financial tools", "management software", "financial platform"]
            },

            # Grant and Support
            "grant_usage": {
                "context": "How grant funds will be used",
                "columns": [
                    "Provide a brief statement detailing your financial need for this grant and how the funds will be used to enhance community impact",
                    "Provide a brief statement detailing how the funds will be used to enhance community impact"
                ],
                "keywords": ["grant usage", "fund usage", "how will they use", "spend the money", "use the funds"]
            },

            # Business Challenges
            "business_obstacles": {
                "context": "Major business obstacles and solutions",
                "columns": ["Describe major obstacles your company encountered and how you resolved them"],
                "keywords": ["obstacles", "challenges", "problems", "hurdles", "difficulties", "barriers"]
            },

            # Additional Context
            "additional_context": {
                "context": "Additional relevant information",
                "columns": ["Please include any relevant information or context that you believe would be helpful for the judges to consider when reviewing your application"],
                "keywords": ["other information", "additional info", "other context", "judges", "review"]
            },

            # Financial Advisor Questions
            "financial_advisor_questions": {
                "context": "Questions for financial advisor",
                "columns": ["Please provide your top three (3) questions you would ask a financial advisor or business coach, about your business?"],
                "keywords": ["advisor questions", "financial advice", "business coach", "consult", "questions to ask"]
            },

            # Financial assistance rationale
            "reason_financial_assistance": {
                "context": "What is your main reason for seeking financial assistance for your business?",
                "columns": ["What is your main reason for seeking financial assistance for your business?"],
                "keywords": ["assistance reason", "why need money", "reason for funds", "need funding", "financial need"]
            },

            # Planning responsibility
            "financial_planning_responsible": {
                "context": "Who handles the financial planning and cash flow tracking at your business?",
                "columns": ["Who handles the financial planning and cash flow tracking at your business?"],
                "keywords": ["who handles", "who manages", "responsibility", "financial planner", "accountant", "bookkeeper"]
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
# Helper functions for the enhanced Streamlit app
# ------------------------------------------------------------------------------

def get_base64_of_bin_file(bin_file):
    """Get base64 encoding of binary file for embedding in CSS"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def generate_dummy_chart_data():
    """Generate dummy chart data for the dashboard"""
    # Categories of challenges
    categories = ["Cash Flow Management", "Financial Planning", 
                  "Access to Capital", "Tax Compliance", 
                  "Expense Tracking", "Invoicing/Billing"]
    
    # Generate dummy data
    data = []
    for category in categories:
        data.append({
            "Category": category,
            "Percentage": np.random.randint(20, 80)
        })
    
    return pd.DataFrame(data)

def generate_dummy_industry_data():
    """Generate dummy industry distribution data"""
    industries = ["Retail", "Food & Beverage", "Professional Services", 
                 "Healthcare", "Construction", "Technology", "Other"]
    
    data = []
    for industry in industries:
        data.append({
            "Industry": industry,
            "Count": np.random.randint(10, 100)
        })
    
    df = pd.DataFrame(data)
    df["Percentage"] = (df["Count"] / df["Count"].sum() * 100).round(1)
    return df

def generate_dummy_location_data():
    """Generate dummy location data for map visualization"""
    states = ["California", "Texas", "New York", "Florida", 
              "Illinois", "Pennsylvania", "Ohio", "Georgia", 
              "North Carolina", "Michigan"]
    
    data = []
    for state in states:
        data.append({
            "State": state,
            "Applicants": np.random.randint(5, 50)
        })
    
    return pd.DataFrame(data)

def extract_percentages(text):
    """Extract percentage statistics from AI response for visualization"""
    percentage_pattern = r'(\d{1,3}(?:\.\d+)?)%'
    matches = re.findall(percentage_pattern, text)
    
    if not matches:
        return None
    
    # Get up to 5 percentage stats with made-up categories
    categories = ["Cash Flow Issues", "Manual Processes", "Loan Access", 
                 "Credit Building", "Tax Compliance", "Financial Planning", 
                 "Expense Tracking", "Inventory Management"]
    
    data = []
    for i, match in enumerate(matches[:5]):
        if i < len(categories):
            data.append({
                "Category": categories[i],
                "Percentage": float(match)
            })
    
    if data:
        return pd.DataFrame(data)
    return None

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    # Chat session variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
        
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
        
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time())}"
    
    # UI state variables
    if "current_view" not in st.session_state:
        st.session_state.current_view = "chat"  # Options: "chat", "dashboard", "settings"
    
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"  # Options: "light", "dark", "modern"
    
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False
    
    if "chart_data" not in st.session_state:
        st.session_state.chart_data = generate_dummy_chart_data()
    
    if "industry_data" not in st.session_state:
        st.session_state.industry_data = generate_dummy_industry_data()
    
    if "location_data" not in st.session_state:
        st.session_state.location_data = generate_dummy_location_data()
    
    if "insights_data" not in st.session_state:
        st.session_state.insights_data = None

def initialize_querier():
    """Initialize the VOCDatabaseQuerier and store it in session state"""
    if "querier" not in st.session_state:
        try:
            with st.spinner("Initializing AI chatbot..."):
                # In production, get these from Streamlit secrets
                # For demo purposes, we'll use placeholders
                pinecone_api_key = st.secrets.get("pinecone_api_key", "demo-api-key")
                anthropic_api_key = st.secrets.get("anthropic_api_key", "demo-api-key")
                
                # For demo mode with mock credentials
                if pinecone_api_key == "demo-api-key" or anthropic_api_key == "demo-api-key":
                    st.session_state.demo_mode = True
                    st.success("Running in demo mode with simulated responses.")
                    return True
                
                st.session_state.demo_mode = False
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
    
    chat_export = {
        "session_id": st.session_state.session_id, 
        "timestamp": datetime.now().isoformat(), 
        "messages": st.session_state.messages
    }
    return json.dumps(chat_export, indent=2)

def apply_custom_css():
    """Apply custom CSS based on the selected theme"""
    
    # Common CSS for all themes
    common_css = """
    /* Layout adjustments */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        position: relative;
        max-width: 85%;
    }
    
    .user-message {
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    
    .assistant-message {
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
    
    /* Navigation styling */
    .nav-link {
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        transition: all 0.2s;
    }
    
    .nav-link:hover {
        transform: translateX(5px);
    }
    
    .nav-icon {
        margin-right: 0.5rem;
    }
    
    /* Card styling */
    .stat-card {
        padding: 1.5rem;
        border-radius: 0.75rem;
        transition: all 0.3s;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    /* Header bar */
    .header-bar {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Chat input container */
    .chat-input-container {
        position: sticky;
        bottom: 1rem;
        padding: 1rem;
        border-radius: 1rem;
        margin-top: 1rem;
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        position: absolute;
        bottom: 0.3rem;
        opacity: 0.7;
    }
    
    .user-timestamp {
        right: 0.5rem;
    }
    
    .assistant-timestamp {
        left: 0.5rem;
    }
    
    /* Button styling */
    .icon-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem;
        border-radius: 50%;
        margin: 0 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .icon-button:hover {
        transform: scale(1.1);
    }
    
    /* Dashboard styling */
    .dashboard-container {
        padding: 1rem;
    }
    
    .chart-container {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Better markdown formatting */
    .prose h1 {
        font-size: 1.5rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .prose h2 {
        font-size: 1.25rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .prose h3 {
        font-size: 1.1rem !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .prose p {
        margin-bottom: 0.75rem !important;
    }
    
    .prose ul, .prose ol {
        margin-bottom: 0.75rem !important;
        padding-left: 1.5rem !important;
    }
    
    .prose blockquote {
        border-left: 3px solid;
        padding-left: 1rem;
        margin-left: 0;
        margin-right: 0;
        font-style: italic;
        margin-bottom: 0.75rem !important;
    }
    
    /* Animation for new messages */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .new-message {
        animation: fadeIn 0.3s ease-out forwards;
    }
    """
    
    # Theme-specific CSS
    if st.session_state.theme == "dark":
        theme_css = """
        /* Dark theme */
        :root {
            --bg-color: #121212;
            --text-color: #E0E0E0;
            --primary-color: #6B46C1;
            --secondary-color: #1E1E2D;
            --accent-color: #8B5CF6;
            --card-bg: #1E1E2D;
            --user-bubble-color: #6B46C1;
            --assistant-bubble-color: #2D2D3E;
            --border-color: #333340;
            --hover-color: #353545;
        }
        
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .chat-message {
            color: white;
        }
        
        .user-message {
            background-color: var(--user-bubble-color);
            color: white;
        }
        
        .assistant-message {
            background-color: var(--assistant-bubble-color);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }
        
        .stat-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
        }
        
        .chart-container {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
        }
        
        .header-bar {
            background-color: var(--secondary-color);
            border: 1px solid var(--border-color);
        }
        
        .nav-link {
            color: var(--text-color);
            background-color: var(--secondary-color);
        }
        
        .nav-link:hover {
            background-color: var(--hover-color);
        }
        
        .nav-link-active {
            background-color: var(--primary-color);
            color: white;
        }
        
        .chat-input-container {
            background-color: var(--secondary-color);
            border: 1px solid var(--border-color);
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            background-color: var(--card-bg);
            color: var(--text-color);
            border-color: var(--border-color);
        }
        
        /* Button styling */
        .stButton > button {
            background-color: var(--primary-color);
            color: white;
            border: none;
        }
        
        .stButton > button:hover {
            background-color: var(--accent-color);
        }
        
        .icon-button {
            background-color: var(--card-bg);
            color: var(--text-color);
        }
        
        .icon-button:hover {
            background-color: var(--hover-color);
        }
        
        /* Dashboard */
        .dashboard-container {
            background-color: var(--bg-color);
        }
        
        .prose blockquote {
            border-color: var(--accent-color);
        }
        """
    
    elif st.session_state.theme == "light":
        theme_css = """
        /* Light theme */
        :root {
            --bg-color: #F9FAFB;
            --text-color: #1F2937;
            --primary-color: #6366F1;
            --secondary-color: #F3F4F6;
            --accent-color: #4F46E5;
            --card-bg: #FFFFFF;
            --user-bubble-color: #6366F1;
            --assistant-bubble-color: #FFFFFF;
            --border-color: #E5E7EB;
            --hover-color: #EBEEF2;
        }
        
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .user-message {
            background-color: var(--user-bubble-color);
            color: white;
        }
        
        .assistant-message {
            background-color: var(--assistant-bubble-color);
