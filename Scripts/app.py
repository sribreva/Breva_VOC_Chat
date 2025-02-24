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
import requests
from colorama import Fore, Style
import colorama
colorama.init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------------------
# VOCDatabaseQuerier Class Definition 
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
        logging.info(f"Connecting to Pinecone...")
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
            dummy_vector = [0.0] * 384  # Dimensionality as per your Pinecone index
            
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
You are a friendly, knowledgeable assistant. Here's our conversation so far:
{conversation_history_text}

The user just asked: "{user_query}"

I am providing you with some detailed background information to help answer the question:
---
{summary}
---

Please respond in a clear, structured, and conversational manner. Integrate key details from the background information into your answer and feel free to ask clarifying questions if needed.
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
                model="claude-3-5-sonnet-20241022",
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
# Streamlit Application
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Thrive Grant Application Chatbot", page_icon="🤖", layout="centered")
    
    # Basic CSS Styling for chat messages
    st.markdown(
        """
        <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
        }
        .message {
            padding: 10px 15px;
            border-radius: 15px;
            margin: 8px 0;
            width: fit-content;
            max-width: 80%;
            color: #000000;
            font-size: 16px;
            line-height: 1.4;
        }
        .user {
            background-color: #DCF8C6;
            align-self: flex-end;
        }
        .bot {
            background-color: #F1F0F0;
            align-self: flex-start;
        }
        .chat-box {
            display: flex;
            flex-direction: column;
        }
        textarea {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar instructions
    st.sidebar.title("Instructions")
    st.sidebar.info(
        "Ask any questions regarding the Thrive Grant Application. "
        "The chatbot will respond using our VOC offline summaries."
    )
    
    if not st.secrets.get("anthropic_api_key") or not st.secrets.get("pinecone_api_key"):
        st.sidebar.warning(
            "API keys not found in Streamlit secrets. "
            "Please add them in your Streamlit Cloud dashboard under Settings > Secrets."
        )

    st.title("Thrive Grant Application Chatbot 🤖")
    st.write("Have a conversation with the chatbot regarding the Thrive Grant Application questions.")

    # Initialize the VOCDatabaseQuerier and cache it in session state
    if "querier" not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                pinecone_api_key = st.secrets.get("pinecone_api_key")
                anthropic_api_key = st.secrets.get("anthropic_api_key")
                
                if not pinecone_api_key or not anthropic_api_key:
                    st.error("API keys not found in Streamlit secrets. Please add them in the Streamlit Cloud dashboard.")
                    st.info("Go to your app settings in Streamlit Cloud, navigate to 'Secrets', and add 'pinecone_api_key' and 'anthropic_api_key'.")
                    return
                
                st.session_state.querier = VOCDatabaseQuerier(
                    pinecone_api_key=pinecone_api_key,
                    index_name="voc-index",
                    anthropic_api_key=anthropic_api_key
                )
        except Exception as e:
            st.error(f"Error initializing the VOC Database Querier: {e}")
            return

    # Define memory to store chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []  
    
    # Chat container for displaying messages
    chat_container = st.container()

    with chat_container:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="message user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message bot"><strong>Chatbot:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Enter your query:", height=100, placeholder="Type your question here...")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()  

    # Process the latest user message if no bot response is yet available
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        latest_user_query = st.session_state.messages[-1]["content"]
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.querier.generate_answer(latest_user_query, st.session_state.messages)
                st.session_state.messages.append({"role": "bot", "content": answer})
            except Exception as e:
                st.session_state.messages.append({"role": "bot", "content": f"Error: {str(e)}"})
        st.rerun()  

if __name__ == "__main__":
    main()
