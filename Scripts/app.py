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
    3) Uses Anthropic to produce a final answer based on that summary.
    """

    def __init__(
        self,
        pinecone_api_key: str = "pcsk_5vnC9g_A8MYTbGufDu68CXWkiUCqPQY3bSLRULeJvSJEhxVNU8GHHfdMaYSjSAEKFETDAt",
        index_name: str = "voc-index-2025-q2",
        anthropic_api_key: str = None
        ):
        logging.info("Initializing VOC Database Querier (Offline Summaries)...")
        # Load or read the Anthropic API key
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", None)
        # Debugging to see if the API key is being read correctly
        if not self.api_key:
            raise ValueError("Anthropic API key not provided or found in environment. "
                             "Please pass it or set ANTHROPIC_API_KEY env var.")

        # Connection to Pinecone
        logging.info(f"Connecting to Pinecone...")
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        
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

        # Question Types the same as the VOC_map_reduce.py script
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
        Purpose: Determine the question type based on the user query using keyword matching.
        Input: User query.
        Output: Question type.
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
            logging.info(f"[determine_question_type] No keywords matched. Using default question type: financial_challenges_1")
            return "financial_challenges_1"
        
        logging.info(f"[determine_question_type] Query mapped to '{best_type[0]}' (score={best_type[1]})")
        return best_type[0]

    def get_offline_summary(self, question_type: str) -> str:
        """
        Purpose: Fetch the offline summary for a given question type from Pinecone.
        Input: Question type.
        Output: Offline summary.
        """
        # Append "_summary" to the question type to get the summary question type
        summary_qtype = f"{question_type}_summary"
        logging.info(f"Fetching offline summary for question_type='{summary_qtype}'")
        
        try:
            # For Pinecone, we need to provide a vector for the query
            # Since we're only filtering by metadata, create a simple dummy vector
            dummy_vector = [0.0] * 384  # Use the dimensionality of your Pinecone index
            
            # Query with metadata filter
            query_results = self.index.query(
                vector=dummy_vector,
                filter={"question_type": summary_qtype},
                top_k=1,
                include_metadata=True
            )
            
            # Check if results are empty
            if not query_results.matches:
                logging.warning(f"No summary found for {summary_qtype}")
                return ""
            
            # Extract the text from the metadata
            summary_text = query_results.matches[0].metadata.get("text", "")
            if not summary_text:
                logging.warning(f"No text found in metadata for {summary_qtype}")
                return ""
                
            logging.info(f"Successfully retrieved summary for {summary_qtype}")
            return summary_text
            
        except Exception as e:
            logging.error(f"Error fetching summary for {summary_qtype}: {e}")
            return ""

    def build_prompt_with_offline_summary(self, user_query: str, summary: str) -> str:
        """
        Purpose: Build a prompt with the user query and offline summary for Anthropic.
        Input: User query, offline summary.
        Output: Prompt for Anthropic.
        """
        prompt = f"""
            You are a helpful assistant with access to a detailed Voice of Customer (VOC) offline summary. Your goal is to have a natural conversation with the user while providing a well-structured, comprehensive answer that incorporates significant parts of the summary.

            **User Query**: {user_query}

            ---
            **OFFLINE SUMMARY**:
            {summary}
            ---

            When crafting your response, please follow these guidelines:

            1. **Structure Your Answer Clearly:**
            - **Introduction:** Begin with a brief overview of the key challenges or insights derived from the summary.
            - **Detailed Analysis:** Break your response into sections or bullet points. Use subheadings like "Credit Challenges," "Application Process Issues," etc., if relevant. Include important statistics, quotes, and details directly from the summary.
            - **Conclusion:** Wrap up with a summary of the main points and ask a clarifying question to continue the conversation.

            2. **Incorporate Relevant Chunks from the Summary:**
            - Reference important data points (e.g., percentages, key quotes, or notable trends) from the summary.
            - Ensure that you integrate at least 60-70% of the content from the summary into your explanation.

            3. **Maintain a Conversational and Friendly Tone:**
            - Engage naturally, as if you are having a friendly discussion with the user.
            - Feel free to ask clarifying questions at the end to further explore the topic.

            4. **Distinguish Between Data and Analysis:**
            - Clearly indicate which parts of your response are directly drawn from the summary and which parts are your own interpretations or additional insights.

            Our company is Breva, a financial management tool for small businesses, and this information is based on real user feedback from forums, surveys, and interviews. Use this context to ensure your response is both informative and actionable.

            Now, please provide a structured, detailed, and friendly response based on the above information.
        """
        return prompt

    def generate_answer(self, user_query: str) -> str:
        """
        Purpose: Generate an answer using Anthropic based on the user query.
        Input: User query.
        Output: Final answer.
        """
        # 1. Determine the question type based on the user query
        qtype = self.determine_question_type(user_query)
        # 2. Fetch the offline summary for the question type
        offline_summary = self.get_offline_summary(qtype)
        if not offline_summary:
            return (f"No offline summary found for question type '{qtype}'. "
                    f"Try a different approach or run offline summarization first.")
        
        # 3. Build the final prompt for Anthropic
        final_prompt = self.build_prompt_with_offline_summary(user_query, offline_summary)

        # Now we need to print the final prompt for debugging
        print(f"{Fore.GREEN}FINAL PROMPT:{Style.RESET_ALL}")
        print(final_prompt)
        
        logging.info("FINAL PROMPT constructed.")

        try:
            # 4. Call Anthropic to generate the final answer
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
            logging.error(f"Error calling Anthropic: {e}")
            return f"[Error generating final answer: {str(e)}]"

# ------------------------------------------------------------------------------
# Streamlit Application
# ------------------------------------------------------------------------------
def main():
    # Define the page title 
    st.set_page_config(page_title="Thrive Grant Application Chatbot", page_icon="🤖", layout="centered")
    
    # Very Basic CSS Styling for the chat messages
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
            color: #000000;  /* dark text for readability */
            font-size: 16px;
            line-height: 1.4;
        }
        .user {
            background-color: #DCF8C6;  /* light green */
            align-self: flex-end;
        }
        .bot {
            background-color: #F1F0F0;  /* light gray */
            align-self: flex-start;
        }
        .chat-box {
            display: flex;
            flex-direction: column;
        }
        /* Style the text input area to have contrasting text */
        textarea {
            color: #000000 !important;  /* force dark text */
            background-color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with instructions
    st.sidebar.title("Instructions")
    st.sidebar.info(
        "Ask any questions regarding the Thrive Grant Application. "
        "The chatbot will respond using our VOC offline summaries."
    )

    # Title and description
    st.title("Thrive Grant Application Chatbot 🤖")
    st.write("Have a conversation with the chatbot regarding the Thrive Grant Application questions.")

    # Initialize the VOCDatabaseQuerier once and cache it
    if "querier" not in st.session_state:
        try:
            with st.spinner("Initializing chatbot..."):
                # Get API keys from Streamlit secrets if available, otherwise use hardcoded values
                pinecone_api_key = st.secrets.get("pinecone_api_key", "pcsk_5vnC9g_A8MYTbGufDu68CXWkiUCqPQY3bSLRULeJvSJEhxVNU8GHHfdMaYSjSAEKFETDAt")
                anthropic_api_key = st.secrets.get("anthropic_api_key", "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")
                
                st.session_state.querier = VOCDatabaseQuerier(
                    pinecone_api_key=pinecone_api_key,
                    index_name="voc-index-2025-q2",
                    anthropic_api_key=anthropic_api_key  
                )
        except Exception as e:
            st.error(f"Error initializing the VOC Database Querier: {e}")
            return

    # Define the memory component to store the chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []  
    
    # The chat container to display the chat messages
    chat_container = st.container()

    # Display the chat messages
    with chat_container:
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        # For each message in the chat, display it with the appropriate styling
        for msg in st.session_state.messages:
            # If the message is from the user, display it on the right
            if msg["role"] == "user":
                st.markdown(f'<div class="message user"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
            # If the message is from the bot, display it on the left
            else:
                st.markdown(f'<div class="message bot"><strong>Chatbot:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Enter your query:", height=100, placeholder="Type your question here...")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input.strip():
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Update UI immediately before processing
        st.rerun()  

    # If a new user message was added but no bot response yet, process it
    # We'll check if the last message is from the user.
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # Get the most recent user message to generate an answer
        latest_user_query = st.session_state.messages[-1]["content"]
        # Spinner to indicate that the bot is generating
        with st.spinner("Generating answer..."):
            answer = st.session_state.querier.generate_answer(latest_user_query)
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()  

if __name__ == "__main__":
    main()
