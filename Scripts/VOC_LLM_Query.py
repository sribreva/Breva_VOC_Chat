#!/usr/bin/env python3

import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import numpy as np
from typing import Dict, List
import math

# For color logging
from colorama import Fore, Style
import colorama
colorama.init()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class VOCDatabaseQuerier:
    """
    This script queries:
    1) Maps user's question to a question_type
    2) Fetches the offline summary for that question_type (e.g. "desc_community_impact_summary")
    3) Uses Anthropic to produce a final answer based on that summary.
    """

    def __init__(
        self,
        persist_directory: str = "/Users/sveerisetti/Desktop/VOC/chroma_database_update_17",
        collection_name: str = "voc_responses",
        anthropic_api_key: str = None
    ):
        """
        :param persist_directory: Path to the Chroma persistent database
        :param collection_name: Name of the single collection we created
        :param anthropic_api_key: If None, uses os.environ["ANTHROPIC_API_KEY"]
        """
        print(f"{Fore.GREEN}Initializing VOC Database Querier (Offline Summaries){Style.RESET_ALL}")

        # Load or read the Anthropic API key
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", None)
        if not self.api_key:
            raise ValueError("Anthropic API key not provided or found in environment. "
                             "Please pass it or set ANTHROPIC_API_KEY env var.")

        # Initialize the embedding model (same as in VOC_chroma_db_upload.py)
        print(f"{Fore.YELLOW}Loading SentenceTransformer model: all-MiniLM-L6-v2...{Style.RESET_ALL}")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"{Fore.GREEN}Embedding model loaded successfully{Style.RESET_ALL}")

        # Connect to Chroma
        print(f"{Fore.YELLOW}Connecting to Chroma at '{persist_directory}'...{Style.RESET_ALL}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Retrieve the single collection
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        )
        doc_count = self.collection.count()
        print(f"{Fore.GREEN}Loaded collection '{collection_name}'. Document count: {doc_count}{Style.RESET_ALL}")

        # Initialize Anthropic client
        print(f"{Fore.YELLOW}Initializing Anthropic client...{Style.RESET_ALL}")
        self.anthropic = Anthropic(api_key=self.api_key)
        print(f"{Fore.GREEN}Anthropic client initialized successfully{Style.RESET_ALL}")

        # Same question type definitions as in VOC_chroma_db_upload.py
        self.question_types = {
            # Financial Challenges
            "financial_challenges_1": {
                "context": "What specific challenges and difficulties do businesses face with cash flow management, forecasting, and financial planning?"
            },
            "financial_challenges_2": {
                "context": "What financial and accounting tasks take up most time for business owners and how does this impact business growth?"
            },
            "financial_challenges_3": {
                "context": "Describe difficult experiences with managing finances or obtaining loans, including challenges faced and ideal solutions needed"
            },
            "financial_challenges_4": {
                "context": "What are the main obstacles and pain points when applying for business loans and what improvements would help?"
            },

            # Business Description
            "desc_business_brief": {
                "context": "Comprehensive description of the business including its purpose, operations, and core offerings"
            },
            "desc_primary_products": {
                "context": "Detailed breakdown of main products and services offered by the business to customers"
            },
            "desc_community_impact": {
                "context": "How does the business contribute to and positively impact its local community and stakeholders?"
            },
            "desc_equity_inclusion": {
                "context": "What initiatives and efforts does the business take to promote diversity, equity and inclusion internally and externally?"
            },

            # Business Goals and Growth
            "business_goals_1": {
                "context": "What are the business's key accomplishments to date and strategic goals for growth in the next year?"
            },
            "business_goals_2": {
                "context": "What financial management tasks would businesses want automated or handled by an AI-powered virtual CFO?"
            },

            # Financial Tools and Advisory
            "financial_tool_needs": {
                "context": "What specific features and capabilities do businesses need in financial management and credit building tools?"
            },
            "financial_advisor_questions": {
                "context": "What key questions do business owners want to ask financial advisors about managing and growing their business?"
            },

            # Grant and Support
            "grant_usage": {
                "context": "How would businesses use grant funding to improve operations and increase community impact?"
            },
            "additional_context": {
                "context": "Additional relevant business information and context for grant application consideration"
            },

            # Business Challenges
            "business_obstacles": {
                "context": "Major business challenges encountered and strategies used to overcome them successfully"
            }
        }

        # Prepare embeddings for these question-type contexts
        self._prepare_type_context_embeddings()

    def _prepare_type_context_embeddings(self):
        """Pre-embed each question_type's 'context' field for quick similarity checks."""
        self.type_embeddings = {}
        for qtype, info in self.question_types.items():
            ctx = info["context"]
            emb = self.embedding_model.encode(ctx, show_progress_bar=False)
            self.type_embeddings[qtype] = emb

    def determine_question_type(self, user_query: str) -> str:
        """Map user query to most relevant question type."""
        query_emb = self.embedding_model.encode(user_query, show_progress_bar=False)

        best_type = None
        best_score = float('-inf')

        for qtype, emb in self.type_embeddings.items():
            # Dot product for similarity
            score = float(np.dot(query_emb, emb))
            if score > best_score:
                best_score = score
                best_type = qtype

        logging.info(f"[determine_question_type] Query mapped to '{best_type}' (score={best_score:.4f})")
        return best_type

    def get_offline_summary(self, question_type: str) -> str:
        """Retrieve the offline summary doc from Chroma."""
        summary_qtype = f"{question_type}_summary"
        logging.info(f"Attempting to fetch offline summary for question_type='{summary_qtype}'")
        
        try:
            results = self.collection.get(
                where={"question_type": summary_qtype},
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get("documents"):
                logging.warning(f"No summary found for {summary_qtype}")
                return ""
                
            docs = results["documents"]
            if not docs:
                logging.warning(f"Empty documents returned for {summary_qtype}")
                return ""
                
            logging.info(f"Successfully retrieved summary for {summary_qtype}")
            return docs[0]
            
        except Exception as e:
            logging.error(f"Error fetching summary for {summary_qtype}: {e}")
            return ""

    def build_prompt_with_offline_summary(self, user_query: str, summary: str) -> str:
        prompt = f"""

You are provided with the following **Offline Summary** containing Voice of Customer (VOC) information.  
Your task is to extract and present **all relevant details** exactly as they appear in the summary, while ensuring the response follows the exact structure below.

**User Query**: {user_query}

---
**OFFLINE SUMMARY**:
{summary}
---

**Response Format** (Your output **must** follow this structure exactly):

1. **Executive Summary**  
   - Copy all key points, numbers, and findings word-for-word.  
   - Do **not** remove or condense any information.

2. **Major Themes & Patterns**  
   - Extract every listed theme and sub-theme.  
   - Preserve all percentages, bullet points, and explanations.  

3. **Comprehensive Statistical Analysis**  
   - Include **every** number, breakdown, and correlation from the summary.  
   - If any data point is listed multiple times, ensure all instances are included.

4. **Deep Insights & Implications (Including Quotes)**  
   - Extract **all** direct quotes and their contextual explanations.  
   - If quotes are synthesized, still include them exactly as written.  
   - Maintain every surprising or unexpected finding.

5. **Areas for Further Investigation**  
   - List **all** research questions and blind spots.  
   - Do **not** summarize—copy the exact wording from the summary.

6. **Conclusion**  
   - Extract the entire conclusion section verbatim.  
   - Ensure every key takeaway is included in full.

**Rules**:
- **Do not** summarize, interpret, or analyze—your response must be a **direct extraction** of all relevant details.
- ** When providing statistics, you must include counts and percentages. The counts must be a fraction where the numerator is the count and the denominator is the total number of responses.**
- **Do not omit any section or subsection.**
- **If a section contains no data, explicitly state "No data provided."**
- **Use the exact words, structure, and formatting found in the summary.**

Your final output should be a **structured, section-by-section data extraction** that maintains 100% fidelity to the original summary.

        """

        return prompt

    def generate_answer(self, user_query: str) -> str:
        """Generate final answer using offline summary and Claude."""
        qtype = self.determine_question_type(user_query)
        offline_summary = self.get_offline_summary(qtype)
        if not offline_summary:
            return (f"No offline summary found for question type '{qtype}'. "
                    f"Try a different approach or run offline summarization first.")

        final_prompt = self.build_prompt_with_offline_summary(user_query, offline_summary)

        print(f"FINAL PROMPT ******:  {final_prompt}")

        try:
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8192,
                temperature=0.1,
                messages=[{
                    "role": "user",
                    "content": final_prompt
                }]
            )
            # Handle TextBlock response
            if hasattr(response.content[0], 'text'):
                return response.content[0].text
            else:
                return str(response.content[0])
        except Exception as e:
            logging.error(f"Error calling Anthropic with offline summary: {e}")
            return "[Error generating final answer with offline summary]"

def main():
    try:
        print(f"{Fore.CYAN}Starting VOC LLM Query Script (Using Offline Summaries)...{Style.RESET_ALL}")

        querier = VOCDatabaseQuerier(
            persist_directory="/Users/sveerisetti/Desktop/VOC/chroma_database_update_17",
            collection_name="voc_responses",
            anthropic_api_key="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"  
        )

        # Query
        test_queries = [
            "If there were no constraints, what tasks would you want an advanced technology like a virtual Chief Financial Officer to handle for you daily?",
        ]

        for query in test_queries:
            print(f"\n{Fore.BLUE}User Query:{Style.RESET_ALL} {query}\n")
            answer = querier.generate_answer(query)
            print(f"{Fore.GREEN}Claude's Final Answer (Offline Summary):\n{Style.RESET_ALL}{answer}\n")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()