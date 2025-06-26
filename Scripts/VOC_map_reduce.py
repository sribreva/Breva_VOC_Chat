#!/usr/bin/env python3

import chromadb
from chromadb.utils import embedding_functions
import anthropic
from typing import List, Dict
import logging
from tqdm import tqdm
import time
import os
from collections import defaultdict
import uuid
import sys
from datetime import datetime
from colorama import Fore, Style
import colorama
colorama.init()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This class is responsible for logging the output to a file
class Logger:
    def __init__(self, filename="/Users/sveerisetti/Desktop/Breva_VOC_Chat-main/Outputs/claude_q1_2025_update_screen.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# This class is responsible for processing the Voice of Customer (VOC) data using a Map-Reduce pipeline
# We will sequentially process each question type, split the responses into batches, summarize each batch, and then create a meta-summary
class VOCMapReduceProcessor:
    def __init__(
        self,
        persist_directory="/Users/sveerisetti/Desktop/Breva_VOC_Chat-main/chroma_database_update_2025_q2",
        batch_size: int = 60,
        # Insert Anthropic Key
        anthropic_api_key: str = "" 
    ):
        """
        Initialize the Map-Reduce processor for VOC data.

        Args:
            persist_directory: Path to ChromaDB storage
            batch_size: Number of responses to process per batch
            anthropic_api_key: Anthropic API key for Claude
        """
        print(f"{Fore.GREEN}Initializing VOC Map-Reduce Processor{Style.RESET_ALL}")
        
        # This is the batch size (number of responses to process per batch)
        self.batch_size = batch_size

        # This is the order of which the questions will be processed
        self.question_order = [
            # Business Description Questions
            "desc_business_brief",
            "desc_primary_products",
            "desc_community_impact",
            "desc_equity_inclusion",
            
            # Business Obstacles and Goals
            "business_obstacles",
            "business_goals_1",
            "business_goals_2",
            
            # Financial Challenges Questions
            "financial_challenges_1",
            "financial_challenges_2",
            "financial_challenges_3",
            "financial_challenges_4",
            
            # Financial Tools and Needs
            "financial_tool_needs",
            "financial_advisor_questions",
            
            # Grant Related
            "grant_usage",
            "additional_context",

            # Newly added question types
            "reason_financial_assistance",
            "financial_planning_responsible"
        ]
        
        # These are the question types. These are the same as the keys in the question_types dictionary in the VOC_chroma_db_upload.py file
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
        
        # Here we are connecting to the ChromaDB database
        try:
            print(f"{Fore.YELLOW}Connecting to ChromaDB...{Style.RESET_ALL}")
            # Connection to the ChromaDB database
            self.client = chromadb.PersistentClient(path=persist_directory)
            # Using the 'voc_responses' collection with the SentenceTransformerEmbeddingFunction
            # Embedding function is used to convert text data into numerical vectors
            self.collection = self.client.get_collection(
                name="voc_responses",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='all-MiniLM-L6-v2'
                )
            )
            print(f"{Fore.GREEN}Successfully connected to ChromaDB{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to connect to ChromaDB: {e}{Style.RESET_ALL}")
            raise

        # Initialize the Anthropic client for interacting with Claude
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

    def get_question_types(self) -> List[str]:
        """
        Purpose: Get all unique question types from the ChromaDB collection.
        Input: None
        Output: List of unique question
        """
        # Get all documents and metadata from the collection
        results = self.collection.get()
        question_types = set()
        # For each metadata in the results, if the 'question_type' key exists, add it to the question_types set
        for metadata in results['metadatas']:
            if 'question_type' in metadata:
                # Only include question types that don't end with '_summary'
                if not metadata['question_type'].endswith('_summary'):
                    question_types.add(metadata['question_type'])
        return sorted(list(question_types))

    def get_responses_for_question(self, question_type: str) -> List[Dict]:
        """
        Purpose: Get all responses for a specific question type.
        Input: question_type (str)
        Output: List of dictionaries containing response text and metadata
        """
        # The output dictionary will contain the documents and metadata for the specified question type
        # We query the ChromaDB collection for documents where the 'question_type' key matches the input question_type
        results = self.collection.get(
            where={"question_type": question_type}
        )
        
        # For each document and metadata in the results, we create a dictionary with the 'text' and 'metadata' keys
        responses = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            responses.append({
                'text': doc,
                'metadata': meta
            })
        return responses

    def create_batch_summary_prompt(self, responses: List[Dict], question_type: str) -> str:
        """
        Purpose: Create a prompt for generating a batch summary from a list of responses.
        Input: responses (List[Dict]), question_type (str)
        Output: Prompt for generating a batch summary as a string
        """
        context = self.question_types[question_type]['context']
        prompt = f"""You are analyzing Voice of Customer (VOC) responses for the question: {context}

        Please provide a detailed analysis of the following {len(responses)} responses with these components:

        1. Key Themes & Patterns (40% of your response):
        - Major recurring themes and patterns
        - Subtopics within each theme
        - How these themes interconnect
        - Any contradictions or tensions between different viewpoints

        2. Statistical Breakdown (30% of your response):
        - **Counts** of responses mentioning each key theme (e.g., "15 responses mentioned X").
        - Breakdown of subthemes and their frequencies (e.g., "5 responses mentioned Y under theme X").
        - Notable correlations between different themes
        - Any unexpected patterns or outliers in the data

        3. Notable Insights & Deep Dive (30% of your response):
        - Most insightful or unique responses (quote specific examples)
        - Unexpected or counterintuitive findings
        - Specific success stories or challenges mentioned
        - Business impact and implications
        - Any gaps or unmet needs revealed in the responses

        Responses to analyze:
        """
        # Enumerate gives us the index and the response in the responses list
        # We add the response text to the prompt
        for i, resp in enumerate(responses, 1):
            prompt += f"\n{i}. {resp['text']}\n"

        prompt += "\nPlease structure your response clearly using the sections above. Support your analysis with specific examples and quotes from the responses. Focus on extracting actionable insights and patterns that would be valuable for business decision-making."
        
        print(f"{Fore.GREEN}Prompt for Batch Summary:\n{Style.RESET_ALL}{prompt}\n")
        return prompt

    def create_meta_summary_prompt(self, batch_summaries: List[str], question_type: str) -> str:
        """
        Purpose: Create a prompt for generating a meta-summary from a list of batch summaries.
        Input: batch_summaries (List[str]), question_type (str)
        Output: Prompt for generating a meta-summary as a string
        """
        # Get the context for the question type
        context = self.question_types[question_type]['context']

        # Meta-summary prompt template
        prompt = f"""You are creating a comprehensive meta-analysis of all Voice of Customer (VOC) responses for the question: **{context}**

        Your task is to synthesize insights from {len(batch_summaries)} batch summaries into a cohesive, well-structured analysis. Your response should be tailored for business decision-makers, providing actionable insights and a clear understanding of the key challenges, patterns, and opportunities identified in the data. 

        Structure your response as follows:

        ---

        ### 1. Executive Summary (10% of response):
        - Provide a high-level overview of the most critical findings from the analysis.
        - Highlight the most significant patterns and trends observed across all responses.
        - Summarize the key takeaways for business decision-makers, emphasizing actionable insights.

        ---

        ### 2. Major Themes & Patterns (30% of response):
        - Identify and analyze the **recurring themes** across all batches. Group these themes into clear categories (e.g., revenue unpredictability, seasonality, expense management, funding gaps, external factors).
        - Discuss how these themes **evolved or varied** across different batches. For example, are certain themes more prominent in specific industries or business types?
        - Explore the **interconnections** between themes. For instance, how do revenue fluctuations impact expense management or funding needs?
        - Highlight any **contradictions or nuances** in the responses. For example, why do some businesses report no challenges while others face significant cash flow issues?

        ---

        ### 3. Comprehensive Statistical Analysis (25% of response):
        - Aggregate **counts** of responses mentioning each key theme across all batches (e.g., "150 responses mentioned X out of 960 total responses").
        - Calculate **weighted percentages** for key themes and subthemes based on the total number of responses (e.g., "15.6% of responses mentioned X").
        - Analyze the **distribution of responses** across different categories (e.g., industries, business sizes, geographic regions).
        - Identify **statistical trends and patterns**, such as correlations between themes (e.g., seasonality and inventory challenges). Please be specific with numbers and percentages.
        - Discuss the **confidence levels** in the findings, including potential blind spots or limitations in the data.

        ---

        ### 4. Deep Insights & Implications (25% of response):
        - Share **compelling individual stories or examples** that illustrate key challenges or successes. Use direct quotes from the batch summaries where possible.
        - Highlight **unexpected findings** and explain their significance. For example, were there any counterintuitive insights or unique challenges faced by specific groups (e.g., minority-owned businesses)?
        - Discuss the **business implications** of the findings. What do these insights mean for decision-makers? Provide **actionable recommendations** to address the identified challenges.
        - Identify **gaps and opportunities** revealed by the analysis. For example, are there unmet needs (e.g., better financial tools, access to funding) that businesses or policymakers could address?

        ---

        ### 5. Areas for Further Investigation (10% of response):
        - Pose **specific questions** raised by the analysis. For example, what additional data or research would help clarify contradictions or gaps in the findings?
        - Identify **potential blind spots or limitations** in the current data. Are there underrepresented industries, business types, or regions that should be explored further?
        - Suggest **follow-up research** or actions to address the identified gaps. For example, should future VOC surveys include more detailed questions about specific challenges (e.g., access to capital for minority-owned businesses)?

        ---

        ### Additional Guidelines:
        - Use **clear, concise language** and avoid jargon. The analysis should be accessible to a non-technical audience.
        - Support all major findings with **specific examples, statistics, and quotes** from the batch summaries.
        - Prioritize **actionable insights** that can inform decision-making. For example, what strategies or tools could help businesses mitigate cash flow challenges?
        - Ensure the analysis is **balanced**, highlighting both challenges and opportunities.

        ---

        ### Batch Summaries to Synthesize:
        """
        for i, summary in enumerate(batch_summaries, 1):
            prompt += f"\nBatch {i}:\n{summary}\n"       

        # Additional guidelines for the meta-summary
        prompt += "\nProvide a thorough, well-structured analysis that gives decision-makers a clear understanding of the voice of customer data. Support all major findings with specific examples and statistics from the batch summaries."
        print(f"{Fore.CYAN}Prompt for Meta-Summary:\n{Style.RESET_ALL}{prompt}\n")
        return prompt

    def get_llm_summary(self, prompt: str, initial_delay=5) -> str:
        """
        Purpose: Generate a summary using the Anthropic Claude model with infinite retries
        Input: prompt (str), initial_delay (int)
        Output: Summary generated by the model as a string
        """
        delay = initial_delay
        attempt = 1
        
        while True:
            try:
                message = self.anthropic_client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                if hasattr(message.content[0], 'text'):
                    return message.content[0].text
                return str(message.content[0])
                
            except anthropic.InternalServerError as e:
                print(f"{Fore.YELLOW}Server overloaded on attempt {attempt}, retrying in {delay} seconds...{Style.RESET_ALL}")
                time.sleep(delay)
                # Exponential backoff with max delay of 5 minutes
                delay = min(delay * 2, 300)
                attempt += 1
                continue
            except Exception as e:
                print(f"{Fore.RED}Unexpected error on attempt {attempt}: {e}{Style.RESET_ALL}")
                time.sleep(delay)
                delay = min(delay * 2, 300)
                attempt += 1
                continue

    def store_summary(self, question_type: str, summary: str):
        """
        Purpose: Store the generated summary in the ChromaDB collection.
        Input: question_type (str), summary (str)
        Output: None
        """
        try:
            # Create a unique document ID using the question type and a random UUID
            doc_id = f"{question_type}_summary_{uuid.uuid4()}"
            # Add the summary to the collection with metadata
            self.collection.add(
                documents=[summary],
                metadatas=[{
                    "question_type": f"{question_type}_summary",
                    "summary_type": "map_reduce",
                    "timestamp": time.time()
                }],
                ids=[doc_id]
            )
            print(f"{Fore.GREEN}Successfully stored summary for {question_type}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error storing summary: {e}{Style.RESET_ALL}")
            raise

    def count_tokens(self, text: str, max_retries=3, delay=5) -> int:
        """
        Purpose: Helper function to count the number of tokens in a given text.
        Input: text (str), max_retries (int), delay (int)
        Output: Number of tokens in the text
        """
        for attempt in range(max_retries):
            try:
                message = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1,
                    messages=[{"role": "user", "content": text}]
                )
                return message.usage.input_tokens
            except anthropic.InternalServerError as e:
                if attempt == max_retries - 1:
                    return 0
                print(f"{Fore.YELLOW}Server overloaded, retrying token count in {delay} seconds...{Style.RESET_ALL}")
                time.sleep(delay)
                delay *= 2

    def process_question_type(self, question_type: str):
        """
        Purpose: Process a specific question type using the Map-Reduce pipeline.
        Input: question_type (str)
        Output: Store the final summary in the ChromaDB collection
        """
        print(f"\n{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Processing question type: {question_type}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Context: {self.question_types[question_type]['context']}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}")
        
        # Here we get all the responses for the specified question type
        responses = self.get_responses_for_question(question_type)
        if not responses:
            print(f"{Fore.YELLOW}No responses found for {question_type}. Skipping.{Style.RESET_ALL}")
            return

        # We split the responses into batches based on the batch size
        batches = [responses[i:i + self.batch_size] for i in range(0, len(responses), self.batch_size)]
        print(f"{Fore.GREEN}Split {len(responses)} responses into {len(batches)} batches{Style.RESET_ALL}")

        # For each batch, we will create a batch summary prompt, generate a summary, and store the summary
        batch_summaries = []
        for i, batch in enumerate(batches):
            print(f"\n{Fore.CYAN}Processing Batch {i+1}/{len(batches)}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-'*80}{Style.RESET_ALL}")
            
            # Log input batch
            print(f"{Fore.YELLOW}Input Batch Contents:{Style.RESET_ALL}")
            for j, resp in enumerate(batch, 1):
                print(f"{Fore.YELLOW}{j}. {resp['text']}...{Style.RESET_ALL}")
            
            # Here we use the create_batch_summary_prompt function to create a prompt for the batch summary
            prompt = self.create_batch_summary_prompt(batch, question_type)
            print(f"\n{Fore.GREEN}Prompt for Batch {i+1}:{Style.RESET_ALL}")

            # Here we use the helper function count_tokens to count the number of tokens in the prompt
            token_count = self.count_tokens(prompt)
            print(f"{Fore.BLUE}Token count for prompt: {token_count}{Style.RESET_ALL}")

            # Here we generate the summary using the prompt
            summary = self.get_llm_summary(prompt)
            
            # Log batch summary
            print(f"\n{Fore.GREEN}Batch {i+1} Summary:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'-'*40}{Style.RESET_ALL}")
            print(summary)
            print(f"{Fore.GREEN}{'-'*40}{Style.RESET_ALL}")
            
            batch_summaries.append(summary)

        # Reduce: Create meta-summary
        print(f"\n{Fore.BLUE}Creating meta-summary from {len(batch_summaries)} batch summaries{Style.RESET_ALL}")
        meta_prompt = self.create_meta_summary_prompt(batch_summaries, question_type)
        print(f"\n{Fore.CYAN}Prompt for Meta-Summary:{Style.RESET_ALL}")

        meta_token_count = self.count_tokens(meta_prompt)
        print(f"{Fore.BLUE}Token count for meta-summary prompt: {meta_token_count}{Style.RESET_ALL}")

        final_summary = self.get_llm_summary(meta_prompt)

        # Log final summary
        print(f"\n{Fore.MAGENTA}FINAL META-SUMMARY FOR {question_type}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")
        print(final_summary)
        print(f"{Fore.MAGENTA}{'='*80}{Style.RESET_ALL}")

        # Store the final summary
        self.store_summary(question_type, final_summary)

    def process_all_questions(self):
        """Process only those question types which do not yet have a meta‐summary."""
        print(f"{Fore.GREEN}Determining which question types still need summaries{Style.RESET_ALL}")

        # 1) Find all existing meta‐summaries
        all_docs = self.collection.get()
        existing_summaries = {
            m["question_type"].rsplit("_summary", 1)[0]
            for m in all_docs["metadatas"]
            if m.get("question_type", "").endswith("_summary")
        }

        # 2) Compute defined vs available vs to_do
        defined   = set(self.question_types.keys())
        available = set(self.get_question_types())
        to_do     = (defined & available) - existing_summaries

        print(f"{Fore.BLUE}Already have summaries for: {sorted(existing_summaries)}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Will generate summaries for: {sorted(to_do)}{Style.RESET_ALL}")

        # 3) Process in your defined order, only missing ones
        for q in self.question_order:
            if q in to_do:
                self.process_question_type(q)

        # 4) Any leftovers not in the explicit order
        leftovers = to_do - set(self.question_order)
        for q in sorted(leftovers):
            self.process_question_type(q)

    def cleanup_old_summaries(self):
        """
        Purpose: Remove all previous summary documents from the collection
        Input: None
        Output: None
        """
        try:
            print(f"{Fore.YELLOW}Cleaning up old summaries...{Style.RESET_ALL}")
            # Get all documents
            results = self.collection.get()
            
            # Find documents to delete (those with question_type ending in _summary)
            ids_to_delete = [
                doc_id for doc_id, metadata in zip(results['ids'], results['metadatas'])
                if metadata['question_type'].endswith('_summary')
            ]
            
            if ids_to_delete:
                print(f"{Fore.BLUE}Found {len(ids_to_delete)} summaries to delete{Style.RESET_ALL}")
                self.collection.delete(ids=ids_to_delete)
                print(f"{Fore.GREEN}Successfully deleted old summaries{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}No summaries found to delete{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error cleaning up summaries: {e}{Style.RESET_ALL}")
            raise

    def run_as_job(self):
        """
        Purpose: Run the processing as a continuous job that won't stop until complete
        Input: None
        Output: None
        """
        while True:
            try:
                print(f"\n{Fore.BLUE}Starting VOC Processing Job{Style.RESET_ALL}")
                self.process_all_questions()
                print(f"\n{Fore.GREEN}Job completed successfully!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error in job: {e}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Retrying job in 60 seconds...{Style.RESET_ALL}")
                time.sleep(60)
                continue


def main():
    """Main function to run the map-reduce processing as a continuous job."""
    try:
        sys.stdout = Logger()
        
        print(f"\n{Fore.BLUE}Starting VOC Map-Reduce Processing Job{Style.RESET_ALL}\n")
        
        processor = VOCMapReduceProcessor(
            persist_directory="/Users/sveerisetti/Desktop/Breva_VOC_Chat-main/chroma_database_update_2025_q2",
            batch_size=100,
            # Anthropic Key
            anthropic_api_key=""
        )
        
        processor.run_as_job()
        
        print(f"\n{Fore.GREEN}Job completed successfully!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Fatal error in main: {e}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
