#!/usr/bin/env python3

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from collections import defaultdict
import uuid
import os
from tqdm import tqdm
import time
from colorama import Fore, Style
import colorama
colorama.init()

# For debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This class will be responsible for creating the VOC database
# The datatabase is a vectorized database of responses to VOC questions
class VOCDatabaseCreator:
    def __init__(
        self,
        persist_directory="/Users/sveerisetti/Desktop/VOC/chroma_database_update_2025_q1_update1",
        do_clustering: bool = False,
        chunk_tokens: int = 0
    ):
        # Debuggint to show that the Database Creator is being initialized
        print(f"{Fore.GREEN}Initializing VOC Database Creator{Style.RESET_ALL}")
        
        # Suppress tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Set the parameters for clustering and also chunking
        self.do_clustering = do_clustering
        self.chunk_tokens = chunk_tokens
        
        # This shows us that the embedding model is being loaded in 
        print(f"{Fore.YELLOW}Loading embedding model...{Style.RESET_ALL}")

        # We are using a standard Sentence Transformer model for the embedding
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Debugging to show that the embedding model has been loaded in
        print(f"{Fore.GREEN}Embedding model loaded successfully{Style.RESET_ALL}")
        
        # Track processed IDs to avoid duplicates
        self.processed_ids = set()
        
        # Each csv has its own question type. This will help us map to a single column.
        # Each csv column has its own question type. The idea is that we will tag responses to these question_types. The idea is that instead of querying all questions at once, in the VOC_LLM_Query.py we will 
        # use semantic matching to find the most relevant question type and then query the database for that question type.
        # question_types is a dictionary that maps a question type to a column in the CSV
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
            }
        }
        
        # Here we initialize the ChromaDB client
        try:
            # Debuggint to show that the ChromaDB is being initialized
            print(f"{Fore.YELLOW}Initializing ChromaDB...{Style.RESET_ALL}")
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # The collection name is the name of the collection that we are storing the responses in
            self.collection_name = "voc_responses"
            existing = [c.name for c in self.client.list_collections()]
            # If the collection name already exists, then we will delete it and then create a new collection with the same name 
            if self.collection_name in existing:
                print(f"{Fore.YELLOW}Collection '{self.collection_name}' exists. Deleting old collection...{Style.RESET_ALL}")
                self.client.delete_collection(self.collection_name)
            
            print(f"{Fore.BLUE}Creating collection '{self.collection_name}'...{Style.RESET_ALL}")

            # We use the create_collection method to create a new collection. We use the SentenceTransformerEmbeddingFunction to embed the responses
            # Create_collection is a method that creates a new collection in the ChromaDB
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='all-MiniLM-L6-v2'
                )
            )
            
            # Debugging to show that the database was created
            print(f"{Fore.GREEN}ChromaDB initialized successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize ChromaDB: {e}{Style.RESET_ALL}")
            raise

    def preprocess_text(self, text: str) -> Optional[str]:
        """
        Purpose: Preprocess a text response before storing it in the database.  \
            One of the issues that I saw was that some of the responses were very short and did not provide any information. Some people just put jibberish in the response, which should be removed. Some people have put "N/A" or "None" in the response, which should also be removed.
        Input: A string of text.
        Output: A string of text that has been preprocessed
        """
        if not isinstance(text, str):
            return None
        
        text = text.strip()
        
        # Stripping out short or uninformative responses
        if any([
            len(text) < 10,
            text.lower() in ['none', 'n/a', '#name?', 'na', 'none.', 'none at this time.'],
            len(text.split()) < 3
        ]):
            return None
    
        return ' '.join(text.split())

    def maybe_chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Purpose: This is optional, but the purpose of this function is to chunk the text into smaller pieces. This is useful when the text is too long and we want to break it down into smaller pieces. This is useful for clustering.
        Input: A string of text and the chunk size
        Output: A list of strings that have been chunked
        """
        if chunk_size <= 0:
            return [text]
        
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end
        return chunks

    def _run_clustering(self, texts: List[str]) -> Dict[int, List[int]]:
        """
        Purpose: I experimented with DBSCAN for clustering. This function takes in a list of texts and returns a dictionary where the key is the cluster label and the value is a list of indices of the text that belong to that cluster.
        Input: A list of texts
        Output: A dictionary where the key is the cluster label and the value is a list of indices of the text that belong to that
        """
        print(f"{Fore.BLUE}Embedding for clustering... (DBSCAN){Style.RESET_ALL}")
        embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
        print(f"{Fore.GREEN}Embeddings generated. Running DBSCAN...{Style.RESET_ALL}")
        
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(embeddings)
        
        cluster_map = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            cluster_map[label].append(idx)
        print(f"{Fore.GREEN}DBSCAN complete. Found {len(cluster_map)} clusters (including noise).{Style.RESET_ALL}")
        return cluster_map

    def _generate_cluster_theme(self, cluster_texts: List[str]) -> str:
        """
        Purpose: This function generates a theme for a cluster of responses. The theme is generated by finding the response that is closest to the centroid of the cluster. This response is then used to generate a theme for the cluster.
        Inpit: A list of texts that belong to the same cluster
        Output: A string that is the theme for the cluster
        """
        if len(cluster_texts) == 1:
            # Just return the snippet
            snippet = cluster_texts[0][:100]
            return snippet + "..." if len(cluster_texts[0]) > 100 else snippet
        
        # Embeddings for cluster
        embeddings = self.embedding_model.encode(cluster_texts, batch_size=32, show_progress_bar=False)
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        rep_idx = np.argmin(distances)
        
        snippet = cluster_texts[rep_idx][:100]
        theme = f"Cluster of {len(cluster_texts)} responses. Rep snippet: {snippet}..."
        return theme

    def create_database(self, csv_path: str):
        """
        Purpose: This function is responsible for creating the database. It reads the CSV file and then processes the responses. It then stores the responses in the database.
        Input: The path to the CSV file
        Output: The populated database
        """
        try:
            # Debugging to show us that the CSV file is being read in 
            print(f"{Fore.YELLOW}Reading CSV file: {csv_path}{Style.RESET_ALL}")

            # Start time to measure how long it takes to create the database
            start_time = time.time()

            # Pandas is used to read in the CSV file
            df = pd.read_csv(csv_path)
            print(f"{Fore.GREEN}Loaded CSV with {len(df)} rows{Style.RESET_ALL}")

            # For each question type, gather relevant responses from that single column
            for q_type, type_info in self.question_types.items():
                print(f"\n{Fore.BLUE}Processing question type: {q_type}{Style.RESET_ALL}")
                type_start_time = time.time()

                # There's exactly 1 column in "columns" for each type (but we keep "columns" as a list in case you want to expand)
                question_columns = type_info["columns"]
                aggregated_rows = []
                total_skipped = 0

                # For each column within the question type, if there are any responses, we will preprocess the text and then store it in the database
                # If we cannot find it, then we will skip it
                for col in question_columns:
                    if col not in df.columns:
                        print(f"{Fore.YELLOW}Skipping column not found in CSV: {col}{Style.RESET_ALL}")
                        continue

                    # Dropping any NaN values in the column
                    col_series = df[col].dropna()

                    # For each response wihtin the column, we will preprocess the text and then store it in the database
                    # This references the preprocess_text function
                    for idx, response_text in col_series.items():
                        cleaned = self.preprocess_text(str(response_text))
                        if not cleaned:
                            total_skipped += 1
                            continue
                        
                        # Chunk the text if needed
                        chunks = self.maybe_chunk_text(cleaned, self.chunk_tokens)
                        for ch in chunks:
                            aggregated_rows.append({
                                "text": ch,
                                "question": col, 
                                "row_id": df.at[idx, 'ID'] if 'ID' in df.columns else str(uuid.uuid4()),
                                "business_name": df.at[idx, 'Business_Name'] if 'Business_Name' in df.columns else "",
                            })

                # If there are no valid responses, then we will skip it
                if not aggregated_rows:
                    print(f"{Fore.YELLOW}No valid responses found for question type '{q_type}'. Skipping.{Style.RESET_ALL}")
                    logger.info(f"Skipped {total_skipped} rows for question type '{q_type}' due to filtering.")
                    continue
                
                # Debugging to show that the responses have been collected
                print(f"{Fore.GREEN}Collected {len(aggregated_rows)} total (possibly chunked) responses for '{q_type}'{Style.RESET_ALL}")
                logger.info(f"Skipped {total_skipped} rows for question type '{q_type}' due to filter constraints.")

                # -- Optional clustering
                if self.do_clustering:
                    texts = [r["text"] for r in aggregated_rows]
                    cluster_map = self._run_clustering(texts)
                    
                    stored_count = 0
                    for cluster_label, indices in cluster_map.items():
                        cluster_responses = [aggregated_rows[i] for i in indices]
                        cluster_texts = [cr["text"] for cr in cluster_responses]
                        
                        theme = self._generate_cluster_theme(cluster_texts)
                        is_cluster = (cluster_label != -1)
                        
                        for resp in cluster_responses:
                            doc_id = f"{q_type}_{resp['row_id']}_{uuid.uuid4()}"
                            if doc_id in self.processed_ids:
                                continue
                            self.processed_ids.add(doc_id)

                            # Precompute embedding
                            doc_embedding = self.embedding_model.encode(resp['text'], show_progress_bar=False)

                            try:
                                self.collection.add(
                                    documents=[resp['text']],
                                    metadatas=[{
                                        "question_type": q_type,
                                        "question": resp["question"], 
                                        "group_label": str(cluster_label),
                                        "group_theme": theme,
                                        "is_cluster": is_cluster,
                                        "business_name": resp["business_name"]
                                    }],
                                    ids=[doc_id],
                                    embeddings=[doc_embedding]
                                )
                                stored_count += 1
                            except Exception as e:
                                print(f"{Fore.RED}Error storing document {doc_id}: {e}{Style.RESET_ALL}")
                    
                    print(f"{Fore.GREEN}Stored {stored_count} documents for '{q_type}' with clustering enabled.{Style.RESET_ALL}")

                else:
                    # If there is no clustering involved, then we will store the responses as is
                    print(f"{Fore.BLUE}Storing raw responses for '{q_type}' without clustering...{Style.RESET_ALL}")

                    # This tell us how many responses have been stored
                    stored_count = 0

                    # For each response within the aggregated_rows, we will store the response in the database
                    for resp in tqdm(aggregated_rows, desc=f"Storing raw documents for '{q_type}'"):
                        # Generate a unique document ID
                        doc_id = f"{q_type}_{resp['row_id']}_{uuid.uuid4()}"
                        # If the document ID has already been processed, then we will skip it
                        if doc_id in self.processed_ids:
                            continue
                        # Here we add the document ID to the processed IDs
                        self.processed_ids.add(doc_id)
                        
                        # Here we precompute the embedding
                        doc_embedding = self.embedding_model.encode(resp['text'], show_progress_bar=False)

                        # Debygging to show that the document is being stored
                        logger.debug(f"Preparing to store doc_id={doc_id[:8]}")
                        print(f"{Fore.LIGHTBLUE_EX}[DEBUG] Uploading document '{doc_id}' for question_type='{q_type}'{Style.RESET_ALL}")
                        print(f"{Fore.LIGHTBLUE_EX}Text:\n{resp['text']}\n{Style.RESET_ALL}")

                        try:
                            # Here we add the information collected into a chroma database
                            self.collection.add(
                                # The text is the response
                                documents=[resp['text']],
                                # The metadata is the question type, the question, and the business name
                                metadatas=[{
                                    "question_type": q_type,  
                                    "question": resp["question"],
                                    "business_name": resp["business_name"]
                                }],
                                ids=[doc_id],
                                embeddings=[doc_embedding]
                            )
                            stored_count += 1
                        except Exception as e:
                            print(f"{Fore.RED}Error storing document {doc_id}: {e}{Style.RESET_ALL}")
                    # Debugging to show that the responses have been stored
                    print(f"{Fore.GREEN}Stored {stored_count} raw documents for '{q_type}' (no clustering).{Style.RESET_ALL}")

                type_end_time = time.time()
                print(f"{Fore.GREEN}Finished processing '{q_type}' in {type_end_time - type_start_time:.2f} seconds{Style.RESET_ALL}")

            end_time = time.time()
            print(f"{Fore.GREEN}\nDatabase creation complete in {end_time - start_time:.2f} seconds{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error creating database: {e}{Style.RESET_ALL}")
            raise

def main():
    """
    Purpose: This is the main function that will be called to create the VOC database
    Ouput: The VOC database will be created
    """
    try:
        # Debugging to show that the main function is being called
        print(f"\n{Fore.BLUE}Starting VOC Database Creation{Style.RESET_ALL}\n")
        
        # Example: clustering disabled, no chunking
        creator = VOCDatabaseCreator(
            persist_directory="/Users/sveerisetti/Desktop/VOC/chroma_database_update_2025_q1_update1",
            do_clustering=False,
            chunk_tokens=0
        )
        creator.create_database('/Users/sveerisetti/Desktop/VOC/merged_grant_applications.csv')
    except Exception as e:
        print(f"{Fore.RED}Error in main: {e}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    main()
