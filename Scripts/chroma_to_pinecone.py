#!/usr/bin/env python3

import chromadb
from chromadb.utils import embedding_functions
from pinecone import Pinecone
import logging
import time
import os
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import colorama
from colorama import Fore, Style
import argparse
import json

# Initialize colorama
colorama.init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChromaToPineconeMigrator:
    def __init__(
        self,
        chroma_persist_directory: str,
        pinecone_api_key: str,
        batch_size: int = 100,
        collection_name: str = "voc_responses",
        index_name: str = "voc-index-2025-q2"
    ):
        """
        Purpose: Initialize the migrator with ChromaDB and Pinecone configurations.
        Input: 
            - chroma_persist_directory: Path to the ChromaDB persistence directory.
            - pinecone_api_key: Pinecone API key for authentication.
            - batch_size: Number of vectors to upload to Pinecone in each batch.
            - collection_name: Name of the ChromaDB collection to migrate.
            - index_name: Name of the Pinecone index to create or use.
        Output: Initializes the migrator with connections to ChromaDB and Pinecone.
        """
        print(f"{Fore.GREEN}Initializing ChromaDB to Pinecone Migrator{Style.RESET_ALL}")
        
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.index_name = index_name
        self.batch_size = batch_size
        self.pinecone_api_key = pinecone_api_key
        
        # Here we connect to ChromaDB using the provided directory and collection name.
        try:
            print(f"{Fore.YELLOW}Connecting to ChromaDB at {chroma_persist_directory}{Style.RESET_ALL}")
            self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
            self.chroma_collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='all-MiniLM-L6-v2'
                )
            )
            print(f"{Fore.GREEN}Successfully connected to ChromaDB{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to connect to ChromaDB: {e}{Style.RESET_ALL}")
            raise
            
        # Initialize Pinecone with V2 API
        try:
            print(f"{Fore.YELLOW}Initializing Pinecone{Style.RESET_ALL}")
            # Gather the Pinecone API key from the environment or provided argument
            self.pc = Pinecone(api_key=pinecone_api_key)
            print(f"{Fore.GREEN}Successfully initialized Pinecone{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize Pinecone: {e}{Style.RESET_ALL}")
            raise
    
    def retrieve_chroma_data(self):
        """
        Purpose: Retrieve all data from the specified ChromaDB collection.
        Input: None
        Output: Returns a dictionary containing IDs, embeddings, metadata, and documents.
        """
        try:
            print(f"{Fore.YELLOW}Retrieving data from ChromaDB collection '{self.collection_name}'...{Style.RESET_ALL}")
            start_time = time.time()
            
            # Get all data from the collection, explicitly include embeddings
            results = self.chroma_collection.get(include=["embeddings", "metadatas", "documents"])
            
            end_time = time.time()
            print(f"{Fore.GREEN}Successfully retrieved {len(results.get('ids', []))} records from ChromaDB in {end_time - start_time:.2f} seconds{Style.RESET_ALL}")
            
            # Here is where we check the results
            print(f"\n{Fore.YELLOW}DEBUG: ChromaDB results structure{Style.RESET_ALL}")
            for key, value in results.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"  {key}: {type(value)} with {len(value)} items")
                    if key == 'embeddings':
                        print(f"    First embedding shape: {value[0].shape if hasattr(value[0], 'shape') else len(value[0])}")
                    elif value and key != 'embeddings':  
                        print(f"    Sample: {value[0]}")
                else:
                    print(f"  {key}: {type(value)}")
            
            #Using numpy to ensure embeddings are in the correct format
            if 'embeddings' not in results or (isinstance(results['embeddings'], np.ndarray) and results['embeddings'].size == 0):
                print(f"{Fore.RED}WARNING: No embeddings found in the data.{Style.RESET_ALL}")
                return None
            
            return results
        except Exception as e:
            print(f"{Fore.RED}Error retrieving data from ChromaDB: {e}{Style.RESET_ALL}")
            raise
    
    def create_pinecone_index(self, dimension: int):
        """
        Purpose: Create a Pinecone index with the specified dimension.
        Input:
            - dimension: The dimensionality of the vectors to be stored in the index.
        Output: Creates a Pinecone index if it does not already exist.
        """
        try:
            # Check if the database exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            # If there is an existing index with the same name, we will use it
            if self.index_name in existing_indexes:
                print(f"{Fore.YELLOW}Pinecone index '{self.index_name}' already exists{Style.RESET_ALL}")
                index = self.pc.Index(self.index_name)
                stats = index.describe_index_stats()
                print(f"{Fore.YELLOW}Index stats: {stats}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Creating new Pinecone index '{self.index_name}' with dimension {dimension}{Style.RESET_ALL}")
                
                # Pinecone requires a dimension for the index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"  
                        }
                    }
                )
                print(f"{Fore.GREEN}Successfully created Pinecone index{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error with Pinecone index: {e}{Style.RESET_ALL}")
            
            # If the region for some reason is not available, we can try to provide guidance
            if "region" in str(e).lower():
                try:
                    print(f"{Fore.YELLOW}Attempting to retrieve available regions...{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Please check the Pinecone console for available regions for your account.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Free tier typically uses 'gcp-starter' with region 'us-central1'.{Style.RESET_ALL}")
                except Exception:
                    pass
            raise
    
    def format_for_pinecone(self, chroma_data: Dict[str, Any]):
        """
        Purpose: Format data retrieved from ChromaDB to the structure required by Pinecone.
        Input:
            - chroma_data: Dictionary containing IDs, embeddings, metadata, and documents from ChromaDB.
        Output: Returns a list of vectors formatted for Pinecone.
        """
        try:
            print(f"{Fore.YELLOW}Formatting ChromaDB data for Pinecone...{Style.RESET_ALL}")
            
            # Get the necessary fields from the ChromaDB data
            ids = chroma_data.get('ids', [])
            embeddings = chroma_data.get('embeddings', [])
            metadatas = chroma_data.get('metadatas', [])
            documents = chroma_data.get('documents', [])
            
            # Convert numpy array to list of lists if needed
            if isinstance(embeddings, np.ndarray):
                print(f"{Fore.YELLOW}Converting NumPy array to list...{Style.RESET_ALL}")
                embeddings = embeddings.tolist()
            
            vectors = []
            for i in range(len(ids)):
                # Include document text in metadata
                metadata = metadatas[i].copy() if i < len(metadatas) else {}
                if i < len(documents) and documents[i]:
                    metadata['text'] = documents[i]
                
                # Format for V2 API
                vector = {
                    'id': ids[i],
                    'values': embeddings[i],
                    'metadata': metadata
                }
                vectors.append(vector)
            
            print(f"{Fore.GREEN}Successfully formatted {len(vectors)} vectors for Pinecone{Style.RESET_ALL}")
            
            # Debugging to check the structure of the first vector
            if vectors:
                print(f"\n{Fore.YELLOW}DEBUG: Sample vector structure{Style.RESET_ALL}")
                sample = vectors[0].copy()
                sample['values'] = f"[...] (length: {len(sample['values'])})"
                print(json.dumps(sample, indent=2))
            
            return vectors
        except Exception as e:
            print(f"{Fore.RED}Error formatting data for Pinecone: {e}{Style.RESET_ALL}")
            raise
    
    def upload_to_pinecone(self, vectors: List[Dict], batch_size: int = None):
        """
        Purpose: Upload formatted vectors to the Pinecone index in batches. 
        Input:
            - vectors: List of vectors formatted for Pinecone.
            - batch_size: Number of vectors to upload in each batch (default is self.batch_size).
        Output: Uploads vectors to the Pinecone index.
        """

        # If the batch_size is not provided, use the instance's batch_size
        if batch_size is None:
            batch_size = self.batch_size
            
        try:
            print(f"{Fore.YELLOW}Uploading {len(vectors)} vectors to Pinecone index '{self.index_name}' in batches of {batch_size}{Style.RESET_ALL}")
            
            # Connect to pinecone
            index = self.pc.Index(self.index_name)
            
            # Upload the chroma db vectors as batches
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading batches", total=total_batches):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch)
                time.sleep(0.1)  # Small delay to avoid rate limits
                
            print(f"{Fore.GREEN}Successfully uploaded all vectors to Pinecone{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error uploading to Pinecone: {e}{Style.RESET_ALL}")
            raise
    
    def migrate(self):
        """
        Purpose: Main method to perform the migration from ChromaDB to Pinecone.
        Input: None
        Output: Migrates data from ChromaDB to Pinecone, creating the index and uploading vectors.
        """
        try:
            print(f"\n{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Starting migration from ChromaDB to Pinecone{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")
            
            # Get data from ChromaDB
            chroma_data = self.retrieve_chroma_data()
            
            # If not, data is found, exit early
            if not chroma_data or not chroma_data.get('ids'):
                print(f"{Fore.RED}No data found in ChromaDB collection '{self.collection_name}'{Style.RESET_ALL}")
                return
            
            # Get vector dimension
            embeddings = chroma_data.get('embeddings', [])
            if isinstance(embeddings, np.ndarray) and embeddings.size == 0:
                print(f"{Fore.RED}No embeddings found in ChromaDB collection{Style.RESET_ALL}")
                return
                
            # If embeddings is a numpy array, get the first element correctly
            if isinstance(embeddings, np.ndarray):
                dimension = embeddings[0].shape[0] if embeddings.ndim > 1 else len(embeddings[0])
            else:
                dimension = len(embeddings[0])
                
            print(f"{Fore.BLUE}Vector dimension: {dimension}{Style.RESET_ALL}")
            
            # Create Pinecone index
            self.create_pinecone_index(dimension)
            
            # Format data for Pinecone
            vectors = self.format_for_pinecone(chroma_data)
            
            # Upload vectors to Pinecone
            self.upload_to_pinecone(vectors)
            
            print(f"\n{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Migration completed successfully!{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Total records migrated: {len(chroma_data.get('ids', []))}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"\n{Fore.RED}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.RED}Migration failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.RED}{'=' * 80}{Style.RESET_ALL}")
            raise


def main():
    """
    Purpose: Main function to parse command line arguments and initiate the migration process.
    Input: None
    Output: Parses command line arguments and starts the migration process.
    """
    parser = argparse.ArgumentParser(description='Migrate data from ChromaDB to Pinecone')
    parser.add_argument('--chroma-dir', type=str, default="/Users/sveerisetti/Desktop/Breva_VOC_Chat-main/chroma_database_update_2025_q2",
                        help='Path to ChromaDB persistence directory')
    parser.add_argument('--pinecone-api-key', type=str, default="pcsk_5vnC9g_A8MYTbGufDu68CXWkiUCqPQY3bSLRULeJvSJEhxVNU8GHHfdMaYSjSAEKFETDAt",
                        help='Pinecone API key')
    parser.add_argument('--collection-name', type=str, default="voc_responses",
                        help='ChromaDB collection name')
    parser.add_argument('--index-name', type=str, default="voc-index-2025-q2",
                        help='Pinecone index name')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for Pinecone uploads')
    
    args = parser.parse_args()
    
    try:
        # Use ChromaToPineconeMigrator to perform the migration
        migrator = ChromaToPineconeMigrator(
            chroma_persist_directory=args.chroma_dir,
            pinecone_api_key=args.pinecone_api_key,
            batch_size=args.batch_size,
            collection_name=args.collection_name,
            index_name=args.index_name
        )
        
        migrator.migrate()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        print(f"{Fore.RED}Migration failed. See logs for details.{Style.RESET_ALL}")
        exit(1)


if __name__ == "__main__":
    main()
