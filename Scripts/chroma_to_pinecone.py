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
        index_name: str = "voc-index"
    ):
        """
        Initialize the migrator to transfer data from ChromaDB to Pinecone.
        """
        print(f"{Fore.GREEN}Initializing ChromaDB to Pinecone Migrator{Style.RESET_ALL}")
        
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.index_name = index_name
        self.batch_size = batch_size
        self.pinecone_api_key = pinecone_api_key
        
        # Connect to ChromaDB
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
            self.pc = Pinecone(api_key=pinecone_api_key)
            print(f"{Fore.GREEN}Successfully initialized Pinecone{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize Pinecone: {e}{Style.RESET_ALL}")
            raise
    
    def retrieve_chroma_data(self):
        """
        Retrieve all data from ChromaDB collection.
        """
        try:
            print(f"{Fore.YELLOW}Retrieving data from ChromaDB collection '{self.collection_name}'...{Style.RESET_ALL}")
            start_time = time.time()
            
            # Get all data from the collection, explicitly include embeddings
            results = self.chroma_collection.get(include=["embeddings", "metadatas", "documents"])
            
            end_time = time.time()
            print(f"{Fore.GREEN}Successfully retrieved {len(results.get('ids', []))} records from ChromaDB in {end_time - start_time:.2f} seconds{Style.RESET_ALL}")
            
            # Debug: Check what's in the results
            print(f"\n{Fore.YELLOW}DEBUG: ChromaDB results structure{Style.RESET_ALL}")
            for key, value in results.items():
                if isinstance(value, (list, np.ndarray)):
                    print(f"  {key}: {type(value)} with {len(value)} items")
                    if key == 'embeddings':
                        print(f"    First embedding shape: {value[0].shape if hasattr(value[0], 'shape') else len(value[0])}")
                    elif value and key != 'embeddings':  # Don't print huge embedding arrays
                        print(f"    Sample: {value[0]}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # Properly check if embeddings exist using numpy check
            if 'embeddings' not in results or (isinstance(results['embeddings'], np.ndarray) and results['embeddings'].size == 0):
                print(f"{Fore.RED}WARNING: No embeddings found in the data.{Style.RESET_ALL}")
                return None
            
            return results
        except Exception as e:
            print(f"{Fore.RED}Error retrieving data from ChromaDB: {e}{Style.RESET_ALL}")
            raise
    
    def create_pinecone_index(self, dimension: int):
        """
        Create a new Pinecone index if it doesn't exist.
        """
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                print(f"{Fore.YELLOW}Pinecone index '{self.index_name}' already exists{Style.RESET_ALL}")
                index = self.pc.Index(self.index_name)
                stats = index.describe_index_stats()
                print(f"{Fore.YELLOW}Index stats: {stats}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Creating new Pinecone index '{self.index_name}' with dimension {dimension}{Style.RESET_ALL}")
                
                # Use gcp-starter region which is available for free tier
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"  # Use GCP free tier region
                        }
                    }
                )
                print(f"{Fore.GREEN}Successfully created Pinecone index{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error with Pinecone index: {e}{Style.RESET_ALL}")
            # If it fails again, try to give helpful information about available regions
            if "region" in str(e).lower():
                try:
                    print(f"{Fore.YELLOW}Attempting to retrieve available regions...{Style.RESET_ALL}")
                    # Try to list available regions if possible
                    print(f"{Fore.YELLOW}Please check the Pinecone console for available regions for your account.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}Free tier typically uses 'gcp-starter' with region 'us-central1'.{Style.RESET_ALL}")
                except Exception:
                    pass
            raise
    
    def format_for_pinecone(self, chroma_data: Dict[str, Any]):
        """
        Format ChromaDB data for Pinecone.
        """
        try:
            print(f"{Fore.YELLOW}Formatting ChromaDB data for Pinecone...{Style.RESET_ALL}")
            
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
            
            # Debug: Print a sample vector
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
        Upload vectors to Pinecone in batches.
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        try:
            print(f"{Fore.YELLOW}Uploading {len(vectors)} vectors to Pinecone index '{self.index_name}' in batches of {batch_size}{Style.RESET_ALL}")
            
            # Connect to the index with V2 API
            index = self.pc.Index(self.index_name)
            
            # Upload in batches
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
        Main migration function that orchestrates the entire process.
        """
        try:
            print(f"\n{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}Starting migration from ChromaDB to Pinecone{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")
            
            # Step 1: Retrieve data from ChromaDB
            chroma_data = self.retrieve_chroma_data()
            
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
            
            # Step 2: Create Pinecone index
            self.create_pinecone_index(dimension)
            
            # Step 3: Format data for Pinecone
            vectors = self.format_for_pinecone(chroma_data)
            
            # Step 4: Upload to Pinecone
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
    """Main function to run the migration."""
    parser = argparse.ArgumentParser(description='Migrate data from ChromaDB to Pinecone')
    parser.add_argument('--chroma-dir', type=str, default="/Users/sveerisetti/Desktop/VOC/chroma_database_update_2025_q1_update1",
                        help='Path to ChromaDB persistence directory')
    parser.add_argument('--pinecone-api-key', type=str, default="pcsk_5vnC9g_A8MYTbGufDu68CXWkiUCqPQY3bSLRULeJvSJEhxVNU8GHHfdMaYSjSAEKFETDAt",
                        help='Pinecone API key')
    parser.add_argument('--collection-name', type=str, default="voc_responses",
                        help='ChromaDB collection name')
    parser.add_argument('--index-name', type=str, default="voc-index",
                        help='Pinecone index name')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for Pinecone uploads')
    
    args = parser.parse_args()
    
    try:
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