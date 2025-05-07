import os
import pandas as pd
import numpy as np
import logging
import json
from sentence_transformers import SentenceTransformer
import torch
import faiss
import time
import datetime
import re
from tqdm import tqdm

class MINDDataConnector:
    """
    Handles loading, processing, and encoding news articles from the MIND dataset.
    Prepares data for the agentic news editor system.
    """
    
    def __init__(self, data_dir='agentic_news_editor/processed_data'):
        """Initialize the MIND data connector"""
        self.data_dir = data_dir
        self.news_file = os.path.join(data_dir, 'news.tsv')
        self.behaviors_file = os.path.join(data_dir, 'behaviors.tsv')
        self.processed_file = os.path.join(data_dir, 'processed_news.csv')
        self.embeddings_file = os.path.join(data_dir, 'articles_with_embeddings.csv')
        self.faiss_index_file = 'articles_faiss.index'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def load_news_data(self):
        """Load news data from TSV file"""
        try:
            if os.path.exists(self.news_file):
                # Define column names based on MIND dataset structure
                columns = ['news_id', 'category', 'subcategory', 'title', 'abstract', 
                           'url', 'title_entities', 'abstract_entities']
                
                news_df = pd.read_csv(self.news_file, sep='\t', names=columns)
                logging.info(f"Loaded {len(news_df)} news articles from {self.news_file}")
                return news_df
            else:
                logging.error(f"News file not found: {self.news_file}")
                return None
        except Exception as e:
            logging.error(f"Error loading news data: {e}")
            return None
    
    def load_behaviors_data(self):
        """Load user behaviors data from TSV file"""
        try:
            if os.path.exists(self.behaviors_file):
                # Define column names based on MIND dataset structure
                columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
                
                behaviors_df = pd.read_csv(self.behaviors_file, sep='\t', names=columns)
                logging.info(f"Loaded {len(behaviors_df)} behavior records from {self.behaviors_file}")
                return behaviors_df
            else:
                logging.error(f"Behaviors file not found: {self.behaviors_file}")
                return None
        except Exception as e:
            logging.error(f"Error loading behaviors data: {e}")
            return None
    
    def process_news_data(self, news_df):
        """Process raw news data into a cleaner format"""
        try:
            if news_df is None or len(news_df) == 0:
                logging.error("No news data to process")
                return None
            
            # Create a copy to avoid modifying the original
            processed_df = news_df.copy()
            
            # Filter out rows with missing titles or abstracts
            processed_df = processed_df.dropna(subset=['title', 'abstract'])
            
            # Clean text data
            processed_df['title'] = processed_df['title'].apply(self._clean_text)
            processed_df['abstract'] = processed_df['abstract'].apply(self._clean_text)
            
            # Add a publication date field (simulated for MIND dataset)
            # Use the current date for demonstration
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            processed_df['pub_date'] = current_date
            
            # Extract entities if available
            processed_df['entities'] = processed_df.apply(
                lambda row: self._extract_entities(row['title_entities'], row['abstract_entities']), 
                axis=1
            )
            
            # Calculate content length
            processed_df['title_length'] = processed_df['title'].apply(len)
            processed_df['abstract_length'] = processed_df['abstract'].apply(len)
            
            # Drop columns that aren't needed
            processed_df = processed_df.drop(['title_entities', 'abstract_entities'], axis=1)
            
            logging.info(f"Processed {len(processed_df)} news articles")
            
            # Save processed data
            processed_df.to_csv(self.processed_file, index=False)
            logging.info(f"Saved processed data to {self.processed_file}")
            
            return processed_df
            
        except Exception as e:
            logging.error(f"Error processing news data: {e}")
            return None
    
    def _clean_text(self, text):
        """Clean text by removing extra whitespace and special characters"""
        if not isinstance(text, str):
            return ""
            
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Replace HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        return text
    
    def _extract_entities(self, title_entities, abstract_entities):
        """Extract and combine entities from title and abstract"""
        entities = []
        
        try:
            # Extract from title if available
            if isinstance(title_entities, str) and title_entities:
                title_ents = json.loads(title_entities)
                for ent in title_ents:
                    if isinstance(ent, dict) and 'WikidataId' in ent:
                        entities.append({
                            'id': ent.get('WikidataId', ''),
                            'surface_form': ent.get('SurfaceForms', [''])[0],
                            'type': ent.get('Type', ''),
                            'source': 'title'
                        })
            
            # Extract from abstract if available
            if isinstance(abstract_entities, str) and abstract_entities:
                abstract_ents = json.loads(abstract_entities)
                for ent in abstract_ents:
                    if isinstance(ent, dict) and 'WikidataId' in ent:
                        entities.append({
                            'id': ent.get('WikidataId', ''),
                            'surface_form': ent.get('SurfaceForms', [''])[0],
                            'type': ent.get('Type', ''),
                            'source': 'abstract'
                        })
            
            return json.dumps(entities) if entities else '[]'
            
        except Exception as e:
            logging.warning(f"Error extracting entities: {e}")
            return '[]'
    
    def compute_article_embeddings(self, processed_df=None, model_name='paraphrase-MiniLM-L6-v2'):
        """Compute embeddings for articles using SentenceTransformer"""
        try:
            # Load processed data if not provided
            if processed_df is None:
                if os.path.exists(self.processed_file):
                    processed_df = pd.read_csv(self.processed_file)
                    logging.info(f"Loaded {len(processed_df)} processed articles")
                else:
                    logging.error(f"Processed file not found: {self.processed_file}")
                    return None
            
            # Initialize SentenceTransformer model
            logging.info(f"Loading SentenceTransformer model: {model_name}")
            model = SentenceTransformer(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            logging.info(f"Using device: {device}")
            
            # Prepare text for embedding (combine title and abstract)
            texts = []
            for _, row in processed_df.iterrows():
                title = row['title'] if isinstance(row['title'], str) else ""
                abstract = row['abstract'] if isinstance(row['abstract'], str) else ""
                combined_text = f"{title} {abstract}"
                texts.append(combined_text)
            
            # Compute embeddings in batches
            logging.info("Computing embeddings...")
            batch_size = 32
            embeddings = []
            
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
            
            # Convert embeddings to string format for storage
            processed_df['embedding'] = [','.join(map(str, emb)) for emb in embeddings]
            
            # Save dataframe with embeddings
            processed_df.to_csv(self.embeddings_file, index=False)
            logging.info(f"Saved {len(processed_df)} articles with embeddings to {self.embeddings_file}")
            
            return processed_df
            
        except Exception as e:
            logging.error(f"Error computing article embeddings: {e}")
            return None
    
    def build_faiss_index(self, articles_df=None):
        """Build a FAISS index for fast similarity search"""
        try:
            # Load data with embeddings if not provided
            if articles_df is None:
                if os.path.exists(self.embeddings_file):
                    articles_df = pd.read_csv(self.embeddings_file)
                    logging.info(f"Loaded {len(articles_df)} articles with embeddings")
                else:
                    logging.error(f"Embeddings file not found: {self.embeddings_file}")
                    return None
            
            # Extract embeddings from dataframe
            embeddings_list = []
            for i, row in articles_df.iterrows():
                if 'embedding' in row and isinstance(row['embedding'], str):
                    emb = np.array([float(x) for x in row['embedding'].split(',')])
                    embeddings_list.append(emb)
            
            if not embeddings_list:
                logging.error("No valid embeddings found in dataframe")
                return None
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            # Save index to disk
            faiss.write_index(index, self.faiss_index_file)
            logging.info(f"Built FAISS index with {index.ntotal} vectors of dimension {dimension}")
            logging.info(f"Saved FAISS index to {self.faiss_index_file}")
            
            return index
            
        except Exception as e:
            logging.error(f"Error building FAISS index: {e}")
            return None
    
    def calculate_ctr_from_behaviors(self, news_df, behaviors_df):
        """Calculate historical CTR for news articles based on user behaviors"""
        try:
            # Initialize click counters
            news_clicks = {}
            news_impressions = {}
            
            # Process each behavior record
            for _, row in tqdm(behaviors_df.iterrows(), total=len(behaviors_df), desc="Processing behaviors"):
                if not isinstance(row['impressions'], str):
                    continue
                    
                # Parse impressions
                impressions = row['impressions'].split()
                
                for impression in impressions:
                    parts = impression.split('-')
                    if len(parts) != 2:
                        continue
                        
                    news_id, click = parts
                    
                    # Count impression
                    if news_id not in news_impressions:
                        news_impressions[news_id] = 0
                    news_impressions[news_id] += 1
                    
                    # Count click if applicable
                    if click == '1':
                        if news_id not in news_clicks:
                            news_clicks[news_id] = 0
                        news_clicks[news_id] += 1
            
            # Calculate CTR for each article
            ctr_data = []
            for news_id, impressions in news_impressions.items():
                clicks = news_clicks.get(news_id, 0)
                ctr = clicks / impressions if impressions > 0 else 0
                ctr_data.append({
                    'news_id': news_id,
                    'clicks': clicks,
                    'impressions': impressions,
                    'ctr': ctr
                })
            
            # Create dataframe
            ctr_df = pd.DataFrame(ctr_data)
            
            # Merge with news data
            news_with_ctr = pd.merge(news_df, ctr_df, on='news_id', how='left')
            
            # Fill missing CTR values with mean
            mean_ctr = news_with_ctr['ctr'].mean()
            news_with_ctr['ctr'] = news_with_ctr['ctr'].fillna(mean_ctr)
            
            # Save to file
            output_file = os.path.join(self.data_dir, 'headline_ctr_data.csv')
            news_with_ctr.to_csv(output_file, index=False)
            logging.info(f"Saved news articles with CTR data to {output_file}")
            
            return news_with_ctr
            
        except Exception as e:
            logging.error(f"Error calculating CTR from behaviors: {e}")
            return None
    
    def run_full_data_pipeline(self):
        """Run the complete data preparation pipeline"""
        start_time = time.time()
        logging.info("Starting full data pipeline")
        
        # Load data
        news_df = self.load_news_data()
        behaviors_df = self.load_behaviors_data()
        
        if news_df is None:
            logging.error("News data loading failed. Aborting pipeline.")
            return False
        
        # Process news data
        processed_df = self.process_news_data(news_df)
        if processed_df is None:
            logging.error("News data processing failed. Aborting pipeline.")
            return False
        
        # Calculate CTR if behaviors data is available
        if behaviors_df is not None:
            news_with_ctr = self.calculate_ctr_from_behaviors(news_df, behaviors_df)
            if news_with_ctr is None:
                logging.warning("CTR calculation failed. Continuing without CTR data.")
        
        # Compute embeddings
        articles_with_embeddings = self.compute_article_embeddings(processed_df)
        if articles_with_embeddings is None:
            logging.error("Embedding computation failed. Aborting pipeline.")
            return False
        
        # Build FAISS index
        index = self.build_faiss_index(articles_with_embeddings)
        if index is None:
            logging.error("FAISS index building failed. Aborting pipeline.")
            return False
        
        elapsed_time = time.time() - start_time
        logging.info(f"Full data pipeline completed in {elapsed_time:.2f} seconds")
        
        return True


if __name__ == "__main__":
    # Example usage
    connector = MINDDataConnector()
    success = connector.run_full_data_pipeline()
    
    if success:
        print("Data pipeline completed successfully")
    else:
        print("Data pipeline failed")