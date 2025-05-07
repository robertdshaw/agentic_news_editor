import os
import logging
import pandas as pd
import datetime
import argparse
from data_connector import MINDDataConnector
from headline_model_trainer import HeadlineModelTrainer
from headline_metrics import HeadlineMetrics
from headline_learning import HeadlineLearningLoop
from headline_evaluator import HeadlineEvaluator

class NewsEditorController:
    """
    Main controller for the Agentic AI News Editor system.
    Orchestrates data loading, model training, headline rewriting,
    and evaluation across all components.
    """
    
    def __init__(self, data_dir='agentic_news_editor/processed_data'):
        """Initialize the News Editor Controller"""
        self.data_dir = data_dir
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Initialize components
        self.data_connector = MINDDataConnector(data_dir=data_dir)
        self.model_trainer = HeadlineModelTrainer(processed_data_dir=data_dir)
        self.headline_metrics = HeadlineMetrics()
        self.headline_learning = HeadlineLearningLoop()
        self.evaluator = HeadlineEvaluator()
        
        # Status tracking
        self.system_status = {
            'data_prepared': False,
            'model_trained': False,
            'last_run': None,
            'articles_processed': 0,
            'headlines_improved': 0
        }
    
    def prepare_data(self, force_rebuild=False):
        """Prepare data for the news editor system"""
        logging.info("Starting data preparation...")
        
        # Check if data is already prepared
        embeddings_file = os.path.join(self.data_dir, 'articles_with_embeddings.csv')
        faiss_index_file = 'articles_faiss.index'
        
        if not force_rebuild and os.path.exists(embeddings_file) and os.path.exists(faiss_index_file):
            logging.info("Data already prepared. Use force_rebuild=True to rebuild.")
            self.system_status['data_prepared'] = True
            return True
        
        # Run the full data pipeline
        success = self.data_connector.run_full_data_pipeline()
        
        if success:
            self.system_status['data_prepared'] = True
            logging.info("Data preparation completed successfully")
        else:
            logging.error("Data preparation failed")
        
        return success
    
    def train_headline_model(self, force_retrain=False):
        """Train the headline CTR prediction model"""
        logging.info("Starting headline model training...")
        
        # Check if model already exists
        model_file = 'headline_ctr_model.pkl'
        
        if not force_retrain and os.path.exists(model_file):
            logging.info("Headline model already exists. Use force_retrain=True to retrain.")
            self.system_status['model_trained'] = True
            return True
        
        # Run the model training pipeline
        result = self.model_trainer.run_training_pipeline()
        
        if result is not None:
            self.system_status['model_trained'] = True
            logging.info(f"Headline model training completed successfully. R² = {result['r2']:.4f}")
            return True
        else:
            logging.error("Headline model training failed")
            return False
    
    def run_headline_learning_cycle(self):
        """Execute a cycle of the headline learning loop"""
        logging.info("Running headline learning cycle...")
        
        # Check if we have curated articles to learn from
        curated_file = 'curated_full_daily_output.csv'
        if not os.path.exists(curated_file):
            logging.warning("No curated articles found. Learning cycle skipped.")
            return False
        
        # Load curated articles
        try:
            curated_df = pd.read_csv(curated_file)
            
            # Update the learning system
            count = self.headline_learning.add_headlines_from_dataframe(curated_df)
            logging.info(f"Added {count} headline pairs to learning system")
            
            # Generate insights report
            if count > 0:
                self.headline_learning.prompt_improvement_report()
                
                # Check if enough data for retraining
                if len(curated_df) > 50:
                    self.headline_learning.retrain_model()
            
            return count > 0
            
        except Exception as e:
            logging.error(f"Error in headline learning cycle: {e}")
            return False
    
    def run_evaluation(self):
        """Run a complete evaluation of the system"""
        logging.info("Running system evaluation...")
        
        result = self.evaluator.run_full_evaluation()
        
        if result is not None:
            logging.info(f"Evaluation completed. Report available at: {result['report_path']}")
            return result
        else:
            logging.error("Evaluation failed")
            return None
    
    def run_complete_pipeline(self, daily_run=False):
        """Run the complete pipeline from data prep to evaluation"""
        start_time = datetime.datetime.now()
        logging.info(f"Starting complete pipeline at {start_time}")
        
        # Prepare data (skip if daily run)
        if not daily_run or not self.system_status['data_prepared']:
            self.prepare_data(force_rebuild=not daily_run)
        
        # Train headline model (skip if daily run)
        if not daily_run or not self.system_status['model_trained']:
            self.train_headline_model(force_retrain=not daily_run)
        
        # Run headline learning cycle
        self.run_headline_learning_cycle()
        
        # Run evaluation
        eval_result = self.run_evaluation()
        
        # Update status
        self.system_status['last_run'] = datetime.datetime.now()
        elapsed_time = (self.system_status['last_run'] - start_time).total_seconds()
        
        logging.info(f"Complete pipeline finished in {elapsed_time:.1f} seconds")
        
        # Return evaluation results
        return eval_result
    
    def get_system_status(self):
        """Get the current status of the system"""
        # Update status with latest info
        if os.path.exists('headline_learning_data.csv'):
            try:
                learning_df = pd.read_csv('headline_learning_data.csv')
                self.system_status['headlines_processed'] = len(learning_df)
                self.system_status['headlines_improved'] = (learning_df['headline_improvement'] > 0).sum()
            except Exception as e:
                logging.error(f"Error updating status from learning data: {e}")
        
        return self.system_status


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Agentic AI News Editor Controller')
    parser.add_argument('--daily', action='store_true', help='Run daily update (skips data prep and model training)')
    parser.add_argument('--full', action='store_true', help='Run full pipeline including data prep and model training')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation only')
    parser.add_argument('--learn', action='store_true', help='Run headline learning cycle only')
    args = parser.parse_args()
    
    # Initialize controller
    controller = NewsEditorController()
    
    if args.evaluate:
        print("Running evaluation only...")
        controller.run_evaluation()
    elif args.learn:
        print("Running headline learning cycle only...")
        controller.run_headline_learning_cycle()
    elif args.daily:
        print("Running daily update...")
        controller.run_complete_pipeline(daily_run=True)
    elif args.full:
        print("Running full pipeline...")
        controller.run_complete_pipeline(daily_run=False)
    else:
        print("No action specified. Use --daily, --full, --evaluate, or --learn")