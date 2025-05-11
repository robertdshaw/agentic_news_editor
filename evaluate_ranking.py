"""
Evaluate headline model ranking ability on validation data
"""
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your HeadlineModelTrainer class
from headline_model_trainer import HeadlineModelTrainer

def main():
    # Create trainer instance
    trainer = HeadlineModelTrainer()
    
    # Load model
    model_path = os.path.join(trainer.output_dir, 'headline_classifier_model.pkl')
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return
    
    # Load model data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Load validation data
    val_data = trainer.load_data('val')
    if val_data is None:
        logging.error("Validation data not found")
        return
    
    # Run ranking evaluation
    evaluation = trainer.evaluate_model_ranking(model_data, val_data)
    
    # Display summary
    print("\nRanking Evaluation Summary:")
    print(f"Spearman Rank Correlation: {evaluation['rank_correlation']:.4f}")
    print(f"Top Bucket CTR Lift: {evaluation['top_bucket_lift']:.1f}%")
    print(f"Top 3 Buckets CTR Lift: {evaluation['top_3_buckets_lift']:.1f}%")
    print(f"\nResults saved to {trainer.output_dir}/ranking_performance.png")
    print(f"Detailed data saved to {trainer.output_dir}/headline_ranking_results.csv")

if __name__ == "__main__":
    main()
    """
Evaluate headline model ranking ability on validation data
"""
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your HeadlineModelTrainer class
from headline_model_trainer import HeadlineModelTrainer

def main():
    # Create trainer instance
    trainer = HeadlineModelTrainer()
    
    # Load model
    model_path = os.path.join(trainer.output_dir, 'headline_classifier_model.pkl')
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return
    
    # Load model data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Load validation data
    val_data = trainer.load_data('val')
    if val_data is None:
        logging.error("Validation data not found")
        return
    
    # Run ranking evaluation
    evaluation = trainer.evaluate_model_ranking(model_data, val_data)
    
    # Display summary
    print("\nRanking Evaluation Summary:")
    print(f"Spearman Rank Correlation: {evaluation['rank_correlation']:.4f}")
    print(f"Top Bucket CTR Lift: {evaluation['top_bucket_lift']:.1f}%")
    print(f"Top 3 Buckets CTR Lift: {evaluation['top_3_buckets_lift']:.1f}%")
    print(f"\nResults saved to {trainer.output_dir}/ranking_performance.png")
    print(f"Detailed data saved to {trainer.output_dir}/headline_ranking_results.csv")

if __name__ == "__main__":
    main()