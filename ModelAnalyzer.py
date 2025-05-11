import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import json
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelAnalyzer:
    """Utility class for analyzing and improving headline CTR prediction models"""
    
    def __init__(self, model_dir='model_output', cache_dir='model_cache'):
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.current_model = None
        self.current_model_path = None
        
        # Check directories
        if not os.path.exists(model_dir):
            logging.error(f"Model directory not found: {model_dir}")
        if not os.path.exists(cache_dir):
            logging.error(f"Cache directory not found: {cache_dir}")
    
    def list_models(self):
        """List all available models in the model directory"""
        models = glob.glob(os.path.join(self.model_dir, '*.pkl'))
        print("\n=== Available Models ===")
        if not models:
            print("No models found in", self.model_dir)
            return []
            
        for i, model_path in enumerate(models):
            model_name = os.path.basename(model_path)
            model_time = os.path.getmtime(model_path)
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            
            # Try to get metadata if available
            meta_path = model_path.replace('.pkl', '_meta.json')
            meta_info = "No metadata"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    r2_train = meta['metrics']['train']['r2'] if 'train' in meta['metrics'] else "N/A"
                    r2_val = meta['metrics']['val']['r2'] if 'val' in meta['metrics'] else "N/A"
                    meta_info = f"Train R²: {r2_train:.4f}, Val R²: {r2_val:.4f}"
                except:
                    pass
            
            print(f"{i+1}. {model_name} - {meta_info} - {model_size:.2f} MB")
        
        return models
    
    def list_cached_data(self):
        """List all available cached data"""
        cache_files = glob.glob(os.path.join(self.cache_dir, '*.pkl'))
        print("\n=== Available Cached Data ===")
        if not cache_files:
            print("No cache files found in", self.cache_dir)
            return []
            
        # Group by prefix
        prefixes = {}
        for file_path in cache_files:
            file_name = os.path.basename(file_path)
            prefix = file_name.split('_')[0]
            if prefix not in prefixes:
                prefixes[prefix] = []
            prefixes[prefix].append(file_path)
        
        # Print grouped
        for prefix, files in prefixes.items():
            print(f"\n{prefix.upper()} Data:")
            for i, file_path in enumerate(files):
                file_name = os.path.basename(file_path)
                file_time = os.path.getmtime(file_path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                print(f"  {i+1}. {file_name} - {file_size:.2f} MB")
        
        return cache_files
    
    def load_model(self, model_index_or_path):
        """Load a model by index or path"""
        # If it's an index, get the path
        if isinstance(model_index_or_path, int):
            models = glob.glob(os.path.join(self.model_dir, '*.pkl'))
            if not models:
                logging.error("No models found")
                return False
                
            if model_index_or_path < 1 or model_index_or_path > len(models):
                logging.error(f"Invalid model index: {model_index_or_path}")
                return False
                
            model_path = models[model_index_or_path - 1]
        else:
            model_path = model_index_or_path
            if not os.path.exists(model_path):
                model_path = os.path.join(self.model_dir, model_path)
            
            if not os.path.exists(model_path):
                logging.error(f"Model not found: {model_path}")
                return False
        
        # Load the model
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.current_model = model_data
            self.current_model_path = model_path
            
            print(f"\nLoaded model: {os.path.basename(model_path)}")
            print(f"Training date: {model_data.get('training_date', 'Unknown')}")
            print(f"Log transform: {model_data.get('use_log_transform', 'Unknown')}")
            print(f"Feature count: {len(model_data.get('feature_names', []))}")
            
            # Display metrics if available
            if 'metrics' in model_data:
                train_metrics = model_data['metrics'].get('train', {})
                val_metrics = model_data['metrics'].get('val', {})
                
                print("\nTraining Metrics:")
                for k, v in train_metrics.items():
                    print(f"  {k}: {v:.6f}")
                    
                if val_metrics:
                    print("\nValidation Metrics:")
                    for k, v in val_metrics.items():
                        print(f"  {k}: {v:.6f}")
            
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def analyze_features(self):
        """Analyze features in the current model"""
        if not self.current_model:
            print("No model loaded. Use load_model() first.")
            return
            
        if 'model' not in self.current_model or 'feature_names' not in self.current_model:
            print("Current model is missing required data.")
            return
            
        model = self.current_model['model']
        feature_names = self.current_model['feature_names']
        
        # Get feature importances
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n=== Feature Importance Analysis ===")
            print("\nTop 20 Most Important Features:")
            for i in range(min(20, len(indices))):
                idx = indices[i]
                print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.6f}")
                
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            top_indices = indices[:20]
            plt.barh([feature_names[i] for i in top_indices[::-1]], 
                     [importances[i] for i in top_indices[::-1]])
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.show()
            
            # Plot correlations if feature data is available
            # (Can be implemented if cached feature data is available)
            
        except Exception as e:
            logging.error(f"Error analyzing features: {e}")
    
    def finetune_model(self, learning_rate=0.001, additional_estimators=50, early_stopping_rounds=20):
        """Fine-tune the current model with additional training"""
        if not self.current_model:
            print("No model loaded. Use load_model() first.")
            return
            
        if 'model' not in self.current_model:
            print("Current model is missing required data.")
            return
            
        # Load training data
        train_data = self._load_cached_data('train_features')
        train_target = self._load_cached_data('train_target')
        val_data = self._load_cached_data('val_features')
        val_target = self._load_cached_data('val_target')
        
        if train_data is None or train_target is None:
            print("Could not load training data from cache.")
            print("Make sure you've run the full training pipeline first.")
            return
            
        # Extract necessary components
        model = self.current_model['model']
        feature_names = self.current_model['feature_names']
        use_log_transform = self.current_model.get('use_log_transform', False)
        
        # Ensure we have the right features
        train_features = train_data[feature_names] if isinstance(train_data, pd.DataFrame) else train_data
        val_features = val_data[feature_names] if val_data is not None and isinstance(val_data, pd.DataFrame) else val_data
        
        # Transform targets if needed
        if use_log_transform:
            train_y = np.log1p(train_target)
            val_y = np.log1p(val_target) if val_target is not None else None
        else:
            train_y = train_target
            val_y = val_target
        
        # Continue training
        print(f"\nFine-tuning model with {additional_estimators} additional estimators...")
        print(f"Learning rate: {learning_rate}")
        
        try:
            # Create evaluation set
            eval_set = [(train_features, train_y)]
            if val_features is not None and val_y is not None:
                eval_set.append((val_features, val_y))
            
            # Continue training
            model.learning_rate = learning_rate  # Update learning rate
            model.n_estimators += additional_estimators  # Add more estimators
            
            # Fit model
            model.fit(
                train_features, train_y,
                xgb_model=model.get_booster(),  # Use existing model
                eval_set=eval_set,
                eval_metric='rmse',
                early_stopping_rounds=early_stopping_rounds,
                verbose=True
            )
            
            # Evaluate
            train_pred = model.predict(train_features)
            if use_log_transform:
                train_pred = np.expm1(train_pred)
                
            train_metrics = {
                'mse': mean_squared_error(train_target, train_pred),
                'rmse': np.sqrt(mean_squared_error(train_target, train_pred)),
                'mae': mean_absolute_error(train_target, train_pred),
                'r2': r2_score(train_target, train_pred)
            }
            
            val_metrics = {}
            if val_features is not None and val_target is not None:
                val_pred = model.predict(val_features)
                if use_log_transform:
                    val_pred = np.expm1(val_pred)
                    
                val_metrics = {
                    'mse': mean_squared_error(val_target, val_pred),
                    'rmse': np.sqrt(mean_squared_error(val_target, val_pred)),
                    'mae': mean_absolute_error(val_target, val_pred),
                    'r2': r2_score(val_target, val_pred)
                }
            
            # Update model data
            self.current_model['metrics'] = {
                'train': train_metrics,
                'val': val_metrics
            }
            
            # Save fine-tuned model
            output_file = os.path.basename(self.current_model_path).replace(
                '.pkl', f'_finetuned_lr{learning_rate}_n{additional_estimators}.pkl'
            )
            output_path = os.path.join(self.model_dir, output_file)
            
            with open(output_path, 'wb') as f:
                pickle.dump(self.current_model, f)
                
            print(f"\nFine-tuned model saved to: {output_file}")
            print("\nTraining Metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.6f}")
                
            if val_metrics:
                print("\nValidation Metrics:")
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.6f}")
                    
            # Update current model path
            self.current_model_path = output_path
            
            return output_path
            
        except Exception as e:
            logging.error(f"Error fine-tuning model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_headlines(self):
        """Test the current model on example headlines"""
        if not self.current_model:
            print("No model loaded. Use load_model() first.")
            return
            
        if 'model' not in self.current_model:
            print("Current model is missing required data.")
            return
            
        # Get some example headlines
        print("\n=== Test Headlines ===")
        print("Enter headlines to test (one per line). Type 'done' when finished:")
        
        headlines = []
        while True:
            line = input("> ")
            if line.lower() == 'done':
                break
            headlines.append(line)
            
        if not headlines:
            print("No headlines provided.")
            return
            
        # Load headline model trainer (simplified version)
        from headline_model_trainer import HeadlineModelTrainer
        
        # Create trainer instance
        trainer = HeadlineModelTrainer()
        
        # Extract features
        print("\nExtracting features...")
        features = trainer.extract_features(headlines)
        
        # Get predictions
        print("Making predictions...")
        predictions = trainer.predict_ctr(headlines, self.current_model)
        
        # Display results
        print("\n=== Prediction Results ===")
        results = pd.DataFrame({
            'headline': headlines,
            'predicted_ctr': predictions
        })
        
        results = results.sort_values('predicted_ctr', ascending=False).reset_index(drop=True)
        
        for i, row in results.iterrows():
            print(f"{i+1}. '{row['headline']}'")
            print(f"   Predicted CTR: {row['predicted_ctr']:.6f}")
        
        # Analyze top headline
        top_headline = results.iloc[0]['headline']
        trainer.headline_analysis(top_headline, self.current_model)
    
    def optimize_headlines(self):
        """Optimize headlines with the current model"""
        if not self.current_model:
            print("No model loaded. Use load_model() first.")
            return
            
        if 'model' not in self.current_model:
            print("Current model is missing required data.")
            return
            
        # Get headline to optimize
        print("\n=== Headline Optimization ===")
        headline = input("Enter a headline to optimize: ")
        if not headline:
            print("No headline provided.")
            return
            
        n_variations = int(input("Number of variations to generate (default 10): ") or 10)
            
        # Load headline model trainer
        from headline_model_trainer import HeadlineModelTrainer
        
        # Create trainer instance
        trainer = HeadlineModelTrainer()
        
        # Optimize headline
        results = trainer.optimize_headline(
            headline, 
            n_variations=n_variations,
            model_data=self.current_model
        )
        
        if results is None:
            print("Failed to optimize headline.")
            return
            
        # Display results
        print(f"\nOriginal headline: '{headline}'")
        original_ctr = results[results['is_original']]['predicted_ctr'].iloc[0]
        print(f"Original CTR: {original_ctr:.6f}")
        
        print("\nTop optimized headlines:")
        for i, row in results[~results['is_original']].head(5).iterrows():
            improvement = (row['predicted_ctr'] / original_ctr - 1) * 100
            print(f"{i+1}. '{row['headline']}'")
            print(f"   Predicted CTR: {row['predicted_ctr']:.6f} ({improvement:.1f}% improvement)")
    
    def add_custom_feature(self):
        """Add a custom feature to the model"""
        print("\n=== Add Custom Feature ===")
        print("This functionality requires modifying the HeadlineModelTrainer class.")
        print("You would need to:")
        print("1. Add the new feature extraction logic to extract_features()")
        print("2. Retrain the model with the new feature")
        print("\nThis is best done by editing the headline_model.py file directly.")
    
    def _load_cached_data(self, data_type):
        """Load data from cache"""
        cache_path = os.path.join(self.cache_dir, f"{data_type}.pkl")
        if not os.path.exists(cache_path):
            # Try to find files that match the pattern
            matches = glob.glob(os.path.join(self.cache_dir, f"{data_type}*.pkl"))
            if not matches:
                return None
            cache_path = matches[0]
            
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logging.error(f"Error loading {data_type} from cache: {e}")
            return None


def main():
    """Main function to run the model analyzer utility"""
    analyzer = ModelAnalyzer()
    
    print("\n=== Headline CTR Model Analyzer ===")
    print("This utility helps you analyze and improve your trained models.")
    
    while True:
        print("\n=== Menu ===")
        print("1. List available models")
        print("2. List cached data")
        print("3. Load a model")
        print("4. Analyze feature importance")
        print("5. Fine-tune current model")
        print("6. Test headlines")
        print("7. Optimize headlines")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            analyzer.list_models()
        elif choice == '2':
            analyzer.list_cached_data()
        elif choice == '3':
            models = analyzer.list_models()
            if models:
                model_choice = input("\nEnter model number to load: ")
                try:
                    analyzer.load_model(int(model_choice))
                except ValueError:
                    print("Invalid input. Enter a number.")
        elif choice == '4':
            analyzer.analyze_features()
        elif choice == '5':
            if not analyzer.current_model:
                print("No model loaded. Use option 3 to load a model first.")
                continue
                
            lr = float(input("Enter learning rate (default 0.001): ") or 0.001)
            n_est = int(input("Enter number of additional estimators (default 50): ") or 50)
            early_stop = int(input("Enter early stopping rounds (default 20): ") or 20)
            
            analyzer.finetune_model(learning_rate=lr, additional_estimators=n_est, early_stopping_rounds=early_stop)
        elif choice == '6':
            analyzer.test_headlines()
        elif choice == '7':
            analyzer.optimize_headlines()
        elif choice == '8':
            print("\nExiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")


if __name__ == "__main__":
    main()