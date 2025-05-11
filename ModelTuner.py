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
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTuner:
    """
    Simple utility for fine-tuning headline CTR models you just trained
    """
    
    def __init__(self, model_dir='model_output', data_dir='agentic_news_editor/processed_data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model = None
        self.model_path = None
        
        # Automatically load the most recent model on startup
        self.load_most_recent_model()
    
    def load_most_recent_model(self):
        """Automatically load the most recently created/modified model"""
        print("\nLooking for the most recent model...")
        models = glob.glob(os.path.join(self.model_dir, '*.pkl'))
        if not models:
            print(f"No models found in {self.model_dir}")
            return False
            
        # Sort models by modification time (newest first)
        models.sort(key=os.path.getmtime, reverse=True)
        most_recent = models[0]
        
        # Load the model
        try:
            with open(most_recent, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data
            self.model_path = most_recent
            
            print(f"\n✅ Loaded most recent model: {os.path.basename(most_recent)}")
            print(f"   Created: {time.ctime(os.path.getmtime(most_recent))}")
            
            # Display metrics if available
            if 'metrics' in model_data:
                train_metrics = model_data['metrics'].get('train', {})
                val_metrics = model_data['metrics'].get('val', {})
                
                if 'r2' in train_metrics:
                    print(f"   Training R²: {train_metrics['r2']:.4f}")
                if val_metrics and 'r2' in val_metrics:
                    print(f"   Validation R²: {val_metrics['r2']:.4f}")
            
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False
    
    def finetune_model(self):
        """Fine-tune the current model with additional training"""
        if not self.model:
            print("❌ No model loaded. Please check that there's a model in your output directory.")
            return
            
        print("\n=== Fine-tune Current Model ===")
        
        # Get tuning parameters
        try:
            lr = float(input("Enter learning rate (default: 0.001): ") or 0.001)
            n_est = int(input("Enter number of additional trees/estimators (default: 50): ") or 50)
            early_stop = int(input("Enter early stopping rounds (default: 10): ") or 10)
            output_name = input("Enter name for fine-tuned model (default: auto-generate): ")
        except ValueError:
            print("❌ Invalid input. Using default values.")
            lr = 0.001
            n_est = 50
            early_stop = 10
            output_name = ""
            
        if not output_name:
            # Auto-generate name based on parameters
            base_name = os.path.basename(self.model_path).replace('.pkl', '')
            output_name = f"{base_name}_tuned_lr{lr}_n{n_est}.pkl"
        elif not output_name.endswith('.pkl'):
            output_name += '.pkl'
            
        # Make sure we have the core headline model class
        try:
            from headline_model_trainer import HeadlineModelTrainer
        except ImportError:
            print("❌ Could not import HeadlineModelTrainer. Make sure headline_model.py is in the same directory.")
            return
            
        print("\n⏳ Loading data and preparing for fine-tuning...")
        
        # Create a trainer instance for loading data
        trainer = HeadlineModelTrainer(processed_data_dir=self.data_dir)
        
        # Load the necessary data
        train_data = trainer.load_data('train')
        val_data = trainer.load_data('val')
        
        if train_data is None:
            print("❌ Could not load training data. Check your data directory.")
            return
            
        # Get the necessary components from the model
        model = self.model['model']
        feature_names = self.model['feature_names']
        use_log_transform = self.model.get('use_log_transform', False)
        
        print("\n⏳ Extracting features from headlines...")
        
        # Extract features
        train_features = trainer.extract_features(train_data['title'].values)
        
        if val_data is not None:
            val_features = trainer.extract_features(val_data['title'].values)
        else:
            val_features = None
            
        # Filter to just the features used by the model
        train_features = train_features[feature_names]
        if val_features is not None:
            val_features = val_features[feature_names]
            
        # Prepare targets
        if use_log_transform:
            train_y = np.log1p(train_data['ctr'].values)
            val_y = np.log1p(val_data['ctr'].values) if val_data is not None else None
        else:
            train_y = train_data['ctr'].values
            val_y = val_data['ctr'].values if val_data is not None else None
            
        # Set up evaluation set
        eval_set = [(train_features, train_y)]
        if val_features is not None and val_y is not None:
            eval_set.append((val_features, val_y))
            
        print(f"\n⏳ Fine-tuning model with {n_est} additional trees...")
        print(f"   Learning rate: {lr}")
        print(f"   Early stopping rounds: {early_stop}")
        
        # Update model parameters
        model.learning_rate = lr
        
        # Continue training from the current model
        try:
            model.fit(
                train_features, train_y,
                xgb_model=model.get_booster(),  # Use existing model
                eval_set=eval_set,
                eval_metric='rmse',
                early_stopping_rounds=early_stop,
                verbose=True
            )
            
            # Evaluate the fine-tuned model
            print("\n⏳ Evaluating fine-tuned model...")
            
            train_pred = model.predict(train_features)
            if use_log_transform:
                train_pred_orig = np.expm1(train_pred)
            else:
                train_pred_orig = train_pred
                
            train_metrics = {
                'mse': mean_squared_error(train_data['ctr'].values, train_pred_orig),
                'rmse': np.sqrt(mean_squared_error(train_data['ctr'].values, train_pred_orig)),
                'mae': mean_absolute_error(train_data['ctr'].values, train_pred_orig),
                'r2': r2_score(train_data['ctr'].values, train_pred_orig)
            }
            
            val_metrics = {}
            if val_features is not None:
                val_pred = model.predict(val_features)
                if use_log_transform:
                    val_pred_orig = np.expm1(val_pred)
                else:
                    val_pred_orig = val_pred
                    
                val_metrics = {
                    'mse': mean_squared_error(val_data['ctr'].values, val_pred_orig),
                    'rmse': np.sqrt(mean_squared_error(val_data['ctr'].values, val_pred_orig)),
                    'mae': mean_absolute_error(val_data['ctr'].values, val_pred_orig),
                    'r2': r2_score(val_data['ctr'].values, val_pred_orig)
                }
                
            # Update metrics in model data
            self.model['metrics'] = {
                'train': train_metrics,
                'val': val_metrics
            }
            
            # Save the fine-tuned model
            output_path = os.path.join(self.model_dir, output_name)
            with open(output_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            # Also save metadata in JSON for easier inspection
            meta_data = {
                'model_type': str(type(model).__name__),
                'use_log_transform': use_log_transform,
                'feature_count': len(feature_names),
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'fine_tuned_from': os.path.basename(self.model_path),
                'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in self.model['metrics'].items()}
            }
            
            meta_path = output_path.replace('.pkl', '_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
                
            print(f"\n✅ Fine-tuned model saved to: {output_name}")
            
            # Print metrics
            print("\nTraining Metrics:")
            for k, v in train_metrics.items():
                print(f"  {k}: {v:.6f}")
                
            if val_metrics:
                print("\nValidation Metrics:")
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.6f}")
                    
            # Update current model
            self.model_path = output_path
            
            # Visualize improvement if matplotlib is available
            try:
                if val_metrics and 'val' in self.model['metrics']:
                    # Compare before and after
                    old_val_r2 = self.model['metrics']['val'].get('r2', 0)
                    new_val_r2 = val_metrics['r2']
                    
                    improvement = (new_val_r2 - old_val_r2) / old_val_r2 * 100 if old_val_r2 > 0 else 0
                    
                    print(f"\nValidation R² improvement: {improvement:.2f}%")
                    
                    # Plot metrics
                    plt.figure(figsize=(8, 5))
                    metrics = ['r2', 'rmse', 'mae']
                    
                    x = np.arange(len(metrics))
                    width = 0.35
                    
                    before_vals = [self.model['metrics']['val'].get(m, 0) for m in metrics]
                    after_vals = [val_metrics.get(m, 0) for m in metrics]
                    
                    plt.bar(x - width/2, before_vals, width, label='Before Fine-tuning')
                    plt.bar(x + width/2, after_vals, width, label='After Fine-tuning')
                    
                    plt.xlabel('Metric')
                    plt.ylabel('Value')
                    plt.title('Model Performance Before vs After Fine-tuning')
                    plt.xticks(x, metrics)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.model_dir, 'tuning_comparison.png'))
                    plt.close()
                    
                    print(f"Performance comparison chart saved to: tuning_comparison.png")
            except Exception as e:
                logging.error(f"Error visualizing improvement: {e}")
                
            return output_path
            
        except Exception as e:
            logging.error(f"Error fine-tuning model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_headlines(self):
        """Test current model with custom headlines"""
        if not self.model:
            print("❌ No model loaded. Please check that there's a model in your output directory.")
            return
            
        print("\n=== Test Headlines with Current Model ===")
        print("Enter headlines (one per line). Enter an empty line when finished.")
        
        headlines = []
        while True:
            headline = input("> ")
            if not headline:
                break
            headlines.append(headline)
            
        if not headlines:
            print("No headlines entered.")
            return
            
        # Load the trainer
        try:
            from headline_model_trainer import HeadlineModelTrainer
        except ImportError:
            print("❌ Could not import HeadlineModelTrainer. Make sure headline_model.py is in the same directory.")
            return
            
        trainer = HeadlineModelTrainer()
        
        # Extract features
        features = trainer.extract_features(headlines)
        
        # Get model components
        model = self.model['model']
        feature_names = self.model['feature_names']
        use_log_transform = self.model.get('use_log_transform', False)
        
        # Select only the features used by the model
        features_filtered = features[feature_names]
        
        # Make predictions
        predictions = model.predict(features_filtered)
        
        # Apply inverse transform if log transform was used
        if use_log_transform:
            predictions = np.expm1(predictions)
            
        # Create results
        results = pd.DataFrame({
            'headline': headlines,
            'predicted_ctr': predictions
        })
        
        # Sort by predicted CTR
        results = results.sort_values('predicted_ctr', ascending=False).reset_index(drop=True)
        
        # Display results
        print("\nHeadlines ranked by predicted CTR:")
        for i, row in results.iterrows():
            print(f"{i+1}. '{row['headline']}'")
            print(f"   Predicted CTR: {row['predicted_ctr']:.6f}")
            
        # Analyze what made the top headline score high
        try:
            if len(results) > 0:
                top_headline = results.iloc[0]['headline']
                print(f"\n=== Analysis of Top Headline ===")
                print(f"'{top_headline}'")
                
                # Check which features contributed most
                features_top = features_filtered.iloc[0]
                
                # Get feature importances
                importances = model.feature_importances_
                
                # Calculate feature contributions
                contributions = []
                for i, feat in enumerate(feature_names):
                    value = features_top[feat]
                    importance = importances[i]
                    contributions.append({
                        'feature': feat,
                        'value': value,
                        'importance': importance,
                        'contribution': abs(value * importance)
                    })
                    
                # Sort by contribution
                contributions = sorted(contributions, key=lambda x: x['contribution'], reverse=True)
                
                print("\nTop 5 contributing features:")
                for i, feat in enumerate(contributions[:5]):
                    print(f"{i+1}. {feat['feature']}: {feat['contribution']:.6f}")
        except Exception as e:
            logging.error(f"Error analyzing top headline: {e}")
    
    def optimize_headline(self):
        """Generate optimized variations of a headline"""
        if not self.model:
            print("❌ No model loaded. Please check that there's a model in your output directory.")
            return
            
        print("\n=== Optimize a Headline ===")
        
        headline = input("Enter a headline to optimize: ")
        if not headline:
            print("No headline provided.")
            return
            
        try:
            n_variations = int(input("Number of variations to generate (default: 10): ") or 10)
        except ValueError:
            print("Invalid input. Using default value.")
            n_variations = 10
            
        # Load the trainer
        try:
            from headline_model_trainer import HeadlineModelTrainer
        except ImportError:
            print("❌ Could not import HeadlineModelTrainer. Make sure headline_model.py is in the same directory.")
            return
            
        trainer = HeadlineModelTrainer()
        
        # Define common headline optimizations
        optimizations = [
            lambda h: f"How {h}",  # Add "How" at the beginning
            lambda h: f"Why {h}",  # Add "Why" at the beginning
            lambda h: f"{h}?",  # Add question mark
            lambda h: f"{h}!",  # Add exclamation mark
            lambda h: f"Top 10 {h}",  # Add list prefix
            lambda h: f"{h}: What You Need to Know",  # Add clarifying suffix
            lambda h: f"Breaking: {h}",  # Add urgency
            lambda h: f"Expert Reveals: {h}",  # Add authority
            lambda h: f"The Truth About {h}",  # Add intrigue
            lambda h: f"You Won't Believe {h}",  # Add surprise
            lambda h: h.title(),  # Title case
            lambda h: h.upper(),  # All caps for emphasis
            lambda h: f"New Study Shows {h}",  # Add credibility
            lambda h: f"{h} [PHOTOS]",  # Add media indicator
            lambda h: f"{h} - Here's Why",  # Add explanation indicator
        ]
        
        # Generate variations (unique ones only)
        variations = [headline]  # Include original
        for optimize_func in optimizations:
            try:
                variation = optimize_func(headline)
                if variation not in variations:
                    variations.append(variation)
                    
                # Stop if we have enough variations
                if len(variations) >= n_variations + 1:  # +1 for original
                    break
            except Exception as e:
                logging.warning(f"Error generating headline variation: {e}")
                
        # Extract features
        features = trainer.extract_features(variations)
        
        # Get model components
        model = self.model['model']
        feature_names = self.model['feature_names']
        use_log_transform = self.model.get('use_log_transform', False)
        
        # Select only the features used by the model
        features_filtered = features[feature_names]
        
        # Make predictions
        predictions = model.predict(features_filtered)
        
        # Apply inverse transform if log transform was used
        if use_log_transform:
            predictions = np.expm1(predictions)
            
        # Create results
        results = pd.DataFrame({
            'headline': variations,
            'predicted_ctr': predictions,
            'is_original': [i == 0 for i in range(len(variations))]
        })
        
        # Sort by predicted CTR
        results = results.sort_values('predicted_ctr', ascending=False).reset_index(drop=True)
        
        # Get original CTR
        original_ctr = results[results['is_original']]['predicted_ctr'].iloc[0]
        
        # Display results
        print(f"\nOriginal headline: '{headline}'")
        print(f"Original predicted CTR: {original_ctr:.6f}")
        
        print("\nTop optimized headlines:")
        for i, row in results[~results['is_original']].head(5).iterrows():
            improvement = (row['predicted_ctr'] / original_ctr - 1) * 100
            print(f"{i+1}. '{row['headline']}'")
            print(f"   Predicted CTR: {row['predicted_ctr']:.6f} ({improvement:.1f}% improvement)")
    
    def feature_importance(self):
        """Analyze feature importance in the current model"""
        if not self.model:
            print("❌ No model loaded. Please check that there's a model in your output directory.")
            return
            
        print("\n=== Feature Importance Analysis ===")
        
        # Get model components
        model = self.model['model']
        feature_names = self.model['feature_names']
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Display top features
        print("\nTop 15 Most Important Features:")
        for i in range(min(15, len(indices))):
            idx = indices[i]
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.6f}")
            
        # Plot feature importances
        try:
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(feature_names))
            top_indices = indices[:top_n]
            
            # Prepare data for horizontal bar chart
            features = [feature_names[i] for i in top_indices[::-1]]
            importance_values = [importances[i] for i in top_indices[::-1]]
            
            # Create bar chart
            plt.barh(range(top_n), importance_values, align='center')
            plt.yticks(range(top_n), features)
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            
            # Save the figure
            output_path = os.path.join(self.model_dir, 'feature_importance.png')
            plt.savefig(output_path)
            plt.close()
            
            print(f"\nFeature importance chart saved to: feature_importance.png")
        except Exception as e:
            logging.error(f"Error plotting feature importance: {e}")
    
    def add_feature_guide(self):
        """Provide guidance for adding a new feature to the model"""
        print("\n=== Adding a New Feature Guide ===")
        print("To add a new feature to your headline model, follow these steps:")
        
        print("\n1. Edit headline_model.py")
        print("   Open the file in your editor and locate the _extract_batch_features method")
        
        print("\n2. Add your new feature code")
        print("   Inside the loop that processes each headline, add code like:")
        
        feature_name = input("\nWhat will you name your new feature? ") or "my_new_feature"
        
        print(f"\n   # Example code to add to the _extract_batch_features method:")
        print(f"   features['{feature_name}'] = # Your feature extraction code here")
        print("   # For example, if you want to count exclamation marks:")
        print(f"   features['{feature_name}'] = headline.count('!')")
        
        print("\n3. Run the full training pipeline again")
        print("   This will retrain the model with your new feature:")
        print("   python headline_model.py")
        
        print("\n4. After training completes, run this utility again")
        print("   It will automatically load your new model with the added feature")
        
        print("\n5. Check feature importance")
        print("   Use option 4 in the main menu to see if your new feature is important")
        
        print("\nCommon Feature Ideas for Headlines:")
        print("- Count specific punctuation marks (!, ?, :, etc.)")
        print("- Detect specific emotional words (amazing, shocking, etc.)")
        print("- Calculate readability scores (Flesch-Kincaid, etc.)")
        print("- Analyze headline structure (starts with number, ends with question, etc.)")
        print("- Count named entities (people, places, organizations)")


def main():
    """Main function for the model tuner utility"""
    tuner = ModelTuner()
    
    while True:
        print("\n==== Headline Model Tuner ====")
        print("1. Fine-tune current model")
        print("2. Test headlines with current model")
        print("3. Optimize a headline")
        print("4. Analyze feature importance")
        print("5. Guide: How to add a new feature")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            tuner.finetune_model()
        elif choice == '2':
            tuner.test_headlines()
        elif choice == '3':
            tuner.optimize_headline()
        elif choice == '4':
            tuner.feature_importance()
        elif choice == '5':
            tuner.add_feature_guide()
        elif choice == '6':
            print("\nExiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()