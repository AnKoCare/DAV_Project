"""
Main execution script for Gaming Behavior Prediction Project
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import GameBehaviorVisualizer
from config.config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gaming_behavior_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
TARGET_COLUMN = 'EngagementLevel'

# Initial categorical features (before feature engineering)
INITIAL_CATEGORICAL_FEATURES = ['Gender', 'Location', 'Platform', 'GameGenre']

# Model parameters for hyperparameter tuning
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    'logistic_regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

class GamingBehaviorPipeline:
    """Complete pipeline for gaming behavior prediction"""
    
    def __init__(self):
        self.data_loader = DataLoader(RAW_DATA_DIR)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(task_type='classification', random_state=RANDOM_STATE)
        self.visualizer = GameBehaviorVisualizer()
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def create_sample_data(self):
        """Create sample data for demonstration"""
        logger.info("Creating sample data...")
        
        np.random.seed(RANDOM_STATE)
        n_samples = 5000
        
        # Generate realistic gaming behavior data
        sample_data = {
            'PlayerID': range(1, n_samples + 1),
            'Age': np.random.normal(28, 8, n_samples).astype(int).clip(13, 65),
            'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.6, 0.35, 0.05]),
            'Location': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil'], n_samples),
            'Platform': np.random.choice(['PC', 'Mobile', 'Console', 'VR'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
            'GameGenre': np.random.choice(['Action', 'RPG', 'Strategy', 'Sports', 'Simulation', 'Puzzle'], n_samples),
            'PlayTimeHours': np.random.exponential(20, n_samples).clip(0.1, 200),
            'SessionsPerWeek': np.random.poisson(4, n_samples).clip(1, 20),
            'AvgSessionDurationMinutes': np.random.normal(45, 20, n_samples).clip(5, 300),
            'InGamePurchases': np.random.exponential(5, n_samples).clip(0, 100),
            'GameDifficulty': np.random.randint(1, 6, n_samples),
        }
        
        # Create realistic engagement levels based on other features
        df_temp = pd.DataFrame(sample_data)
        
        # Calculate engagement score based on multiple factors
        engagement_score = (
            0.3 * (df_temp['PlayTimeHours'] / df_temp['PlayTimeHours'].max()) +
            0.2 * (df_temp['SessionsPerWeek'] / df_temp['SessionsPerWeek'].max()) +
            0.2 * (df_temp['InGamePurchases'] / df_temp['InGamePurchases'].max()) +
            0.15 * (df_temp['AvgSessionDurationMinutes'] / df_temp['AvgSessionDurationMinutes'].max()) +
            0.15 * (df_temp['GameDifficulty'] / 5)
        )
        
        # Convert to categorical engagement levels
        engagement_levels = pd.cut(engagement_score, 
                                 bins=[0, 0.33, 0.67, 1.0], 
                                 labels=['Low', 'Medium', 'High'])
        sample_data['EngagementLevel'] = engagement_levels
        
        # Save sample data
        df = pd.DataFrame(sample_data)
        df.to_csv(RAW_DATA_DIR / 'sample_gaming_behavior_dataset.csv', index=False)
        
        logger.info(f"Sample data created with {len(df)} records")
        return df
    
    def load_and_prepare_data(self):
        """Load and prepare data for modeling"""
        logger.info("Loading data...")
        
        try:
            # Try to load existing dataset
            self.raw_data = self.data_loader.load_data(DATASET_NAME)
        except FileNotFoundError:
            # Create sample data if dataset doesn't exist
            logger.warning("Dataset not found. Creating sample data...")
            self.raw_data = self.create_sample_data()
            self.data_loader.raw_data = self.raw_data
        
        # Basic data cleaning
        logger.info("Cleaning data...")
        self.processed_data = self.data_loader.basic_cleaning()
        
        # Save processed data
        self.data_loader.save_processed_data('cleaned_gaming_behavior_dataset.csv')
        
        logger.info(f"Data loaded and cleaned. Shape: {self.processed_data.shape}")
        return self.processed_data
    
    def engineer_features(self):
        """Perform feature engineering"""
        logger.info("Engineering features...")
        
        # Create behavioral and engagement features
        self.processed_data = self.feature_engineer.create_engagement_features(self.processed_data)
        self.processed_data = self.feature_engineer.create_behavioral_features(self.processed_data)
        
        # Debug: Check target column
        logger.info(f"Target column '{TARGET_COLUMN}' exists: {TARGET_COLUMN in self.processed_data.columns}")
        if TARGET_COLUMN in self.processed_data.columns:
            logger.info(f"Target column dtype: {self.processed_data[TARGET_COLUMN].dtype}")
            logger.info(f"Target column unique values: {self.processed_data[TARGET_COLUMN].unique()}")
        
        # Encode target column for XGBoost compatibility BEFORE splitting
        if TARGET_COLUMN in self.processed_data.columns and self.processed_data[TARGET_COLUMN].dtype in ['object', 'category']:
            from sklearn.preprocessing import LabelEncoder
            target_encoder = LabelEncoder()
            # Convert category to string first if needed
            if self.processed_data[TARGET_COLUMN].dtype.name == 'category':
                self.processed_data[TARGET_COLUMN] = self.processed_data[TARGET_COLUMN].astype(str)
            
            self.processed_data[TARGET_COLUMN] = target_encoder.fit_transform(self.processed_data[TARGET_COLUMN])
            
            # Update data loader's processed_data
            self.data_loader.processed_data = self.processed_data
            
            # Store target encoder for later use
            self.target_encoder = target_encoder
            logger.info(f"Target labels encoded: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
            logger.info(f"Target column after encoding: {self.processed_data[TARGET_COLUMN].unique()}")
        else:
            logger.info("Target encoding skipped - target column is already numerical or not found")
        
        # Split features and target AFTER encoding
        X, y = self.data_loader.split_features_target(TARGET_COLUMN)
        
        # Debug: Check y values after split
        logger.info(f"Target values after split: {y.unique()}")
        
        # Determine categorical and numerical columns after feature engineering
        categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel',
                           'SessionDurationCategory', 'AgeGroup', 'PlayerType', 'SpenderType']
        
        # Update categorical columns to only include those that exist in the dataframe
        categorical_cols = [col for col in categorical_cols if col in X.columns]
        
        # Get numerical columns (excluding categorical ones)
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Numerical columns: {numerical_cols}")
        
        # Apply preprocessing pipeline
        X_processed = self.feature_engineer.preprocess_pipeline(
            X, 
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            create_features=False  # Features already created
        )
        
        logger.info(f"Feature engineering completed. Final features shape: {X_processed.shape}")
        return X_processed, y
    
    def train_models(self, X, y):
        """Train multiple models"""
        logger.info("Splitting data for training...")
        
        # Split data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.model_trainer.split_data(X, y, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE)
        
        # Train all models with default parameters first
        logger.info("Training models with default parameters...")
        self.model_trainer.train_all_models(self.X_train, self.y_train)
        
        # Evaluate models
        logger.info("Evaluating models...")
        model_scores = self.model_trainer.evaluate_all_models(self.X_test, self.y_test)
        
        # Select best model
        best_model_name = self.model_trainer.select_best_model()
        
        # Optional: Hyperparameter tuning for best model
        logger.info(f"Performing hyperparameter tuning for best model: {best_model_name}")
        if best_model_name in MODEL_PARAMS:
            try:
                tuning_results = self.model_trainer.hyperparameter_tuning(
                    best_model_name, self.X_train, self.y_train, MODEL_PARAMS[best_model_name]
                )
                
                # Re-evaluate with tuned model
                best_scores = self.model_trainer.evaluate_model(best_model_name, self.X_test, self.y_test)
                model_scores[best_model_name] = best_scores
                
                logger.info(f"Hyperparameter tuning completed. Best params: {tuning_results['best_params']}")
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}. Using default parameters.")
        
        logger.info(f"Best model: {best_model_name}")
        return model_scores, best_model_name
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        logger.info("Generating visualizations...")
        
        # Create reports directory
        REPORTS_DIR.mkdir(exist_ok=True)
        
        # 1. Player demographics
        self.visualizer.plot_player_demographics(
            self.processed_data, 
            save_path=REPORTS_DIR / 'player_demographics.png'
        )
        
        # 2. Gaming patterns
        self.visualizer.plot_gaming_patterns(
            self.processed_data,
            save_path=REPORTS_DIR / 'gaming_patterns.png'
        )
        
        # 3. Engagement analysis
        self.visualizer.plot_engagement_analysis(
            self.processed_data, 
            TARGET_COLUMN,
            save_path=REPORTS_DIR / 'engagement_analysis.png'
        )
        
        # 4. Correlation matrix
        self.visualizer.plot_correlation_matrix(
            self.processed_data,
            save_path=REPORTS_DIR / 'correlation_matrix.png'
        )
        
        # 5. Player segments
        self.visualizer.plot_player_segments(
            self.processed_data,
            save_path=REPORTS_DIR / 'player_segments.png'
        )
        
        # 6. Model performance
        if hasattr(self, 'model_scores') and self.model_scores:
            self.visualizer.plot_model_performance(
                self.model_scores,
                task_type='classification',
                save_path=REPORTS_DIR / 'model_performance.png'
            )
        
        # 7. Feature importance
        try:
            importance_df = self.model_trainer.get_feature_importance(
                self.model_trainer.best_model_name,
                feature_names=self.X_train.columns.tolist()
            )
            if importance_df is not None:
                self.visualizer.plot_feature_importance(
                    importance_df,
                    save_path=REPORTS_DIR / 'feature_importance.png'
                )
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")
        
        logger.info("Visualizations generated successfully")
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving models...")
        
        # Save all trained models
        for model_name, model in self.model_trainer.trained_models.items():
            model_path = MODELS_DIR / f"{model_name}_model.joblib"
            self.model_trainer.save_model(model_name, model_path)
        
        # Save feature engineering pipeline
        pipeline_path = MODELS_DIR / "feature_engineering_pipeline.joblib"
        joblib.dump(self.feature_engineer, pipeline_path)
        
        logger.info("Models saved successfully")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        logger.info("Generating analysis report...")
        
        report_content = []
        report_content.append("# Gaming Behavior Prediction - Analysis Report")
        report_content.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Data overview
        report_content.append("## Data Overview")
        report_content.append(f"- Total players analyzed: {len(self.processed_data):,}")
        report_content.append(f"- Features used: {self.X_train.shape[1]}")
        report_content.append(f"- Target variable: {TARGET_COLUMN}")
        report_content.append("")
        
        # Key statistics
        report_content.append("## Key Statistics")
        avg_age = self.processed_data['Age'].mean()
        avg_playtime = self.processed_data['PlayTimeHours'].mean()
        avg_sessions = self.processed_data['SessionsPerWeek'].mean()
        total_revenue = self.processed_data['InGamePurchases'].sum()
        
        report_content.append(f"- Average player age: {avg_age:.1f} years")
        report_content.append(f"- Average play time: {avg_playtime:.1f} hours")
        report_content.append(f"- Average sessions per week: {avg_sessions:.1f}")
        report_content.append(f"- Total revenue: ${total_revenue:,.2f}")
        report_content.append("")
        
        # Engagement distribution
        report_content.append("## Engagement Distribution")
        engagement_counts = self.processed_data[TARGET_COLUMN].value_counts()
        for level, count in engagement_counts.items():
            pct = count / len(self.processed_data) * 100
            report_content.append(f"- {level}: {count:,} players ({pct:.1f}%)")
        report_content.append("")
        
        # Model performance
        if hasattr(self, 'model_scores') and self.model_scores:
            report_content.append("## Model Performance")
            comparison_df = self.model_trainer.get_model_comparison_report()
            report_content.append("```")
            report_content.append(comparison_df.to_string())
            report_content.append("```")
            report_content.append("")
            
            report_content.append(f"Best performing model: {self.model_trainer.best_model_name}")
            report_content.append("")
        
        # Key insights
        report_content.append("## Key Insights")
        
        # Platform analysis
        platform_stats = self.processed_data.groupby('Platform')[['PlayTimeHours', 'InGamePurchases']].mean()
        top_platform_playtime = platform_stats['PlayTimeHours'].idxmax()
        top_platform_revenue = platform_stats['InGamePurchases'].idxmax()
        
        report_content.append(f"- {top_platform_playtime} players have the highest average play time")
        report_content.append(f"- {top_platform_revenue} platform generates the highest average revenue per player")
        
        # Spending behavior
        spender_pct = (self.processed_data['InGamePurchases'] > 0).mean() * 100
        report_content.append(f"- {spender_pct:.1f}% of players make in-game purchases")
        
        # Age insights
        if 'AgeGroup' in self.processed_data.columns:
            age_engagement = pd.crosstab(self.processed_data['AgeGroup'], self.processed_data[TARGET_COLUMN], normalize='index')
            if 'High' in age_engagement.columns:
                top_age_group = age_engagement['High'].idxmax()
                report_content.append(f"- {top_age_group} age group shows highest engagement levels")
        
        report_content.append("")
        
        # Recommendations
        report_content.append("## Recommendations")
        report_content.append("1. **Player Retention**: Focus on converting Medium engagement players to High")
        report_content.append("2. **Monetization**: Target high-engagement players with premium content")
        report_content.append("3. **Platform Strategy**: Optimize experience for top-performing platforms")
        report_content.append("4. **Feature Development**: Prioritize features that correlate with high engagement")
        report_content.append("5. **Marketing**: Tailor campaigns based on player demographics and behavior patterns")
        
        # Save report
        report_path = REPORTS_DIR / "analysis_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Analysis report saved to {report_path}")
    
    def run_complete_pipeline(self):
        """Run the complete analysis pipeline"""
        logger.info("Starting Gaming Behavior Prediction Pipeline...")
        
        try:
            # 1. Load and prepare data
            self.load_and_prepare_data()
            
            # 2. Feature engineering
            X, y = self.engineer_features()
            
            # 3. Train models
            self.model_scores, best_model = self.train_models(X, y)
            
            # 4. Generate visualizations
            self.generate_visualizations()
            
            # 5. Save models
            self.save_models()
            
            # 6. Generate report
            self.generate_report()
            
            logger.info("Pipeline completed successfully!")
            
            # Print summary
            print("\n" + "="*60)
            print("GAMING BEHAVIOR PREDICTION - PIPELINE SUMMARY")
            print("="*60)
            print(f"✅ Data processed: {len(self.processed_data):,} players")
            print(f"✅ Features engineered: {X.shape[1]} features")
            print(f"✅ Models trained: {len(self.model_trainer.trained_models)}")
            print(f"✅ Best model: {best_model}")
            print(f"✅ Visualizations saved to: {REPORTS_DIR}")
            print(f"✅ Models saved to: {MODELS_DIR}")
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    pipeline = GamingBehaviorPipeline()
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main() 