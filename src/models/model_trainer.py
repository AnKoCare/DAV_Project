"""
Model trainer module for Gaming Behavior Prediction Project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import joblib
import logging
from typing import Dict, Any, Tuple, List, Union
from pathlib import Path

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. XGBoost models will be skipped.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available. LightGBM models will be skipped.")

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class to handle model training and evaluation for gaming behavior prediction"""
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different models based on task type"""
        if self.task_type == 'classification':
            self.models = {
                'random_forest': RandomForestClassifier(random_state=self.random_state),
                'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'svm': SVC(random_state=self.random_state, probability=True)
            }
            
            # Add optional models if available
            if HAS_XGBOOST:
                self.models['xgboost'] = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            
            if HAS_LIGHTGBM:
                self.models['lightgbm'] = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
                
        else:  # regression
            self.models = {
                'random_forest': RandomForestRegressor(random_state=self.random_state),
                'linear_regression': LinearRegression(),
                'svm': SVR()
            }
            
            # Add optional models if available
            if HAS_XGBOOST:
                self.models['xgboost'] = xgb.XGBRegressor(random_state=self.random_state)
            
            if HAS_LIGHTGBM:
                self.models['lightgbm'] = lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, validation_size: float = 0.2) -> Tuple:
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y if self.task_type == 'classification' else None
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state,
            stratify=y_temp if self.task_type == 'classification' else None
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          hyperparams: Dict = None) -> Any:
        """Train a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in available models")
        
        model = self.models[model_name]
        
        # Apply hyperparameters if provided
        if hyperparams:
            model.set_params(**hyperparams)
        
        # Train the model
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        self.trained_models[model_name] = model
        logger.info(f"{model_name} training completed")
        
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        hyperparams_dict: Dict = None) -> Dict:
        """Train all available models"""
        for model_name in self.models.keys():
            hyperparams = hyperparams_dict.get(model_name, {}) if hyperparams_dict else {}
            self.train_single_model(model_name, X_train, y_train, hyperparams)
        
        logger.info("All models training completed")
        return self.trained_models
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate a single model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} is not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            scores = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Add probability predictions if available
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                scores['y_prob'] = y_prob
            
        else:  # regression
            scores = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        scores['y_pred'] = y_pred
        scores['y_true'] = y_test
        
        self.model_scores[model_name] = scores
        logger.info(f"{model_name} evaluation completed")
        
        return scores
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models"""
        for model_name in self.trained_models.keys():
            self.evaluate_model(model_name, X_test, y_test)
        
        return self.model_scores
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                           cv: int = 5) -> Dict:
        """Perform cross-validation on a model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if self.task_type == 'classification':
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = {}
        for score in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=score)
            cv_results[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        
        logger.info(f"Cross-validation completed for {model_name}")
        return cv_results
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                            param_grid: Dict, cv: int = 5) -> Dict:
        """Perform hyperparameter tuning using GridSearchCV"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if self.task_type == 'classification':
            scoring = 'f1_weighted'
        else:
            scoring = 'neg_mean_squared_error'
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        # Update the trained model with best parameters
        self.trained_models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Hyperparameter tuning completed for {model_name}")
        logger.info(f"Best parameters: {results['best_params']}")
        
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str] = None) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} is not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return None
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def select_best_model(self, metric: str = None) -> str:
        """Select the best model based on specified metric"""
        if not self.model_scores:
            raise ValueError("No models have been evaluated yet")
        
        if metric is None:
            if self.task_type == 'classification':
                metric = 'f1'
            else:
                metric = 'r2'
        
        best_score = float('-inf')
        if metric in ['mse', 'rmse', 'mae']:
            best_score = float('inf')
        
        for model_name, scores in self.model_scores.items():
            if metric not in scores:
                continue
            
            score = scores[metric]
            if metric in ['mse', 'rmse', 'mae']:
                if score < best_score:
                    best_score = score
                    self.best_model_name = model_name
            else:
                if score > best_score:
                    best_score = score
                    self.best_model_name = model_name
        
        self.best_model = self.trained_models[self.best_model_name]
        logger.info(f"Best model selected: {self.best_model_name} with {metric}: {best_score}")
        
        return self.best_model_name
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} is not trained yet")
        
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a trained model from disk"""
        loaded_model = joblib.load(filepath)
        self.trained_models[model_name] = loaded_model
        logger.info(f"Model {model_name} loaded from {filepath}")
    
    def get_model_comparison_report(self) -> pd.DataFrame:
        """Generate a comparison report of all models"""
        if not self.model_scores:
            raise ValueError("No models have been evaluated yet")
        
        comparison_data = []
        for model_name, scores in self.model_scores.items():
            row = {'model': model_name}
            for metric, value in scores.items():
                if metric not in ['y_pred', 'y_true', 'y_prob']:
                    row[metric] = value
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('model')
        
        return comparison_df
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Call select_best_model() first or specify model_name")
            model = self.best_model
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} is not trained yet")
            model = self.trained_models[model_name]
        
        return model.predict(X) 