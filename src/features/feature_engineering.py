"""
Feature engineering module for Gaming Behavior Prediction Project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Class to handle feature engineering for gaming behavior prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_names = []
        
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement-related features"""
        df_new = df.copy()
        
        # Play intensity features
        if 'PlayTimeHours' in df_new.columns and 'SessionsPerWeek' in df_new.columns:
            df_new['AvgHoursPerSession'] = df_new['PlayTimeHours'] / (df_new['SessionsPerWeek'] + 1e-8)
            df_new['PlayIntensity'] = df_new['PlayTimeHours'] * df_new['SessionsPerWeek']
        
        # Purchase behavior features
        if 'InGamePurchases' in df_new.columns:
            df_new['IsPurchaser'] = (df_new['InGamePurchases'] > 0).astype(int)
            if 'PlayTimeHours' in df_new.columns:
                df_new['PurchasePerHour'] = df_new['InGamePurchases'] / (df_new['PlayTimeHours'] + 1e-8)
        
        # Session duration features
        if 'AvgSessionDurationMinutes' in df_new.columns:
            df_new['SessionDurationCategory'] = pd.cut(
                df_new['AvgSessionDurationMinutes'], 
                bins=[0, 30, 60, 120, float('inf')],
                labels=['Short', 'Medium', 'Long', 'VeryLong']
            )
        
        # Age group features
        if 'Age' in df_new.columns:
            df_new['AgeGroup'] = pd.cut(
                df_new['Age'],
                bins=[0, 18, 25, 35, 45, float('inf')],
                labels=['Teen', 'YoungAdult', 'Adult', 'MiddleAged', 'Senior']
            )
        
        # Game difficulty engagement
        if 'GameDifficulty' in df_new.columns and 'PlayTimeHours' in df_new.columns:
            df_new['DifficultyEngagement'] = df_new['GameDifficulty'] * df_new['PlayTimeHours']
        
        logger.info(f"Created engagement features. New shape: {df_new.shape}")
        return df_new
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features"""
        df_new = df.copy()
        
        # Player type based on play patterns
        if all(col in df_new.columns for col in ['PlayTimeHours', 'SessionsPerWeek', 'InGamePurchases']):
            # Casual vs Hardcore classification
            df_new['PlayerType'] = 'Casual'
            hardcore_mask = (
                (df_new['PlayTimeHours'] > df_new['PlayTimeHours'].quantile(0.75)) &
                (df_new['SessionsPerWeek'] > df_new['SessionsPerWeek'].median())
            )
            df_new.loc[hardcore_mask, 'PlayerType'] = 'Hardcore'
            
            # Spender classification
            df_new['SpenderType'] = 'NonSpender'
            light_spender_mask = (df_new['InGamePurchases'] > 0) & (df_new['InGamePurchases'] <= df_new['InGamePurchases'].quantile(0.5))
            heavy_spender_mask = df_new['InGamePurchases'] > df_new['InGamePurchases'].quantile(0.5)
            
            df_new.loc[light_spender_mask, 'SpenderType'] = 'LightSpender'
            df_new.loc[heavy_spender_mask, 'SpenderType'] = 'HeavySpender'
        
        # Consistency metrics
        if 'SessionsPerWeek' in df_new.columns:
            df_new['IsRegularPlayer'] = (df_new['SessionsPerWeek'] >= 3).astype(int)
        
        # Platform preference insights
        if 'Platform' in df_new.columns and 'PlayTimeHours' in df_new.columns:
            platform_avg = df_new.groupby('Platform')['PlayTimeHours'].transform('mean')
            df_new['PlatformEngagementRatio'] = df_new['PlayTimeHours'] / (platform_avg + 1e-8)
        
        logger.info(f"Created behavioral features. New shape: {df_new.shape}")
        return df_new
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  categorical_cols: List[str],
                                  encoding_method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in df_encoded.columns:
                logger.warning(f"Column {col} not found in dataframe")
                continue
                
            if encoding_method == 'onehot':
                if col not in self.encoders:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_features = self.encoders[col].fit_transform(df_encoded[[col]])
                else:
                    encoded_features = self.encoders[col].transform(df_encoded[[col]])
                
                # Create column names
                feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df_encoded.index)
                
                # Drop original column and add encoded features
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                
            elif encoding_method == 'label':
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
        
        logger.info(f"Encoded categorical features using {encoding_method}. New shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                               numerical_cols: List[str],
                               scaling_method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        
        for col in numerical_cols:
            if col not in df_scaled.columns:
                logger.warning(f"Column {col} not found in dataframe")
                continue
            
            if scaling_method == 'standard':
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_scaled[col] = self.scalers[col].fit_transform(df_scaled[[col]])
                else:
                    df_scaled[col] = self.scalers[col].transform(df_scaled[[col]])
        
        logger.info(f"Scaled numerical features using {scaling_method}")
        return df_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       k: int = 10, task_type: str = 'classification') -> pd.DataFrame:
        """Select top k features based on statistical tests"""
        
        if task_type == 'classification':
            score_func = f_classif
        else:
            score_func = f_regression
        
        if 'feature_selector' not in self.feature_selectors:
            self.feature_selectors['feature_selector'] = SelectKBest(score_func=score_func, k=k)
            X_selected = self.feature_selectors['feature_selector'].fit_transform(X, y)
        else:
            X_selected = self.feature_selectors['feature_selector'].transform(X)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selectors['feature_selector'].get_support()]
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Selected {k} features from {X.shape[1]} features")
        logger.info(f"Selected features: {list(selected_features)}")
        
        return X_selected_df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between specified feature pairs"""
        df_interaction = df.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 in df_interaction.columns and feature2 in df_interaction.columns:
                interaction_name = f"{feature1}_x_{feature2}"
                df_interaction[interaction_name] = df_interaction[feature1] * df_interaction[feature2]
                logger.info(f"Created interaction feature: {interaction_name}")
        
        return df_interaction
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                 numerical_cols: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numerical columns"""
        df_poly = df.copy()
        
        for col in numerical_cols:
            if col in df_poly.columns:
                for d in range(2, degree + 1):
                    poly_col_name = f"{col}_poly_{d}"
                    df_poly[poly_col_name] = df_poly[col] ** d
        
        logger.info(f"Created polynomial features up to degree {degree}")
        return df_poly
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering steps"""
        summary = {
            'scalers_fitted': list(self.scalers.keys()),
            'encoders_fitted': list(self.encoders.keys()),
            'feature_selectors_fitted': list(self.feature_selectors.keys()),
            'total_features_created': len(self.feature_names)
        }
        return summary
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                          numerical_cols: List[str],
                          categorical_cols: List[str],
                          target_col: str = None,
                          create_features: bool = True) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        
        # Step 1: Create engineered features
        if create_features:
            df = self.create_engagement_features(df)
            df = self.create_behavioral_features(df)
        
        # Step 2: Encode categorical features
        df = self.encode_categorical_features(df, categorical_cols)
        
        # Step 3: Scale numerical features
        df = self.scale_numerical_features(df, numerical_cols)
        
        logger.info(f"Preprocessing pipeline completed. Final shape: {df.shape}")
        return df 