"""
Configuration file for Gaming Behavior Prediction Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Dataset configuration
DATASET_NAME = 'online_gaming_behavior_dataset.csv'  # Kaggle dataset
# DATASET_NAME = 'sample_gaming_behavior_dataset.csv'  # Sample dataset for fallback

# Data settings
TARGET_COLUMN = "EngagementLevel"  # This may vary based on actual dataset
PLAYER_ID_COLUMN = "PlayerID"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering settings
NUMERICAL_FEATURES = [
    'PlayerID', 'Age', 'PlayTimeHours', 'SessionsPerWeek', 
    'AvgSessionDurationMinutes', 'InGamePurchases', 'GameDifficulty',
    'PlayerLevel', 'AchievementsUnlocked'  # Added new Kaggle fields
]

CATEGORICAL_FEATURES = [
    'Gender', 'Location', 'GameGenre', 'GameDifficulty'  # Added GameDifficulty as it's categorical in Kaggle dataset
]

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': RANDOM_STATE
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}

# Dashboard settings
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True

# Visualization settings
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = "viridis"
DPI = 300

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 