# Getting Started - Gaming Behavior Prediction

Welcome to the Gaming Behavior Prediction project! This guide will help you set up and run the complete analysis pipeline.

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip3 install -r requirements.txt
```

### 2. Run Complete Analysis
```bash
# Run the full pipeline (data processing, modeling, visualization)
python3 src/main.py
```

### 3. Launch Interactive Dashboard
```bash
# Start the interactive dashboard
python3 run_dashboard.py
```
Then open http://127.0.0.1:8050 in your browser.


## Project Components

### 🔧 Core Modules

**Data Processing (`src/data/`)**
- `data_loader.py`: Load and clean gaming behavior data
- Handles missing values, duplicates, and basic preprocessing

**Feature Engineering (`src/features/`)**
- `feature_engineering.py`: Create engagement and behavioral features
- Player segmentation, interaction features, and scaling

**Machine Learning (`src/models/`)**
- `model_trainer.py`: Train and evaluate multiple ML models
- Supports Random Forest, XGBoost, LightGBM, SVM, Logistic Regression

**Visualization (`src/visualization/`)**
- `visualizer.py`: Generate comprehensive analytics charts
- Player demographics, gaming patterns, correlation analysis

### 📊 Analytics Dashboard

The interactive dashboard provides:
- **Real-time Filtering**: By platform, age range, engagement level
- **Key Metrics**: Player count, average playtime, revenue
- **Visual Analytics**: Demographics, engagement patterns, correlations
- **Player Segmentation**: Casual, Hardcore, Whale, VIP segments

### 📈 Analysis Outputs

After running the pipeline, you'll find:

**Reports (`reports/`)**
- `analysis_report.md`: Comprehensive findings and recommendations
- `player_demographics.png`: Age, gender, location, platform analysis
- `gaming_patterns.png`: Play time, sessions, purchases distribution
- `engagement_analysis.png`: Engagement levels by demographics
- `correlation_matrix.png`: Feature relationships
- `player_segments.png`: Player behavior segmentation
- `model_performance.png`: ML model comparison
- `feature_importance.png`: Key predictive factors

**Models (`models/`)**
- Trained ML models saved as `.joblib` files
- Feature engineering pipeline for production use

**Data (`data/`)**
- `raw/`: Original dataset
- `processed/`: Cleaned and feature-engineered data

## Key Features

### 🎯 Predictive Modeling
- **Player Retention Prediction**: Identify players likely to churn
- **Engagement Level Classification**: Low/Medium/High engagement
- **Revenue Prediction**: Estimate player lifetime value
- **Behavior Pattern Recognition**: Casual vs Hardcore classification

### 📊 Business Analytics
- **Player Segmentation**: Strategic grouping for targeted marketing
- **Platform Performance**: Cross-platform engagement comparison
- **Demographic Insights**: Age, gender, location impact analysis
- **Monetization Analysis**: Spending behavior and revenue optimization

### 🔍 Advanced Features
- **Feature Importance**: Understand key engagement drivers
- **Correlation Analysis**: Discover hidden relationships
- **Interactive Filtering**: Real-time data exploration
- **Automated Reporting**: Generated insights and recommendations

## Dataset Information

The project works with gaming behavior data containing:

**Player Demographics**
- Age, Gender, Location
- Platform preference (PC, Mobile, Console, VR)

**Gaming Behavior**
- Play time hours, Sessions per week
- Average session duration
- Game genre preferences, Difficulty levels

**Engagement Metrics**
- In-game purchases
- Engagement level (Low/Medium/High)
- Player segments and behavior patterns

## Customization

### Adding New Features
```python
# In src/features/feature_engineering.py
def create_custom_features(self, df):
    # Add your custom feature engineering logic
    df['custom_metric'] = df['feature1'] / df['feature2']
    return df
```

### Adding New Models
```python
# In src/models/model_trainer.py
def _initialize_models(self):
    self.models['custom_model'] = YourCustomModel()
```

### Customizing Visualizations
```python
# In src/visualization/visualizer.py
def plot_custom_analysis(self, df):
    # Add your custom visualization logic
    pass
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the project root directory
cd path/to/DAV_Project
python src/main.py
```

**Missing Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

**Dashboard Not Loading**
```bash
# Check if port 8050 is available
# Or modify DASHBOARD_PORT in config/config.py
```

**Memory Issues with Large Datasets**
- Reduce sample size in configuration
- Use data chunking for very large files
- Consider upgrading system memory

### Getting Help

1. **Check Logs**: Look at `gaming_behavior_prediction.log` for detailed error messages
2. **Validate Data**: Ensure your dataset matches expected format
3. **Review Configuration**: Check `config/config.py` for correct settings
4. **Test Components**: Run individual modules to isolate issues

## Next Steps

### Production Deployment
1. **Model Serving**: Deploy trained models using Flask/FastAPI
2. **Database Integration**: Connect to production gaming databases
3. **Real-time Pipeline**: Set up streaming data processing
4. **Monitoring**: Implement model performance tracking

### Advanced Analytics
1. **Time Series Analysis**: Track engagement trends over time
2. **Cohort Analysis**: Study player retention patterns
3. **A/B Testing**: Experiment with game features
4. **Churn Prediction**: Proactive player retention strategies

### Business Integration
1. **Marketing Automation**: Target campaigns based on predictions
2. **Game Design**: Use insights for feature development
3. **Revenue Optimization**: Implement dynamic pricing strategies
4. **Player Support**: Identify players needing assistance

---

🎮 **Happy Gaming Analytics!** 

For questions or contributions, please refer to the project documentation or create an issue. 