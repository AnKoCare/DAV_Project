"""
Visualization module for Gaming Behavior Prediction Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not available. Some interactive plots will be skipped.")

try:
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: Some sklearn metrics not available.")

class GameBehaviorVisualizer:
    """Class to handle all visualizations for gaming behavior analysis"""
    
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        self.style = style
        self.figsize = figsize
        # Handle different matplotlib style names
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
                print(f"Warning: Style '{style}' not available, using default.")
        sns.set_palette("husl")
    
    def plot_player_demographics(self, df: pd.DataFrame, save_path: str = None):
        """Plot player demographics distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Demographics Analysis', fontsize=16, fontweight='bold')
        
        # Age distribution
        if 'Age' in df.columns:
            axes[0, 0].hist(df['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Age Distribution')
            axes[0, 0].set_xlabel('Age')
            axes[0, 0].set_ylabel('Frequency')
        
        # Gender distribution
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Gender Distribution')
        
        # Location distribution
        if 'Location' in df.columns:
            location_counts = df['Location'].value_counts().head(10)
            axes[1, 0].bar(location_counts.index, location_counts.values, color='lightcoral')
            axes[1, 0].set_title('Top 10 Locations')
            axes[1, 0].set_xlabel('Location')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Platform distribution
        if 'Platform' in df.columns:
            platform_counts = df['Platform'].value_counts()
            axes[1, 1].bar(platform_counts.index, platform_counts.values, color='lightgreen')
            axes[1, 1].set_title('Platform Distribution')
            axes[1, 1].set_xlabel('Platform')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_gaming_patterns(self, df: pd.DataFrame, save_path: str = None):
        """Plot gaming behavior patterns"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Gaming Behavior Patterns', fontsize=16, fontweight='bold')
        
        # Play time distribution
        if 'PlayTimeHours' in df.columns:
            axes[0, 0].hist(df['PlayTimeHours'], bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[0, 0].set_title('Play Time Distribution (Hours)')
            axes[0, 0].set_xlabel('Hours')
            axes[0, 0].set_ylabel('Frequency')
        
        # Sessions per week
        if 'SessionsPerWeek' in df.columns:
            axes[0, 1].hist(df['SessionsPerWeek'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 1].set_title('Sessions Per Week Distribution')
            axes[0, 1].set_xlabel('Sessions')
            axes[0, 1].set_ylabel('Frequency')
        
        # In-game purchases
        if 'InGamePurchases' in df.columns:
            axes[0, 2].hist(df['InGamePurchases'], bins=25, alpha=0.7, color='green', edgecolor='black')
            axes[0, 2].set_title('In-Game Purchases Distribution')
            axes[0, 2].set_xlabel('Purchases')
            axes[0, 2].set_ylabel('Frequency')
        
        # Average session duration
        if 'AvgSessionDurationMinutes' in df.columns:
            axes[1, 0].hist(df['AvgSessionDurationMinutes'], bins=25, alpha=0.7, color='red', edgecolor='black')
            axes[1, 0].set_title('Average Session Duration (Minutes)')
            axes[1, 0].set_xlabel('Minutes')
            axes[1, 0].set_ylabel('Frequency')
        
        # Game difficulty
        if 'GameDifficulty' in df.columns:
            diff_counts = df['GameDifficulty'].value_counts().sort_index()
            axes[1, 1].bar(diff_counts.index, diff_counts.values, color='brown')
            axes[1, 1].set_title('Game Difficulty Distribution')
            axes[1, 1].set_xlabel('Difficulty Level')
            axes[1, 1].set_ylabel('Count')
        
        # Game genre
        if 'GameGenre' in df.columns:
            genre_counts = df['GameGenre'].value_counts()
            axes[1, 2].bar(genre_counts.index, genre_counts.values, color='pink')
            axes[1, 2].set_title('Game Genre Distribution')
            axes[1, 2].set_xlabel('Genre')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_engagement_analysis(self, df: pd.DataFrame, target_col: str, save_path: str = None):
        """Plot engagement level analysis"""
        if target_col not in df.columns:
            print(f"Target column {target_col} not found in dataframe")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Engagement Analysis', fontsize=16, fontweight='bold')
        
        # Engagement distribution
        engagement_counts = df[target_col].value_counts()
        axes[0, 0].pie(engagement_counts.values, labels=engagement_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Engagement Level Distribution')
        
        # Engagement by Age Group
        if 'Age' in df.columns:
            age_bins = pd.cut(df['Age'], bins=[0, 18, 25, 35, 45, 100], labels=['<18', '18-25', '25-35', '35-45', '45+'])
            engagement_by_age = pd.crosstab(age_bins, df[target_col], normalize='index') * 100
            engagement_by_age.plot(kind='bar', ax=axes[0, 1], stacked=True)
            axes[0, 1].set_title('Engagement by Age Group (%)')
            axes[0, 1].set_xlabel('Age Group')
            axes[0, 1].set_ylabel('Percentage')
            axes[0, 1].legend(title='Engagement Level')
        
        # Engagement by Platform
        if 'Platform' in df.columns:
            engagement_by_platform = pd.crosstab(df['Platform'], df[target_col], normalize='index') * 100
            engagement_by_platform.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Engagement by Platform (%)')
            axes[1, 0].set_xlabel('Platform')
            axes[1, 0].set_ylabel('Percentage')
            axes[1, 0].legend(title='Engagement Level')
        
        # Engagement by Game Genre
        if 'GameGenre' in df.columns:
            engagement_by_genre = pd.crosstab(df['GameGenre'], df[target_col], normalize='index') * 100
            engagement_by_genre.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Engagement by Game Genre (%)')
            axes[1, 1].set_xlabel('Game Genre')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].legend(title='Engagement Level')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str = None):
        """Plot correlation matrix of numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numerical_cols].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15, save_path: str = None):
        """Plot feature importance"""
        if importance_df is None or importance_df.empty:
            print("No feature importance data provided")
            return
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance(self, model_scores: dict, task_type: str = 'classification', save_path: str = None):
        """Plot model performance comparison"""
        if not model_scores:
            print("No model scores provided")
            return
        
        # Prepare data for plotting
        models = list(model_scores.keys())
        
        if task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        else:
            metrics = ['mse', 'rmse', 'mae', 'r2']
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                scores = [model_scores[model].get(metric, 0) for model in models]
                bars = axes[i].bar(models, scores, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].annotate(f'{height:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_player_segments(self, df: pd.DataFrame, save_path: str = None):
        """Plot player segmentation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # Create player segments based on engagement metrics
        if all(col in df.columns for col in ['PlayTimeHours', 'SessionsPerWeek', 'InGamePurchases']):
            # High/Low engagement based on play time
            df_copy = df.copy()
            df_copy['PlayTimeSegment'] = pd.cut(df_copy['PlayTimeHours'], 
                                         bins=[0, df_copy['PlayTimeHours'].quantile(0.33), 
                                               df_copy['PlayTimeHours'].quantile(0.67), float('inf')],
                                         labels=['Low', 'Medium', 'High'])
            
            # Spending behavior
            df_copy['SpendingSegment'] = 'Non-Spender'
            df_copy.loc[df_copy['InGamePurchases'] > 0, 'SpendingSegment'] = 'Light Spender'
            df_copy.loc[df_copy['InGamePurchases'] > df_copy['InGamePurchases'].quantile(0.75), 'SpendingSegment'] = 'Heavy Spender'
            
            # Plot segments
            playtime_counts = df_copy['PlayTimeSegment'].value_counts()
            axes[0, 0].pie(playtime_counts.values, labels=playtime_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Play Time Segments')
            
            spending_counts = df_copy['SpendingSegment'].value_counts()
            axes[0, 1].pie(spending_counts.values, labels=spending_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Spending Behavior Segments')
            
            # Cross-segment analysis
            segment_cross = pd.crosstab(df_copy['PlayTimeSegment'], df_copy['SpendingSegment'])
            segment_cross.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Play Time vs Spending Segments')
            axes[1, 0].set_xlabel('Play Time Segment')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend(title='Spending Segment')
            
            # Session frequency analysis
            if 'SessionsPerWeek' in df.columns:
                try:
                    boxplot_data = [df_copy[df_copy['PlayTimeSegment'] == segment]['SessionsPerWeek'].dropna() 
                                  for segment in ['Low', 'Medium', 'High']]
                    axes[1, 1].boxplot(boxplot_data)
                    axes[1, 1].set_xticklabels(['Low', 'Medium', 'High'])
                    axes[1, 1].set_title('Sessions per Week by Play Time Segment')
                    axes[1, 1].set_xlabel('Play Time Segment')
                    axes[1, 1].set_ylabel('Sessions per Week')
                except Exception as e:
                    print(f"Warning: Could not create boxplot - {e}")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 