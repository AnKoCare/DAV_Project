"""
Interactive Dashboard for Gaming Behavior Prediction Project
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from config.config import *

class GamingBehaviorDashboard:
    """Interactive dashboard for gaming behavior analysis"""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.app.title = "Gaming Behavior Analytics Dashboard"
        self.data = None
        self.setup_layout()
        self.setup_callbacks()
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample gaming behavior data
        data = {
            'PlayerID': range(1, n_samples + 1),
            'Age': np.random.normal(28, 8, n_samples).astype(int).clip(13, 65),
            'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.6, 0.35, 0.05]),
            'Location': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia'], n_samples),
            'Platform': np.random.choice(['PC', 'Mobile', 'Console', 'VR'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
            'GameGenre': np.random.choice(['Action', 'RPG', 'Strategy', 'Sports', 'Simulation'], n_samples),
            'PlayTimeHours': np.random.exponential(20, n_samples).clip(0.1, 200),
            'SessionsPerWeek': np.random.poisson(4, n_samples).clip(1, 20),
            'AvgSessionDurationMinutes': np.random.normal(45, 20, n_samples).clip(5, 300),
            'InGamePurchases': np.random.exponential(5, n_samples).clip(0, 100),
            'GameDifficulty': np.random.randint(1, 6, n_samples),
            'EngagementLevel': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2])
        }
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🎮 Gaming Behavior Analytics Dashboard", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
                html.P("Comprehensive analysis of player engagement and gaming patterns",
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
            
            # Key Metrics Row
            html.Div([
                html.Div([
                    html.H3("Key Metrics", style={'color': '#2c3e50'}),
                    html.Div(id="key-metrics", children=[])
                ], className="twelve columns")
            ], className="row", style={'marginBottom': '30px'}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Select Platform:", style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='platform-dropdown',
                        options=[{'label': 'All Platforms', 'value': 'all'}],
                        value='all',
                        style={'marginBottom': '10px'}
                    ),
                    
                    html.Label("Select Age Range:", style={'fontWeight': 'bold'}),
                    dcc.RangeSlider(
                        id='age-range-slider',
                        min=13, max=65, step=1,
                        marks={i: str(i) for i in range(13, 66, 10)},
                        value=[13, 65]
                    )
                ], className="four columns"),
                
                html.Div([
                    html.Label("Select Engagement Level:", style={'fontWeight': 'bold'}),
                    dcc.Checklist(
                        id='engagement-checklist',
                        options=[
                            {'label': 'Low', 'value': 'Low'},
                            {'label': 'Medium', 'value': 'Medium'},
                            {'label': 'High', 'value': 'High'}
                        ],
                        value=['Low', 'Medium', 'High'],
                        inline=True
                    )
                ], className="eight columns")
            ], className="row", style={'marginBottom': '30px', 'backgroundColor': '#f8f9fa', 'padding': '20px'}),
            
            # Charts Row 1
            html.Div([
                html.Div([
                    dcc.Graph(id="player-demographics-chart")
                ], className="six columns"),
                
                html.Div([
                    dcc.Graph(id="engagement-distribution-chart")
                ], className="six columns")
            ], className="row", style={'marginBottom': '30px'}),
            
            # Charts Row 2
            html.Div([
                html.Div([
                    dcc.Graph(id="platform-analysis-chart")
                ], className="six columns"),
                
                html.Div([
                    dcc.Graph(id="gaming-patterns-chart")
                ], className="six columns")
            ], className="row", style={'marginBottom': '30px'}),
            
            # Charts Row 3
            html.Div([
                html.Div([
                    dcc.Graph(id="correlation-heatmap")
                ], className="six columns"),
                
                html.Div([
                    dcc.Graph(id="player-segments-chart")
                ], className="six columns")
            ], className="row", style={'marginBottom': '30px'}),
            
            # Footer
            html.Div([
                html.P(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                      style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '10px', 'marginTop': '40px'})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('platform-dropdown', 'options'),
             Output('key-metrics', 'children'),
             Output('player-demographics-chart', 'figure'),
             Output('engagement-distribution-chart', 'figure'),
             Output('platform-analysis-chart', 'figure'),
             Output('gaming-patterns-chart', 'figure'),
             Output('correlation-heatmap', 'figure'),
             Output('player-segments-chart', 'figure')],
            [Input('platform-dropdown', 'value'),
             Input('age-range-slider', 'value'),
             Input('engagement-checklist', 'value')]
        )
        def update_dashboard(selected_platform, age_range, engagement_levels):
            if self.data is None:
                self.load_sample_data()
            
            # Filter data based on selections
            filtered_data = self.data.copy()
            
            # Filter by platform
            if selected_platform != 'all':
                filtered_data = filtered_data[filtered_data['Platform'] == selected_platform]
            
            # Filter by age range
            filtered_data = filtered_data[
                (filtered_data['Age'] >= age_range[0]) & 
                (filtered_data['Age'] <= age_range[1])
            ]
            
            # Filter by engagement levels
            filtered_data = filtered_data[filtered_data['EngagementLevel'].isin(engagement_levels)]
            
            # Update platform dropdown options
            platform_options = [{'label': 'All Platforms', 'value': 'all'}]
            platform_options.extend([
                {'label': platform, 'value': platform} 
                for platform in self.data['Platform'].unique()
            ])
            
            # Generate key metrics
            key_metrics = self.generate_key_metrics(filtered_data)
            
            # Generate charts
            demo_chart = self.create_demographics_chart(filtered_data)
            engagement_chart = self.create_engagement_chart(filtered_data)
            platform_chart = self.create_platform_chart(filtered_data)
            patterns_chart = self.create_gaming_patterns_chart(filtered_data)
            correlation_chart = self.create_correlation_chart(filtered_data)
            segments_chart = self.create_segments_chart(filtered_data)
            
            return (platform_options, key_metrics, demo_chart, engagement_chart,
                   platform_chart, patterns_chart, correlation_chart, segments_chart)
    
    def generate_key_metrics(self, data):
        """Generate key metrics cards"""
        total_players = len(data)
        avg_playtime = data['PlayTimeHours'].mean()
        avg_sessions = data['SessionsPerWeek'].mean()
        total_revenue = data['InGamePurchases'].sum()
        
        metrics = html.Div([
            html.Div([
                html.H4(f"{total_players:,}", style={'color': '#3498db', 'margin': '0'}),
                html.P("Total Players", style={'margin': '5px 0'})
            ], className="three columns", style={'textAlign': 'center', 'backgroundColor': 'white', 
                                               'padding': '20px', 'margin': '10px', 'borderRadius': '5px',
                                               'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4(f"{avg_playtime:.1f}h", style={'color': '#e74c3c', 'margin': '0'}),
                html.P("Avg Play Time", style={'margin': '5px 0'})
            ], className="three columns", style={'textAlign': 'center', 'backgroundColor': 'white',
                                               'padding': '20px', 'margin': '10px', 'borderRadius': '5px',
                                               'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4(f"{avg_sessions:.1f}", style={'color': '#2ecc71', 'margin': '0'}),
                html.P("Avg Sessions/Week", style={'margin': '5px 0'})
            ], className="three columns", style={'textAlign': 'center', 'backgroundColor': 'white',
                                               'padding': '20px', 'margin': '10px', 'borderRadius': '5px',
                                               'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4(f"${total_revenue:,.0f}", style={'color': '#f39c12', 'margin': '0'}),
                html.P("Total Revenue", style={'margin': '5px 0'})
            ], className="three columns", style={'textAlign': 'center', 'backgroundColor': 'white',
                                               'padding': '20px', 'margin': '10px', 'borderRadius': '5px',
                                               'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], className="row")
        
        return metrics
    
    def create_demographics_chart(self, data):
        """Create player demographics chart"""
        fig = px.histogram(data, x='Age', nbins=20, title='Player Age Distribution',
                          color_discrete_sequence=['#3498db'])
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        return fig
    
    def create_engagement_chart(self, data):
        """Create engagement level distribution chart"""
        engagement_counts = data['EngagementLevel'].value_counts()
        fig = px.pie(values=engagement_counts.values, names=engagement_counts.index,
                    title='Engagement Level Distribution',
                    color_discrete_sequence=['#e74c3c', '#f39c12', '#2ecc71'])
        return fig
    
    def create_platform_chart(self, data):
        """Create platform analysis chart"""
        platform_engagement = data.groupby(['Platform', 'EngagementLevel']).size().reset_index(name='Count')
        fig = px.bar(platform_engagement, x='Platform', y='Count', color='EngagementLevel',
                    title='Engagement by Platform', barmode='stack',
                    color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#2ecc71'})
        fig.update_layout(plot_bgcolor='white')
        return fig
    
    def create_gaming_patterns_chart(self, data):
        """Create gaming patterns chart"""
        fig = px.scatter(data, x='PlayTimeHours', y='SessionsPerWeek', 
                        color='EngagementLevel', size='InGamePurchases',
                        title='Gaming Patterns: Play Time vs Sessions',
                        color_discrete_map={'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#2ecc71'})
        fig.update_layout(plot_bgcolor='white')
        return fig
    
    def create_correlation_chart(self, data):
        """Create correlation heatmap"""
        numeric_cols = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 
                       'InGamePurchases', 'GameDifficulty']
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title='Feature Correlation Matrix',
                       color_continuous_scale='RdBu_r')
        return fig
    
    def create_segments_chart(self, data):
        """Create player segments chart"""
        # Create simple segments based on play time and spending
        data_copy = data.copy()
        data_copy['Segment'] = 'Casual'
        
        high_playtime = data_copy['PlayTimeHours'] > data_copy['PlayTimeHours'].quantile(0.75)
        high_spending = data_copy['InGamePurchases'] > data_copy['InGamePurchases'].quantile(0.75)
        
        data_copy.loc[high_playtime & ~high_spending, 'Segment'] = 'Hardcore'
        data_copy.loc[~high_playtime & high_spending, 'Segment'] = 'Whale'
        data_copy.loc[high_playtime & high_spending, 'Segment'] = 'VIP'
        
        segment_counts = data_copy['Segment'].value_counts()
        fig = px.bar(x=segment_counts.index, y=segment_counts.values,
                    title='Player Segments',
                    color=segment_counts.index,
                    color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'])
        fig.update_layout(showlegend=False, plot_bgcolor='white')
        return fig
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard"""
        self.app.run_server(host=host, port=port, debug=debug)

if __name__ == '__main__':
    dashboard = GamingBehaviorDashboard()
    dashboard.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG) 