"""
Visualization Module for Bitcoin Correlation Analysis.

This module generates comprehensive visualizations for the analysis results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('plots/interactive', exist_ok=True)

def load_all_data():
    """
    Load all analysis data.
    
    Returns:
        dict: Dictionary containing all data for visualization
    """
    data = {}
    
    # Load basic processed data
    data['prices'] = pd.read_csv("data/aligned_prices.csv", index_col=0, parse_dates=True)
    data['returns'] = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    
    # Load correlation data
    data['pearson_corr'] = pd.read_csv("data/pearson_correlation.csv", index_col=0)
    data['spearman_corr'] = pd.read_csv("data/spearman_correlation.csv", index_col=0)
    
    # Load rolling correlations
    data['rolling_corr'] = pd.read_csv("data/rolling_30d_correlations.csv", index_col=0, parse_dates=True)
    
    # Load market regime data
    if os.path.exists("data/market_regime_correlations.csv"):
        data['market_regimes'] = pd.read_csv("data/market_regime_correlations.csv")
    
    # Load regression results
    if os.path.exists("data/regression_results.csv"):
        data['regression'] = pd.read_csv("data/regression_results.csv")
    
    # Load correlation significance data
    if os.path.exists("data/correlation_significance.csv"):
        data['significance'] = pd.read_csv("data/correlation_significance.csv")
    
    logger.info("Loaded all analysis data for visualization")
    
    return data

def create_interactive_price_chart(prices):
    """
    Create an interactive price chart with Plotly.
    
    Args:
        prices (pandas.DataFrame): DataFrame with price data
    """
    # Convert index to datetime if not already
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index)
    
    # Create normalized prices for comparison
    normalized = prices.copy()
    for col in normalized.columns:
        normalized[col] = 100 * (normalized[col] / normalized[col].iloc[0])
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each asset
    colors = {'bitcoin': 'orange', 'gold': 'gold', 'nasdaq': 'blue'}
    
    for col in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index, 
                y=normalized[col], 
                name=f"{col.capitalize()} (Normalized)",
                line=dict(color=colors.get(col, 'gray'))
            ),
            secondary_y=False
        )
    
    # Add actual Bitcoin price on secondary axis
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices['bitcoin'] if 'bitcoin' in prices.columns else prices.iloc[:, 0],
            name="Bitcoin Price (USD)",
            line=dict(color='red', width=1, dash='dot')
        ),
        secondary_y=True
    )
    
    # Customize layout
    fig.update_layout(
        title="Asset Price Comparison (Normalized to 100)",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Normalized Price (Base=100)", secondary_y=False)
    fig.update_yaxes(title_text="Bitcoin Price (USD)", secondary_y=True)
    
    # Save to HTML file
    fig.write_html("plots/interactive/price_comparison.html")
    logger.info("Created interactive price comparison chart")

def create_interactive_correlation_heatmap(data):
    """
    Create an interactive correlation heatmap with Plotly.
    
    Args:
        data (dict): Dictionary containing all data for visualization
    """
    # Pearson correlation matrix
    pearson_corr = data['pearson_corr']
    
    # Create a mask for the upper triangle
    mask = np.zeros_like(pearson_corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Create heatmap
    fig = px.imshow(
        pearson_corr,
        text_auto=".2f",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix"
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Asset Returns",
        yaxis_title="Asset Returns",
        template="plotly_white"
    )
    
    # Save to HTML file
    fig.write_html("plots/interactive/correlation_heatmap.html")
    logger.info("Created interactive correlation heatmap")

def create_rolling_correlation_chart(rolling_corr):
    """
    Create an interactive rolling correlation chart with Plotly.
    
    Args:
        rolling_corr (pandas.DataFrame): DataFrame with rolling correlation data
    """
    # Convert index to datetime if not already
    if not isinstance(rolling_corr.index, pd.DatetimeIndex):
        rolling_corr.index = pd.to_datetime(rolling_corr.index)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each correlation
    colors = {'gold': 'gold', 'nasdaq': 'blue'}
    
    for col in rolling_corr.columns:
        asset = col.split('_')[1]  # Extract asset name from column name
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index, 
                y=rolling_corr[col], 
                name=f"BTC-{asset.capitalize()} Correlation",
                line=dict(color=colors.get(asset, 'gray'), width=2)
            )
        )
    
    # Add reference line at zero
    fig.add_trace(
        go.Scatter(
            x=[rolling_corr.index[0], rolling_corr.index[-1]],
            y=[0, 0],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            name='Zero Correlation'
        )
    )
    
    # Customize layout
    fig.update_layout(
        title="30-Day Rolling Correlation with Bitcoin",
        xaxis_title="Date",
        yaxis_title="Correlation Coefficient",
        yaxis=dict(range=[-1, 1]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Save to HTML file
    fig.write_html("plots/interactive/rolling_correlation.html")
    logger.info("Created interactive rolling correlation chart")

def create_market_regime_chart(market_regimes):
    """
    Create an interactive chart for market regime analysis with Plotly.
    
    Args:
        market_regimes (pandas.DataFrame): DataFrame with market regime analysis
    """
    # Create figure
    fig = px.bar(
        market_regimes, 
        x='Asset', 
        y='Pearson Correlation', 
        color='Regime',
        barmode='group',
        color_discrete_map={'Bull': 'green', 'Bear': 'red', 'All': 'blue'},
        title="Bitcoin Correlations Across Market Regimes"
    )
    
    # Add reference line at zero
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(market_regimes['Asset'].unique()) - 0.5,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Asset",
        yaxis_title="Correlation Coefficient",
        template="plotly_white"
    )
    
    # Save to HTML file
    fig.write_html("plots/interactive/market_regime_correlations.html")
    logger.info("Created interactive market regime chart")

def create_summary_dashboard():
    """
    Create an HTML dashboard summarizing key findings.
    """
    # Load data
    data = load_all_data()
    
    # Extract key correlations
    pearson_corr = data['pearson_corr']
    btc_gold_corr = pearson_corr.loc['bitcoin_return', 'gold_return'] if 'gold_return' in pearson_corr.columns else "N/A"
    btc_nasdaq_corr = pearson_corr.loc['bitcoin_return', 'nasdaq_return'] if 'nasdaq_return' in pearson_corr.columns else "N/A"
    
    # Format correlations for display
    btc_gold_corr_str = f"{btc_gold_corr:.4f}" if isinstance(btc_gold_corr, (int, float)) else btc_gold_corr
    btc_nasdaq_corr_str = f"{btc_nasdaq_corr:.4f}" if isinstance(btc_nasdaq_corr, (int, float)) else btc_nasdaq_corr
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin Correlation Analysis Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }}
            h2 {{
                color: #3498db;
                margin-top: 30px;
            }}
            .summary-box {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
            }}
            .correlation-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .correlation-table th, .correlation-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .correlation-table th {{
                background-color: #f2f2f2;
            }}
            .highlight {{
                font-weight: bold;
                color: #e74c3c;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
            .chart-container {{
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bitcoin Correlation Analysis Dashboard</h1>
            
            <div class="summary-box">
                <h2>Key Findings</h2>
                <p><strong>Bitcoin-Gold Correlation:</strong> <span class="highlight">{btc_gold_corr_str}</span></p>
                <p><strong>Bitcoin-NASDAQ Correlation:</strong> <span class="highlight">{btc_nasdaq_corr_str}</span></p>
                <p><strong>Conclusion:</strong> Bitcoin is {"more" if abs(btc_nasdaq_corr) > abs(btc_gold_corr) else "less"} correlated with NASDAQ than with Gold.</p>
            </div>
            
            <h2>Interactive Visualizations</h2>
            <p>Click on the links below to open interactive charts:</p>
            <ul>
                <li><a href="interactive/price_comparison.html" target="_blank">Price Comparison Chart</a></li>
                <li><a href="interactive/correlation_heatmap.html" target="_blank">Correlation Heatmap</a></li>
                <li><a href="interactive/rolling_correlation.html" target="_blank">Rolling Correlation Chart</a></li>
                <li><a href="interactive/market_regime_correlations.html" target="_blank">Market Regime Analysis</a></li>
            </ul>
            
            <h2>Static Visualizations</h2>
            <p>The following static visualizations have been generated in the plots directory:</p>
            <ul>
                <li>Normalized price comparison</li>
                <li>Return distributions</li>
                <li>Correlation heatmaps</li>
                <li>Rolling correlations</li>
                <li>Time-lagged correlations</li>
                <li>Market regime analysis</li>
                <li>Regression analysis</li>
            </ul>
            
            <div class="footer">
                <p>Bitcoin Correlation Analysis - Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML content to file
    with open("plots/dashboard.html", "w") as f:
        f.write(html_content)
    
    logger.info("Created summary dashboard")

def generate_visualizations():
    """
    Generate all visualizations.
    """
    # Load all data
    data = load_all_data()
    
    # Create interactive visualizations
    create_interactive_price_chart(data['prices'])
    create_interactive_correlation_heatmap(data)
    create_rolling_correlation_chart(data['rolling_corr'])
    
    # Create market regime chart if data exists
    if 'market_regimes' in data:
        create_market_regime_chart(data['market_regimes'])
    
    # Create summary dashboard
    create_summary_dashboard()
    
    logger.info("Generated all visualizations")

if __name__ == "__main__":
    generate_visualizations() 