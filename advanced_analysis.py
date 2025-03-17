"""
Advanced Analysis Module for Bitcoin Correlation Analysis.

This module performs more sophisticated analyses to understand the relationship
between Bitcoin and other assets.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import ccf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

def load_processed_data():
    """
    Load processed data from CSV files.
    
    Returns:
        tuple: (prices_df, returns_df) - DataFrames with aligned price and return data
    """
    prices = pd.read_csv("data/aligned_prices.csv", index_col=0, parse_dates=True)
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    
    logger.info(f"Loaded processed data: {len(prices)} rows of prices, {len(returns)} rows of returns")
    
    return prices, returns

def analyze_time_lagged_correlations(returns, max_lag=10):
    """
    Analyze time-lagged correlations to identify lead-lag relationships.
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
        max_lag (int): Maximum number of lags to analyze
        
    Returns:
        dict: Dictionary containing lagged correlation results
    """
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return') and not col.endswith('log_return')]
    ret_data = returns[return_columns]
    
    # Dictionary to store results
    lagged_corrs = {}
    
    # Calculate cross-correlation function (CCF) for Bitcoin with other assets
    bitcoin_col = 'bitcoin_return'
    
    for col in return_columns:
        if col != bitcoin_col:
            asset_name = col.split('_')[0]
            
            # Calculate CCF
            ccf_values = ccf(ret_data[bitcoin_col], ret_data[col], adjusted=False)
            
            # Create DataFrame with lag and correlation
            df = pd.DataFrame({
                'lag': range(-max_lag, max_lag + 1),
                'correlation': ccf_values[(len(ccf_values)//2 - max_lag):(len(ccf_values)//2 + max_lag + 1)]
            })
            
            lagged_corrs[asset_name] = df
            
            # Save to CSV
            df.to_csv(f"data/lagged_correlation_btc_{asset_name}.csv", index=False)
    
    logger.info(f"Calculated and saved time-lagged correlations with maximum lag of {max_lag}")
    
    return lagged_corrs

def plot_time_lagged_correlations(lagged_corrs):
    """
    Plot time-lagged correlations.
    
    Args:
        lagged_corrs (dict): Dictionary containing lagged correlation results
    """
    # Plot for each asset
    for asset, df in lagged_corrs.items():
        plt.figure(figsize=(12, 6))
        
        # Plot correlation at each lag
        plt.bar(df['lag'], df['correlation'], color='steelblue')
        
        # Add reference line at zero
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        
        # Add vertical line at lag 0 (contemporaneous correlation)
        plt.axvline(x=0, color='green', linestyle='-', alpha=0.3)
        
        # Highlight maximum correlation
        max_corr_idx = df['correlation'].abs().idxmax()
        max_lag = df.loc[max_corr_idx, 'lag']
        max_corr = df.loc[max_corr_idx, 'correlation']
        plt.scatter(max_lag, max_corr, color='red', s=100, zorder=5)
        plt.annotate(
            f'Max: {max_corr:.3f} at lag {max_lag}',
            xy=(max_lag, max_corr),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12
        )
        
        plt.title(f'Time-Lagged Correlation: Bitcoin - {asset.capitalize()}', fontsize=16)
        plt.xlabel('Lag (days)', fontsize=14)
        plt.ylabel('Correlation Coefficient', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'plots/lagged_correlation_btc_{asset}.png', dpi=300)
        plt.close()
        
        logger.info(f"Created time-lagged correlation plot for Bitcoin-{asset}")

def analyze_market_regimes(returns, window=90):
    """
    Analyze correlations during different market regimes (bull/bear markets).
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
        window (int): Window size to determine market regime
        
    Returns:
        pandas.DataFrame: DataFrame with market regime analysis
    """
    # Focus on daily percentage returns and prices
    return_columns = [col for col in returns.columns if col.endswith('_return') and not col.endswith('log_return')]
    log_return_columns = [col for col in returns.columns if col.endswith('log_return')]
    
    # Create a copy to work with
    data = returns.copy()
    
    # Calculate cumulative returns for Bitcoin over the window to identify regimes
    bitcoin_log_return = 'bitcoin_log_return'
    data['btc_cum_return'] = data[bitcoin_log_return].rolling(window=window).sum()
    
    # Define bull/bear market (positive/negative cumulative return)
    data['market_regime'] = np.where(data['btc_cum_return'] > 0, 'Bull', 'Bear')
    
    # Drop rows with NaN values (due to rolling window)
    data = data.dropna()
    
    # Create empty DataFrame for results
    results = pd.DataFrame(columns=[
        'Asset',
        'Regime',
        'Pearson Correlation',
        'Days'
    ])
    
    # Calculate correlations in each regime
    bitcoin_return = 'bitcoin_return'
    row_idx = 0
    
    for regime in ['Bull', 'Bear']:
        regime_data = data[data['market_regime'] == regime]
        days = len(regime_data)
        
        for col in return_columns:
            if col != bitcoin_return:
                asset = col.split('_')[0].capitalize()
                
                # Calculate correlation in this regime
                corr = regime_data[bitcoin_return].corr(regime_data[col])
                
                # Add to results
                results.loc[row_idx] = [asset, regime, corr, days]
                row_idx += 1
    
    # Add 'All' market regime
    for col in return_columns:
        if col != bitcoin_return:
            asset = col.split('_')[0].capitalize()
            
            # Calculate correlation across all data
            corr = data[bitcoin_return].corr(data[col])
            
            # Add to results
            results.loc[row_idx] = [asset, 'All', corr, len(data)]
            row_idx += 1
    
    # Save results to CSV
    results.to_csv("data/market_regime_correlations.csv", index=False)
    logger.info(f"Analyzed correlations across market regimes using {window}-day window")
    
    return results

def plot_market_regime_correlations(results):
    """
    Plot correlation differences across market regimes.
    
    Args:
        results (pandas.DataFrame): DataFrame with market regime analysis
    """
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar chart
    sns.barplot(
        data=results, 
        x='Asset', 
        y='Pearson Correlation', 
        hue='Regime',
        palette=['green', 'red', 'blue']
    )
    
    plt.title('Bitcoin Correlations Across Market Regimes', fontsize=16)
    plt.xlabel('Asset', fontsize=14)
    plt.ylabel('Correlation Coefficient', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/market_regime_correlations.png', dpi=300)
    plt.close()
    
    logger.info("Created market regime correlation plot")

def regression_analysis(returns):
    """
    Perform regression analysis to quantify relationships.
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
        
    Returns:
        pandas.DataFrame: DataFrame with regression results
    """
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return') and not col.endswith('log_return')]
    ret_data = returns[return_columns].copy()
    
    # Create empty DataFrame for results
    results = pd.DataFrame(columns=[
        'Asset',
        'R-squared',
        'Coefficient',
        'MSE'
    ])
    
    # Perform regression for Bitcoin with each asset
    bitcoin_col = 'bitcoin_return'
    row_idx = 0
    
    for col in return_columns:
        if col != bitcoin_col:
            asset = col.split('_')[0].capitalize()
            
            # Prepare data
            X = ret_data[[col]]
            y = ret_data[bitcoin_col]
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r_squared = r2_score(y_test, y_pred)
            coefficient = model.coef_[0]
            mse = mean_squared_error(y_test, y_pred)
            
            # Add to results
            results.loc[row_idx] = [asset, r_squared, coefficient, mse]
            row_idx += 1
            
            # Create scatter plot with regression line
            plt.figure(figsize=(10, 6))
            sns.regplot(x=ret_data[col], y=ret_data[bitcoin_col], scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            plt.title(f'Regression: Bitcoin vs. {asset}', fontsize=16)
            plt.xlabel(f'{asset} Return (%)', fontsize=14)
            plt.ylabel('Bitcoin Return (%)', fontsize=14)
            plt.annotate(
                f'R²: {r_squared:.4f}\nCoef: {coefficient:.4f}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
            )
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f'plots/regression_btc_{asset.lower()}.png', dpi=300)
            plt.close()
    
    # Save results to CSV
    results.to_csv("data/regression_results.csv", index=False)
    logger.info("Completed regression analysis")
    
    return results

def run_advanced_analysis():
    """
    Run the complete advanced analysis.
    
    Returns:
        tuple: (lagged_corrs, regime_results, regression_results) - The calculated advanced analysis results
    """
    # Load processed data
    _, returns = load_processed_data()
    
    # Time-lagged correlation analysis
    lagged_corrs = analyze_time_lagged_correlations(returns, max_lag=10)
    plot_time_lagged_correlations(lagged_corrs)
    
    # Market regime analysis
    regime_results = analyze_market_regimes(returns, window=90)
    plot_market_regime_correlations(regime_results)
    
    # Regression analysis
    regression_results = regression_analysis(returns)
    
    logger.info("Completed advanced analysis")
    
    return lagged_corrs, regime_results, regression_results

if __name__ == "__main__":
    run_advanced_analysis() 