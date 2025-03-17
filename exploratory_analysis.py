"""
Exploratory Data Analysis Module for Bitcoin Correlation Analysis.

This module performs initial data exploration and generates summary statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

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

def generate_summary_statistics(prices, returns):
    """
    Generate summary statistics for price and return data.
    
    Args:
        prices (pandas.DataFrame): DataFrame with price data
        returns (pandas.DataFrame): DataFrame with return data
        
    Returns:
        tuple: (price_stats, return_stats) - DataFrames with summary statistics
    """
    # Summary statistics for prices
    price_stats = prices.describe()
    
    # Summary statistics for returns (focus on daily percentage returns)
    return_columns = [col for col in returns.columns if col.endswith('_return')]
    return_stats = returns[return_columns].describe()
    
    # Additional statistics for returns
    for col in return_columns:
        return_stats.loc['skew', col] = returns[col].skew()
        return_stats.loc['kurtosis', col] = returns[col].kurtosis()
        
    # Save statistics to CSV
    price_stats.to_csv("data/price_statistics.csv")
    return_stats.to_csv("data/return_statistics.csv")
    
    logger.info("Generated and saved summary statistics")
    
    return price_stats, return_stats

def plot_price_series(prices):
    """
    Plot price series for all assets.
    
    Args:
        prices (pandas.DataFrame): DataFrame with price data
    """
    # Normalize prices to start at 100 for comparison
    normalized = prices.copy()
    for col in normalized.columns:
        normalized[col] = 100 * (normalized[col] / normalized[col].iloc[0])
    
    plt.figure(figsize=(12, 8))
    
    # Plot each asset
    for col in normalized.columns:
        plt.plot(normalized.index, normalized[col], label=col)
    
    plt.title('Normalized Price Comparison (Base = 100)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Normalized Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/normalized_prices.png', dpi=300)
    plt.close()
    
    logger.info("Created normalized price plot")
    
    # Also plot original prices
    fig, axes = plt.subplots(len(prices.columns), 1, figsize=(12, 4*len(prices.columns)), sharex=True)
    
    for i, col in enumerate(prices.columns):
        axes[i].plot(prices.index, prices[col], label=col)
        axes[i].set_title(f'{col} Price', fontsize=14)
        axes[i].set_ylabel('Price (USD)', fontsize=12)
        axes[i].legend()
    
    plt.xlabel('Date', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/individual_prices.png', dpi=300)
    plt.close()
    
    logger.info("Created individual price plots")

def plot_return_distributions(returns):
    """
    Plot return distributions for all assets.
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
    """
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return')]
    ret_data = returns[return_columns]
    
    # Plot histograms
    fig, axes = plt.subplots(len(return_columns), 1, figsize=(12, 4*len(return_columns)))
    
    for i, col in enumerate(return_columns):
        sns.histplot(ret_data[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution', fontsize=14)
        axes[i].set_xlabel('Daily Return (%)', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        
        # Add vertical line at mean
        mean = ret_data[col].mean()
        axes[i].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.4f}')
        axes[i].legend()
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/return_distributions.png', dpi=300)
    plt.close()
    
    logger.info("Created return distribution plots")
    
    # Plot boxplots for returns
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=ret_data)
    plt.title('Return Comparison (Box Plots)', fontsize=16)
    plt.ylabel('Daily Return (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/return_boxplots.png', dpi=300)
    plt.close()
    
    logger.info("Created return boxplot")

def plot_time_series_of_returns(returns):
    """
    Plot time series of returns for all assets.
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
    """
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return')]
    ret_data = returns[return_columns]
    
    fig, axes = plt.subplots(len(return_columns), 1, figsize=(12, 4*len(return_columns)), sharex=True)
    
    for i, col in enumerate(return_columns):
        axes[i].plot(ret_data.index, ret_data[col], label=col)
        axes[i].set_title(f'{col} Time Series', fontsize=14)
        axes[i].set_ylabel('Daily Return (%)', fontsize=12)
        axes[i].legend()
    
    plt.xlabel('Date', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/return_time_series.png', dpi=300)
    plt.close()
    
    logger.info("Created return time series plots")

def run_exploratory_analysis():
    """
    Run the complete exploratory analysis.
    
    Returns:
        tuple: (prices, returns) - The loaded price and return data
    """
    # Load processed data
    prices, returns = load_processed_data()
    
    # Generate summary statistics
    generate_summary_statistics(prices, returns)
    
    # Generate plots
    plot_price_series(prices)
    plot_return_distributions(returns)
    plot_time_series_of_returns(returns)
    
    logger.info("Completed exploratory data analysis")
    
    return prices, returns

if __name__ == "__main__":
    run_exploratory_analysis() 