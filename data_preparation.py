"""
Data Preparation Module for Bitcoin Correlation Analysis.

This module aligns timestamps, handles missing values, and prepares the data for analysis.
Provides options to align data based on gold's trading days or keep all Bitcoin's trading days.
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load data from CSV files.
    
    Returns:
        dict: Dictionary containing DataFrames for each asset
    """
    data = {}
    for asset in ['bitcoin', 'gold', 'nasdaq']:
        file_path = f"data/{asset}_prices.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} rows for {asset}")
            data[asset] = df
        else:
            logger.warning(f"Data file for {asset} not found: {file_path}")
    
    return data

def clean_data(data):
    """
    Clean and prepare the data.
    
    Args:
        data (dict): Dictionary containing DataFrames for each asset
        
    Returns:
        dict: Dictionary containing cleaned DataFrames
    """
    clean_data = {}
    for asset, df in data.items():
        # Make a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Handle missing values
        missing_count = clean_df['Close'].isna().sum()
        if missing_count > 0:
            logger.info(f"Filling {missing_count} missing values in {asset}")
            clean_df['Close'] = clean_df['Close'].fillna(method='ffill')
        
        # Rename columns to include asset name for clarity when merged
        # Use a more descriptive naming convention: asset_column
        clean_df.columns = [f"{asset}_{col}" for col in clean_df.columns]
        
        # Keep the DataFrame with all columns (not just Close)
        clean_data[asset] = clean_df
        
        logger.info(f"Cleaned data for {asset}, column names: {clean_df.columns.tolist()}")
    
    return clean_data

def align_timestamps_to_gold(data):
    """
    Align all datasets based on gold's trading days.
    Bitcoin trades 24/7 while traditional assets like gold have fewer trading days.
    
    Args:
        data (dict): Dictionary containing DataFrames
        
    Returns:
        pandas.DataFrame: Single DataFrame with aligned data
    """
    if 'gold' not in data:
        logger.warning("Gold data not found. Cannot align based on gold trading days.")
        # If gold data is missing, fall back to combining all data
        combined = pd.concat([df for df in data.values()], axis=1)
        combined = combined.dropna()
        logger.info(f"Combined data has {len(combined)} rows after alignment")
        return combined
    
    # Get gold's trading days as the reference
    gold_trading_days = data['gold'].index
    logger.info(f"Using {len(gold_trading_days)} gold trading days as reference for alignment")
    
    # Create a new dictionary with data aligned to gold's trading days
    aligned_data = {}
    for asset, df in data.items():
        # If this is gold, keep as is
        if asset == 'gold':
            aligned_data[asset] = df
            continue
            
        # For other assets, align to gold's trading days
        # First, ensure the asset data is sorted by date
        df = df.sort_index()
        
        # For Bitcoin (which trades 24/7), select only the dates that match gold's trading days
        if asset == 'bitcoin':
            # Find the closest previous trading day for each gold trading day
            aligned_df = pd.DataFrame(index=gold_trading_days)
            
            # Use reindex with method='ffill' to get the most recent price for each gold trading day
            aligned_df = df.reindex(
                aligned_df.index.union(df.index)
            ).sort_index().ffill().loc[gold_trading_days]
            
            aligned_data[asset] = aligned_df
            logger.info(f"Aligned {asset} data to gold trading days: {len(aligned_df)} rows")
        else:
            # For other traditional assets, might have slightly different trading days
            # Find common trading days or use forward fill
            aligned_df = pd.DataFrame(index=gold_trading_days)
            
            # Use reindex with method='ffill' to handle missing days
            aligned_df = df.reindex(
                aligned_df.index.union(df.index)
            ).sort_index().ffill().loc[gold_trading_days]
            
            aligned_data[asset] = aligned_df
            logger.info(f"Aligned {asset} data to gold trading days: {len(aligned_df)} rows")
    
    # Combine all aligned DataFrames
    combined = pd.concat([df for df in aligned_data.values()], axis=1)
    
    # Check for any remaining NaN values
    if combined.isna().any().any():
        logger.warning(f"Combined data still has {combined.isna().sum().sum()} NaN values")
        # Drop rows with any NaN values
        combined = combined.dropna()
        
    logger.info(f"Final combined data has {len(combined)} rows after alignment to gold trading days")
    logger.info(f"Combined data columns: {combined.columns.tolist()}")
    
    return combined

def align_timestamps_to_bitcoin(data):
    """
    Align all datasets based on Bitcoin's full trading timeline (including weekends/holidays).
    This preserves all Bitcoin data points and shows NaN for traditional assets on non-trading days.
    
    Args:
        data (dict): Dictionary containing DataFrames
        
    Returns:
        pandas.DataFrame: Single DataFrame with aligned data, NaN values preserved
    """
    if 'bitcoin' not in data:
        logger.warning("Bitcoin data not found. Cannot align based on Bitcoin's timeline.")
        # If Bitcoin data is missing, fall back to combining all data
        combined = pd.concat([df for df in data.values()], axis=1)
        logger.info(f"Combined data has {len(combined)} rows")
        return combined
    
    # Get Bitcoin's full trading timeline as the reference
    bitcoin_timeline = data['bitcoin'].index
    logger.info(f"Using {len(bitcoin_timeline)} Bitcoin trading days as reference for alignment")
    
    # Create a new dictionary with data aligned to Bitcoin's timeline
    aligned_data = {}
    for asset, df in data.items():
        # If this is Bitcoin, keep as is
        if asset == 'bitcoin':
            aligned_data[asset] = df
            continue
            
        # For traditional assets, align to Bitcoin's timeline
        # First, ensure the asset data is sorted by date
        df = df.sort_index()
        
        # Create a new DataFrame with Bitcoin's timeline
        aligned_df = pd.DataFrame(index=bitcoin_timeline)
        
        # Reindex the traditional asset data to Bitcoin's timeline
        # Note: We're NOT using ffill here, preserving NaNs for non-trading days
        aligned_df = df.reindex(aligned_df.index)
        
        # Store in our dictionary
        aligned_data[asset] = aligned_df
        
        # Count how many NaN values we have (indicating non-trading days)
        nan_count = aligned_df.isna().any(axis=1).sum()
        logger.info(f"Aligned {asset} data to Bitcoin timeline: {len(aligned_df)} rows with {nan_count} NaN days")
    
    # Combine all aligned DataFrames
    combined = pd.concat([df for df in aligned_data.values()], axis=1)
    
    # Log the shape of the final data
    total_rows = len(combined)
    complete_rows = len(combined.dropna())
    nan_rows = total_rows - complete_rows
    
    logger.info(f"Final combined data has {total_rows} rows aligned to Bitcoin timeline")
    logger.info(f"Complete rows (all assets have data): {complete_rows}")
    logger.info(f"Rows with NaN values (likely weekends/holidays): {nan_rows}")
    logger.info(f"Combined data columns: {combined.columns.tolist()}")
    
    return combined

def align_timestamps(data, align_to='gold', handle_nans='drop'):
    """
    Align all datasets based on the specified reference asset.
    
    Args:
        data (dict): Dictionary containing DataFrames
        align_to (str): Asset to use as alignment reference ('gold' or 'bitcoin')
        handle_nans (str): How to handle NaN values ('drop' or 'ffill')
        
    Returns:
        pandas.DataFrame: Single DataFrame with aligned data
    """
    if align_to == 'gold':
        combined = align_timestamps_to_gold(data)
    elif align_to == 'bitcoin':
        combined = align_timestamps_to_bitcoin(data)
        # Handle NaNs according to user preference
        if handle_nans == 'ffill':
            logger.info("Forward-filling NaN values in non-trading days")
            combined = combined.ffill()
        elif handle_nans == 'drop':
            initial_rows = len(combined)
            combined = combined.dropna()
            logger.info(f"Dropped {initial_rows - len(combined)} rows with NaN values")
    else:
        logger.warning(f"Unknown alignment reference '{align_to}'. Using 'gold' as default.")
        combined = align_timestamps_to_gold(data)
    
    return combined

def calculate_returns(df):
    """
    Calculate percentage returns and additional OHLC-based metrics.
    
    Args:
        df (pandas.DataFrame): DataFrame with price data
        
    Returns:
        pandas.DataFrame: DataFrame with returns and additional metrics
    """
    # Get only the OHLC columns for each asset
    ohlc_cols = [col for col in df.columns if any(x in col for x in ['Open', 'High', 'Low', 'Close'])]
    ohlc_df = df[ohlc_cols]
    
    # Initialize empty DataFrame for results
    results = pd.DataFrame(index=ohlc_df.index)
    
    # Calculate metrics for each asset
    for asset in set(col.split('_')[0] for col in ohlc_cols):
        # Daily percentage change
        results[f'{asset}_return'] = ohlc_df[f'{asset}_Close'].pct_change()
        
        # Log returns
        results[f'{asset}_log_return'] = np.log(ohlc_df[f'{asset}_Close'] / ohlc_df[f'{asset}_Close'].shift(1))
        
        # Intraday volatility (High - Low)
        results[f'{asset}_intraday_vol'] = ohlc_df[f'{asset}_High'] - ohlc_df[f'{asset}_Low']
        
        # Normalized intraday volatility (High - Low)/Open
        results[f'{asset}_norm_intraday_vol'] = (ohlc_df[f'{asset}_High'] - ohlc_df[f'{asset}_Low']) / ohlc_df[f'{asset}_Open']
        
        # Close-Open spread
        results[f'{asset}_close_open_spread'] = ohlc_df[f'{asset}_Close'] - ohlc_df[f'{asset}_Open']
        
        # Normalized close-open spread (Close - Open)/Open
        results[f'{asset}_norm_close_open_spread'] = (ohlc_df[f'{asset}_Close'] - ohlc_df[f'{asset}_Open']) / ohlc_df[f'{asset}_Open']
        
        # True Range (max of: High-Low, |High-PrevClose|, |Low-PrevClose|)
        high_low = ohlc_df[f'{asset}_High'] - ohlc_df[f'{asset}_Low']
        high_prev_close = abs(ohlc_df[f'{asset}_High'] - ohlc_df[f'{asset}_Close'].shift(1))
        low_prev_close = abs(ohlc_df[f'{asset}_Low'] - ohlc_df[f'{asset}_Close'].shift(1))
        results[f'{asset}_true_range'] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        
    # Drop NA values from calculations
    results = results.dropna()
    
    logger.info(f"Calculated returns and metrics with {len(results)} rows")
    logger.info(f"Result columns: {results.columns.tolist()}")
    
    return results

def prepare_data(align_to='gold', handle_nans='drop'):
    """
    Complete data preparation pipeline.
    
    Args:
        align_to (str): Asset to use as alignment reference ('gold' or 'bitcoin')
        handle_nans (str): How to handle NaN values ('drop' or 'ffill')
    
    Returns:
        tuple: (prices_df, returns_df) - DataFrames with aligned price and return data
    """
    # Load raw data
    raw_data = load_data()
    
    # Clean individual datasets
    cleaned_data = clean_data(raw_data)
    
    # Align timestamps based on user preference
    aligned_prices = align_timestamps(cleaned_data, align_to=align_to, handle_nans=handle_nans)
    
    # Calculate returns
    returns = calculate_returns(aligned_prices)
    
    # Save processed data
    file_suffix = f"_{align_to}_aligned"
    if align_to == 'bitcoin' and handle_nans == 'ffill':
        file_suffix += "_ffilled"
        
    aligned_prices.to_csv(f"data/aligned_prices{file_suffix}.csv")
    returns.to_csv(f"data/returns{file_suffix}.csv")
    logger.info(f"Processed data saved to CSV files with suffix '{file_suffix}'")
    
    return aligned_prices, returns

if __name__ == "__main__":
    # Default: align to gold's trading days
    prepare_data(align_to='gold')