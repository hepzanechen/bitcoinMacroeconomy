"""
Price Behavior Analysis Module.

This module provides tools to identify abnormal price behavior in financial time series data,
using multiple systematic approaches:
1. Volatility measures (rolling standard deviation, GARCH models)
2. Significant price movements detection (exceeding X standard deviations)
3. Drawdowns and rallies detection
4. Frequency analysis (Fourier transforms)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import logging
from arch import arch_model
from datetime import datetime
import matplotlib.dates as mdates
from data_preparation import load_data, clean_data, align_timestamps, calculate_returns

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
os.makedirs('data/analysis', exist_ok=True)

# Import pandas_ta for technical indicators
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    logger.info("pandas_ta successfully imported for technical indicators")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas_ta not available. Will use pandas for technical indicators")


def calculate_volatility_measures(data_df, asset, window_sizes=[20, 60], garch_order=(1, 1)):
    """
    Calculate volatility measures using rolling standard deviation and GARCH models.
    
    Args:
        data_df (pandas.DataFrame): Combined data with returns
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        window_sizes (list): List of window sizes for rolling volatility
        garch_order (tuple): Order of GARCH model (p, q)
        
    Returns:
        pandas.DataFrame: Volatility measures
    """
    # Get return column for the asset
    log_return_col = f"{asset}_log_return"
    
    if log_return_col not in data_df.columns:
        logger.warning(f"Return column {log_return_col} not found in dataframe")
        return pd.DataFrame()
    
    # Create dataframe for volatility measures
    volatility_df = pd.DataFrame(index=data_df.index)
    
    # 1. Rolling standard deviation
    for window in window_sizes:
        vol_col = f'{asset}_vol_{window}d'
        volatility_df[vol_col] = data_df[log_return_col].rolling(window=window).std() * np.sqrt(252)  # Annualized
        
    # 2. GARCH model for conditional volatility
    logger.info(f"Fitting GARCH model for {asset} volatility estimation...")
    try:
        # Remove NaN values which can cause GARCH model to fail
        clean_returns = data_df[log_return_col].dropna()
        
        # Fit GARCH model
        garch_model = arch_model(clean_returns, mean='Zero', vol='GARCH', p=garch_order[0], q=garch_order[1])
        garch_results = garch_model.fit(disp='off')
        
        # Get conditional volatility
        conditional_vol = garch_results.conditional_volatility
        volatility_df[f'{asset}_garch_vol'] = np.nan
        volatility_df.loc[conditional_vol.index, f'{asset}_garch_vol'] = conditional_vol * np.sqrt(252)  # Annualized
        
        logger.info(f"GARCH model for {asset} fitted successfully")
    except Exception as e:
        logger.error(f"Error fitting GARCH model for {asset}: {str(e)}")
    
    return volatility_df


def identify_single_day_movements(data_df, asset, std_threshold=2.0, window=20):
    """
    Identify significant single-day price movements that exceed X standard deviations from the moving average.
    
    Args:
        data_df (pandas.DataFrame): Combined data with returns
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        std_threshold (float): Threshold in number of standard deviations
        window (int): Window size for moving average and volatility calculation
        
    Returns:
        pandas.DataFrame: DataFrame with significant single-day price movements
    """
    # Get return column for the asset
    log_return_col = f"{asset}_log_return"
    
    if log_return_col not in data_df.columns:
        logger.warning(f"Return column {log_return_col} not found in dataframe")
        return pd.DataFrame()
    
    # Calculate moving average of returns
    ma_returns = data_df[log_return_col].rolling(window=window).mean()
    
    # Calculate volatility
    volatility = data_df[log_return_col].rolling(window=window).std()
    
    # Calculate how many standard deviations each return is from the moving average
    significant_moves = pd.DataFrame(index=data_df.index)
    significant_moves['stddev_from_mean'] = (data_df[log_return_col] - ma_returns) / volatility
    
    # Flag significant movements
    significant_moves['is_significant'] = abs(significant_moves['stddev_from_mean']) > std_threshold
    significant_moves['direction'] = np.where(significant_moves['stddev_from_mean'] > 0, 'up', 'down')
    significant_moves['asset'] = asset
    significant_moves['pct_change'] = data_df[log_return_col] * 100  # Convert to percentage
    significant_moves['movement_type'] = 'single_day'
    
    # Only keep significant movements
    single_day_events = significant_moves[significant_moves['is_significant']].copy()
    
    logger.info(f"Identified {len(single_day_events)} significant single-day {asset} price movements (>{std_threshold} StdDev)")
    return single_day_events

def identify_technical_trends(data_df, asset, method='ma_crossover', 
                             short_window=20, long_window=50, min_trend_pct=5.0):
    """
    Identify multi-day trends using technical indicators like MA crossovers, MACD, etc.
    
    Args:
        data_df (pandas.DataFrame): Combined data with prices
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        method (str): Technical method to use ('ma_crossover', 'macd', etc.)
        short_window (int): Short-term window for indicators
        long_window (int): Long-term window for indicators
        min_trend_pct (float): Minimum percentage change to consider a trend significant
        
    Returns:
        pandas.DataFrame: DataFrame with identified trends
    """
    # Get price column for the asset
    close_col = f"{asset}_Close"
    
    if close_col not in data_df.columns:
        logger.warning(f"Price column {close_col} not found in dataframe")
        return pd.DataFrame()
    
    # Create a DataFrame to store trend information
    trends_df = pd.DataFrame()
    
    # Get the price series
    prices = data_df[close_col]
    
    # Calculate indicators based on method
    if method == 'ma_crossover':
        # Calculate short and long moving averages
        if PANDAS_TA_AVAILABLE:
            # Create a DataFrame with OHLCV data for pandas_ta
            ohlcv = pd.DataFrame({
                'open': prices,
                'high': prices,
                'low': prices,
                'close': prices,
                'volume': 0
            }, index=prices.index)
            
            # Calculate SMAs using pandas_ta
            short_ma = ohlcv.ta.sma(length=short_window)
            long_ma = ohlcv.ta.sma(length=long_window)
        else:
            short_ma = prices.rolling(window=short_window).mean()
            long_ma = prices.rolling(window=long_window).mean()
        
        # Create signal: 1 when short_ma > long_ma (bullish), -1 when short_ma < long_ma (bearish)
        signal = pd.Series(0, index=prices.index)
        signal[short_ma > long_ma] = 1
        signal[short_ma < long_ma] = -1
        
        # Detect crossovers (signal changes)
        signal_shifts = signal.shift(1)
        crossovers = (signal != signal_shifts) & ~signal_shifts.isna()
        
        # Find start points of trends
        trend_starts = prices.index[crossovers]
        
        # Process each trend
        trends = []
        for i, start_date in enumerate(trend_starts):
            # Skip the last detected crossover as we don't know when it ends
            if i >= len(trend_starts) - 1:
                continue
                
            end_date = trend_starts[i+1]
            direction = 'up' if signal.loc[start_date] == 1 else 'down'
            
            # Calculate price change during trend
            start_price = prices.loc[start_date]
            end_price = prices.loc[end_date]
            pct_change = 100 * (end_price - start_price) / start_price
            
            # Only record trends with significant price change
            if abs(pct_change) >= min_trend_pct:
                trend_info = {
                    'asset': asset,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': (end_date - start_date).days,
                    'start_price': start_price,
                    'end_price': end_price,
                    'pct_change': pct_change,
                    'direction': direction,
                    'method': 'ma_crossover',
                    'short_window': short_window,
                    'long_window': long_window,
                    'strength': abs(pct_change) / abs((end_date - start_date).days) * 7,  # % per week
                    'magnitude': 'large' if abs(pct_change) >= 2*min_trend_pct else 'moderate'
                }
                trends.append(trend_info)
    
    elif method == 'macd':
        # MACD indicator
        if PANDAS_TA_AVAILABLE:
            # Create a DataFrame with OHLCV data for pandas_ta
            ohlcv = pd.DataFrame({
                'open': prices,
                'high': prices,
                'low': prices,
                'close': prices,
                'volume': 0
            }, index=prices.index)
            
            # Calculate MACD using pandas_ta
            macd_result = ohlcv.ta.macd(fast=short_window, slow=long_window, signal=9)
            macd = macd_result[f'MACD_{short_window}_{long_window}_9']
            signal = macd_result[f'MACDs_{short_window}_{long_window}_9']
            hist = macd_result[f'MACDh_{short_window}_{long_window}_9']
        else:
            # Calculate MACD manually
            short_ema = prices.ewm(span=short_window, adjust=False).mean()
            long_ema = prices.ewm(span=long_window, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
        
        # Create trend signal: 1 when MACD > Signal (bullish), -1 when MACD < Signal (bearish)
        trend_signal = pd.Series(0, index=prices.index)
        trend_signal[macd > signal] = 1
        trend_signal[macd < signal] = -1
        
        # Detect crossovers (signal changes)
        signal_shifts = trend_signal.shift(1)
        crossovers = (trend_signal != signal_shifts) & ~signal_shifts.isna()
        
        # Find start points of trends
        trend_starts = prices.index[crossovers]
        
        # Process each trend
        trends = []
        for i, start_date in enumerate(trend_starts):
            # Skip the last detected crossover as we don't know when it ends
            if i >= len(trend_starts) - 1:
                continue
                
            end_date = trend_starts[i+1]
            direction = 'up' if trend_signal.loc[start_date] == 1 else 'down'
            
            # Calculate price change during trend
            start_price = prices.loc[start_date]
            end_price = prices.loc[end_date]
            pct_change = 100 * (end_price - start_price) / start_price
            
            # Only record trends with significant price change
            if abs(pct_change) >= min_trend_pct:
                trend_info = {
                    'asset': asset,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': (end_date - start_date).days,
                    'start_price': start_price,
                    'end_price': end_price,
                    'pct_change': pct_change,
                    'direction': direction,
                    'method': 'macd',
                    'short_window': short_window,
                    'long_window': long_window,
                    'strength': abs(pct_change) / abs((end_date - start_date).days) * 7,  # % per week
                    'magnitude': 'large' if abs(pct_change) >= 2*min_trend_pct else 'moderate'
                }
                trends.append(trend_info)
                
    elif method == 'bollinger_breakout':
        # Bollinger Bands
        window = short_window
        num_std = 2
        
        if PANDAS_TA_AVAILABLE:
            # Create a DataFrame with OHLCV data for pandas_ta
            ohlcv = pd.DataFrame({
                'open': prices,
                'high': prices,
                'low': prices,
                'close': prices,
                'volume': 0
            }, index=prices.index)
            
            # Calculate Bollinger Bands using pandas_ta
            bbands = ohlcv.ta.bbands(length=window, std=num_std)
            upper = bbands[f'BBU_{window}_{num_std}.0']
            middle = bbands[f'BBM_{window}_{num_std}.0']
            lower = bbands[f'BBL_{window}_{num_std}.0']
        else:
            # Calculate Bollinger Bands manually
            middle = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = middle + (std * num_std)
            lower = middle - (std * num_std)
        
        # Identify breakouts: 1 for upside breakout, -1 for downside breakout
        breakout = pd.Series(0, index=prices.index)
        breakout[prices > upper] = 1
        breakout[prices < lower] = -1
        
        # Detect breakout changes
        breakout_shifts = breakout.shift(1)
        breakout_starts = (breakout != 0) & (breakout_shifts == 0)
        breakout_ends = (breakout == 0) & (breakout_shifts != 0)
        
        # Find start points of breakouts
        start_dates = prices.index[breakout_starts]
        
        # Process each breakout
        trends = []
        for start_date in start_dates:
            # Find the end of this breakout
            future_ends = prices.index[breakout_ends & (prices.index > start_date)]
            if len(future_ends) == 0:
                continue  # No end found, skip
                
            end_date = future_ends[0]
            direction = 'up' if breakout.loc[start_date] == 1 else 'down'
            
            # Calculate price change during breakout
            start_price = prices.loc[start_date]
            end_price = prices.loc[end_date]
            pct_change = 100 * (end_price - start_price) / start_price
            
            # Only record breakouts with significant price change
            if abs(pct_change) >= min_trend_pct:
                trend_info = {
                    'asset': asset,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': (end_date - start_date).days,
                    'start_price': start_price,
                    'end_price': end_price,
                    'pct_change': pct_change,
                    'direction': direction,
                    'method': 'bollinger_breakout',
                    'window': window,
                    'strength': abs(pct_change) / abs((end_date - start_date).days) * 7,  # % per week
                    'magnitude': 'large' if abs(pct_change) >= 2*min_trend_pct else 'moderate'
                }
                trends.append(trend_info)
                
    else:
        logger.warning(f"Unknown technical method '{method}'. No trends identified.")
        return pd.DataFrame()
    
    # Create DataFrame from trends
    trends_df = pd.DataFrame(trends) if trends else pd.DataFrame()
    
    if not trends_df.empty:
        logger.info(f"Identified {len(trends_df)} {method} trends for {asset}")
        # Set index to start_date for consistency
        if 'start_date' in trends_df.columns:
            trends_df.set_index('start_date', inplace=True)
    
    return trends_df

def analyze_volume_patterns(data_df, asset, lookback_period=20):
    """
    Analyze volume patterns and identify abnormal volume events.
    
    Args:
        data_df (pandas.DataFrame): Combined data with prices and volume
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        lookback_period (int): Lookback period for volume averages
        
    Returns:
        pandas.DataFrame: DataFrame with volume analysis results
    """
    volume_col = f"{asset}_Volume"
    
    if volume_col not in data_df.columns:
        logger.warning(f"Volume column {volume_col} not found in dataframe")
        return pd.DataFrame()
    
    # Create dataframe for volume analysis
    volume_df = pd.DataFrame(index=data_df.index)
    
    # Calculate volume moving average
    volume_df['volume_ma'] = data_df[volume_col].rolling(window=lookback_period).mean()
    
    # Calculate volume ratio (current/average)
    volume_df['volume_ratio'] = data_df[volume_col] / volume_df['volume_ma']
    
    # Identify abnormal volume (> 2 standard deviations)
    volume_std = data_df[volume_col].rolling(window=lookback_period).std()
    volume_df['abnormal_volume'] = data_df[volume_col] > (volume_df['volume_ma'] + 2*volume_std)
    
    # Extract abnormal volume events
    abnormal_events = volume_df[volume_df['abnormal_volume']].copy()
    abnormal_events['asset'] = asset
    abnormal_events['actual_volume'] = data_df.loc[abnormal_events.index, volume_col]
    
    logger.info(f"Identified {len(abnormal_events)} abnormal volume events for {asset}")
    return abnormal_events

def analyze_intraday_volatility(data_df, asset, window=20, volatility_threshold=1.5):
    """
    Analyze intraday volatility using high-low range.
    
    Args:
        data_df (pandas.DataFrame): Combined data with OHLC prices
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        window (int): Lookback window for calculating average ranges
        volatility_threshold (float): Threshold as multiplier of average range
        
    Returns:
        pandas.DataFrame: DataFrame with intraday volatility analysis
    """
    # Check if required columns exist
    required_cols = [f"{asset}_{col}" for col in ['High', 'Low', 'Close']]
    if not all(col in data_df.columns for col in required_cols):
        logger.warning(f"Required High/Low/Close columns for {asset} not found in dataframe")
        return pd.DataFrame()
    
    # Extract required columns
    high_col = f"{asset}_High"
    low_col = f"{asset}_Low"
    close_col = f"{asset}_Close"
    
    # Create dataframe for volatility analysis
    volatility_df = pd.DataFrame(index=data_df.index)
    volatility_df['asset'] = asset
    
    # Calculate high-low range (absolute and percentage)
    volatility_df['hl_range'] = data_df[high_col] - data_df[low_col]
    volatility_df['hl_range_pct'] = 100 * volatility_df['hl_range'] / data_df[close_col]
    
    # Calculate average range over window
    volatility_df['avg_hl_range_pct'] = volatility_df['hl_range_pct'].rolling(window=window).mean()
    
    # Calculate ratio of current range to average range
    volatility_df['range_ratio'] = volatility_df['hl_range_pct'] / volatility_df['avg_hl_range_pct']
    
    # Identify high volatility days (range > threshold * average range)
    volatility_df['high_intraday_volatility'] = volatility_df['hl_range_pct'] > (
        volatility_threshold * volatility_df['avg_hl_range_pct']
    )
    
    # Extract high volatility events
    high_vol_events = volatility_df[volatility_df['high_intraday_volatility']].copy()
    
    logger.info(f"Identified {len(high_vol_events)} high intraday volatility events for {asset}")
    return volatility_df, high_vol_events

def analyze_price_gaps(data_df, asset, window=20, volatility_threshold=1.5):
    """
    Analyze gaps between previous close and current open.
    
    Args:
        data_df (pandas.DataFrame): Combined data with OHLC prices
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        window (int): Lookback window for calculating average gaps
        volatility_threshold (float): Threshold as multiplier of average gap
        
    Returns:
        pandas.DataFrame: DataFrame with gap analysis
    """
    # Check if required columns exist
    required_cols = [f"{asset}_{col}" for col in ['Open', 'Close']]
    if not all(col in data_df.columns for col in required_cols):
        logger.warning(f"Required Open/Close columns for {asset} not found in dataframe")
        return pd.DataFrame()
    
    # Extract required columns
    open_col = f"{asset}_Open"
    close_col = f"{asset}_Close"
    
    # Create dataframe for gap analysis
    gap_df = pd.DataFrame(index=data_df.index)
    gap_df['asset'] = asset
    
    # Calculate gaps (current open - previous close)
    gap_df['gap'] = data_df[open_col] - data_df[close_col].shift(1)
    gap_df['gap_pct'] = 100 * gap_df['gap'] / data_df[close_col].shift(1)
    
    # Calculate average absolute gap percentage
    gap_df['avg_gap_pct'] = gap_df['gap_pct'].abs().rolling(window=window).mean()
    
    # Calculate ratio of current gap to average gap
    gap_df['gap_ratio'] = gap_df['gap_pct'].abs() / gap_df['avg_gap_pct']
    
    # Identify significant gaps
    gap_df['significant_gap'] = gap_df['gap_pct'].abs() > (
        volatility_threshold * gap_df['avg_gap_pct']
    )
    gap_df['gap_direction'] = np.where(gap_df['gap_pct'] > 0, 'up', 'down')
    
    # Extract significant gap events
    significant_gaps = gap_df[gap_df['significant_gap']].copy()
    
    logger.info(f"Identified {len(significant_gaps)} significant price gap events for {asset}")
    return gap_df, significant_gaps

def analyze_price_rejection(data_df, asset, wick_threshold=2.0):
    """
    Analyze price rejection patterns based on candlestick wicks.
    
    Args:
        data_df (pandas.DataFrame): Combined data with OHLC prices
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        wick_threshold (float): Threshold for wick to body ratio to identify rejections
        
    Returns:
        pandas.DataFrame: DataFrame with price rejection analysis
    """
    # Check if required columns exist
    required_cols = [f"{asset}_{col}" for col in ['Open', 'High', 'Low', 'Close']]
    if not all(col in data_df.columns for col in required_cols):
        logger.warning(f"Required OHLC columns for {asset} not found in dataframe")
        return pd.DataFrame()
    
    # Extract OHLC data
    open_col = f"{asset}_Open"
    high_col = f"{asset}_High"
    low_col = f"{asset}_Low"
    close_col = f"{asset}_Close"
    
    # Create dataframe for price rejection analysis
    rejection_df = pd.DataFrame(index=data_df.index)
    rejection_df['asset'] = asset
    
    # Calculate upper and lower wicks
    rejection_df['upper_wick'] = data_df[high_col] - np.maximum(data_df[open_col], data_df[close_col])
    rejection_df['lower_wick'] = np.minimum(data_df[open_col], data_df[close_col]) - data_df[low_col]
    rejection_df['body_size'] = abs(data_df[close_col] - data_df[open_col])
    
    # Calculate wick to body ratios
    # Handle division by zero or very small body sizes
    min_body_size = 1e-10
    rejection_df['upper_wick_ratio'] = rejection_df['upper_wick'] / np.maximum(rejection_df['body_size'], min_body_size)
    rejection_df['lower_wick_ratio'] = rejection_df['lower_wick'] / np.maximum(rejection_df['body_size'], min_body_size)
    
    # Identify rejection patterns (wick at least 2x body size)
    rejection_df['upper_rejection'] = rejection_df['upper_wick_ratio'] > wick_threshold
    rejection_df['lower_rejection'] = rejection_df['lower_wick_ratio'] > wick_threshold
    
    # Assign rejection pattern types
    rejection_df['rejection_pattern'] = 'none'
    rejection_df.loc[rejection_df['upper_rejection'], 'rejection_pattern'] = 'upper'
    rejection_df.loc[rejection_df['lower_rejection'], 'rejection_pattern'] = 'lower'
    rejection_df.loc[rejection_df['upper_rejection'] & rejection_df['lower_rejection'], 'rejection_pattern'] = 'both'
    
    # Extract rejection events
    rejection_events = rejection_df[rejection_df['rejection_pattern'] != 'none'].copy()
    
    logger.info(f"Identified {len(rejection_events)} price rejection events for {asset}")
    return rejection_df, rejection_events

def analyze_daily_strength(data_df, asset):
    """
    Analyze daily strength based on close position within day's range.
    
    Args:
        data_df (pandas.DataFrame): Combined data with OHLC prices
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        
    Returns:
        pandas.DataFrame: DataFrame with daily strength analysis
    """
    # Check if required columns exist
    required_cols = [f"{asset}_{col}" for col in ['High', 'Low', 'Close']]
    if not all(col in data_df.columns for col in required_cols):
        logger.warning(f"Required High/Low/Close columns for {asset} not found in dataframe")
        return pd.DataFrame()
    
    # Extract required columns
    high_col = f"{asset}_High"
    low_col = f"{asset}_Low"
    close_col = f"{asset}_Close"
    
    # Create dataframe for daily strength analysis
    strength_df = pd.DataFrame(index=data_df.index)
    strength_df['asset'] = asset
    
    # Calculate high-low range
    strength_df['hl_range'] = data_df[high_col] - data_df[low_col]
    
    # Position of close within day's range (0-1 scale, higher means close near high)
    strength_df['close_position'] = (data_df[close_col] - data_df[low_col]) / strength_df['hl_range']
    
    # Classify daily strength
    strength_df['daily_strength'] = 'neutral'
    strength_df.loc[strength_df['close_position'] >= 0.8, 'daily_strength'] = 'strong_bullish'
    strength_df.loc[strength_df['close_position'] >= 0.6, 'daily_strength'] = 'bullish'
    strength_df.loc[strength_df['close_position'] <= 0.2, 'daily_strength'] = 'strong_bearish'
    strength_df.loc[strength_df['close_position'] <= 0.4, 'daily_strength'] = 'bearish'
    
    # Extract strong trend days
    strong_days = strength_df[
        (strength_df['daily_strength'] == 'strong_bullish') | 
        (strength_df['daily_strength'] == 'strong_bearish')
    ].copy()
    
    logger.info(f"Identified {len(strong_days)} strong trend days for {asset}")
    return strength_df, strong_days

def analyze_ohlc_behavior(data_df, asset, window=20, volatility_threshold=1.5):
    """
    Analyze price behavior using Open, High, Low, Close data to identify:
    1. Intraday volatility (high-low range)
    2. Gap analysis (difference between previous close and current open)
    3. Price rejection patterns (long wicks/shadows on candlesticks)
    4. Daily trend strength
    
    This function combines the results from specialized analysis functions.
    
    Args:
        data_df (pandas.DataFrame): Combined data with OHLC prices
        asset (str): Asset name ('bitcoin', 'nasdaq', 'gold')
        window (int): Lookback window for calculating average ranges
        volatility_threshold (float): Threshold as multiplier of average range
        
    Returns:
        tuple: (Full analysis DataFrame, Significant events DataFrame)
    """
    # Check if required columns exist
    required_cols = [f"{asset}_{col}" for col in ['Open', 'High', 'Low', 'Close']]
    if not all(col in data_df.columns for col in required_cols):
        logger.warning(f"One or more required OHLC columns for {asset} not found in dataframe")
        return pd.DataFrame(), pd.DataFrame()
    
    # Run individual analyses
    volatility_df, volatility_events = analyze_intraday_volatility(data_df, asset, window, volatility_threshold)
    gap_df, gap_events = analyze_price_gaps(data_df, asset, window, volatility_threshold)
    rejection_df, rejection_events = analyze_price_rejection(data_df, asset)
    strength_df, strength_events = analyze_daily_strength(data_df, asset)
    
    # Combine results
    # Start with volatility dataframe as base
    combined_df = volatility_df.copy()
    
    # Add gap analysis
    for col in ['gap', 'gap_pct', 'significant_gap', 'gap_direction']:
        if col in gap_df.columns:
            combined_df[col] = gap_df[col]
    
    # Add rejection analysis
    for col in ['upper_wick_ratio', 'lower_wick_ratio', 'rejection_pattern']:
        if col in rejection_df.columns:
            combined_df[col] = rejection_df[col]
    
    # Add strength analysis
    for col in ['close_position', 'daily_strength']:
        if col in strength_df.columns:
            combined_df[col] = strength_df[col]
    
    # Create combined events dataframe
    # Concatenate all event dataframes
    all_events = pd.concat([
        volatility_events.assign(event_type='high_volatility'),
        gap_events.assign(event_type='significant_gap'),
        rejection_events.assign(event_type='price_rejection'),
        strength_events.assign(event_type='strong_trend')
    ], axis=0)
    
    # Sort by date
    if not all_events.empty:
        all_events.sort_index(inplace=True)
    
    logger.info(f"Combined analysis identified {len(all_events)} significant OHLC price behavior events for {asset}")
    
    return combined_df, all_events

if __name__ == "__main__":
    # Main execution code would go here
    pass
