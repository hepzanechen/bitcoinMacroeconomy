"""
Data Collection Module for Bitcoin Correlation Analysis.

This module downloads historical price data for Bitcoin, gold, and NASDAQ using Alpha Vantage API.
For Bitcoin data, it uses the dedicated crypto_data_collection module to get data from Binance.
"""

import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
import time
import crypto_data_collection as crypto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
# Alpha Vantage API key - Get from environment variable or use a default for development
API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
if not API_KEY:
    logger.warning("ALPHA_VANTAGE_API_KEY environment variable not set. Please set it before using this module.")


def get_alpha_vantage_data(symbol, function="TIME_SERIES_DAILY", output_size="full", market="USD", **kwargs):
    """
    Download historical price data using Alpha Vantage API.
    
    Args:
        symbol (str): Symbol to download (e.g., "GLD", "QQQ")
        function (str): API function to use
        output_size (str): Amount of data to retrieve (full or compact)
        market (str): Market for cryptocurrency (USD, EUR, etc.)
        **kwargs: Additional parameters for specific API functions
        
    Returns:
        pandas.DataFrame: Historical price data
    """
    logger.info(f"Downloading {symbol} data using Alpha Vantage API")
    # Alpha Vantage API key - Get from environment variable or use a default for development
    API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not API_KEY:
        logger.warning("ALPHA_VANTAGE_API_KEY environment variable not set. Please set it before using this module.")

    # Build base URL
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}"
    
    # Add function-specific parameters
    if function == "TIME_SERIES_DAILY":
        url += f"&outputsize={output_size}"
    elif function == "DIGITAL_CURRENCY_DAILY":
        url += f"&market={market}"
    
    # Add any additional parameters
    for key, value in kwargs.items():
        url += f"&{key}={value}"
    
    # Add API key
    url += f"&apikey={API_KEY}"
    
    logger.info(f"API URL: {url}")
    
    try:
        r = requests.get(url)
        data = r.json()
        
        # Check for API error messages
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API Error: {data['Error Message']}")
            return pd.DataFrame()
            
        if "Note" in data:
            logger.warning(f"Alpha Vantage API Note: {data['Note']}")
            
        if "Information" in data:
            logger.info(f"Alpha Vantage API Information: {data['Information']}")
            return pd.DataFrame()  # Return empty DataFrame for information messages
        
        # Extract time series data based on function
        if function == "TIME_SERIES_DAILY":
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                logger.error(f"Key '{time_series_key}' not found in API response")
                logger.debug(f"Response keys: {list(data.keys())}")
                return pd.DataFrame()
                
            time_series = data.get(time_series_key, {})
            # Rename columns to match our expected format
            df = pd.DataFrame(time_series).T
            df.index = pd.to_datetime(df.index)
            # Sort index to ensure chronological order
            df = df.sort_index()
            # Convert columns to numeric and rename
            df = df.apply(pd.to_numeric)
            if not df.empty:
                df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High', 
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                }, inplace=True)
        
        logger.info(f"Successfully downloaded {len(df)} rows for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {str(e)}")
        if 'df' in locals() and isinstance(df, pd.DataFrame) and df.empty:
            logger.error("Empty data returned from Alpha Vantage")
        return pd.DataFrame()  # Return empty DataFrame on error

def get_all_data(start_date=None, end_date=None):
    """
    Download data for Bitcoin, gold, and NASDAQ.
    Bitcoin data comes from Binance via crypto_data_collection module.
    Gold and NASDAQ data come from Alpha Vantage API.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format (default: 5 years ago)
        end_date (str): End date in YYYY-MM-DD format (default: today)
        
    Returns:
        dict: Dictionary containing DataFrames for each asset
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 5 years of data
        start = datetime.now() - timedelta(days=5*365)
        start_date = start.strftime('%Y-%m-%d')
    
    logger.info(f"Collecting data from {start_date} to {end_date}")
    
    # Define symbols for Alpha Vantage
    symbols = {
        'nasdaq': {
            'symbol': 'QQQ',  # Use QQQ ETF which tracks NASDAQ-100 
            'function': 'TIME_SERIES_DAILY',
            'kwargs': {}
        },
        'gold': {
            'symbol': 'GLD',  # Using GLD ETF as proxy for gold
            'function': 'TIME_SERIES_DAILY',
            'kwargs': {}
        }
    }
    
    # Dictionary to store all data
    data = {}
    
    # Get Bitcoin data from Binance
    logger.info("Getting Bitcoin data from Binance via crypto_data_collection module")
    bitcoin_csv = crypto.save_bitcoin_data(start_date=start_date, end_date=end_date)
    
    if bitcoin_csv and os.path.exists(bitcoin_csv):
        # Read the saved Bitcoin data
        bitcoin_df = pd.read_csv(bitcoin_csv, index_col=0, parse_dates=True)
        data['bitcoin'] = bitcoin_df
        logger.info(f"Retrieved {len(bitcoin_df)} days of Bitcoin data")
    else:
        logger.warning("Failed to retrieve Bitcoin data")
    
    # Get data for gold and NASDAQ using Alpha Vantage
    for name, config in symbols.items():
        # Add delay to avoid API rate limiting
        logger.info(f"Waiting 15 seconds before next API call to avoid rate limiting...")
        time.sleep(15)  # Alpha Vantage free tier has a rate limit
            
        df = get_alpha_vantage_data(
            symbol=config['symbol'],
            function=config['function'],
            **config.get('kwargs', {})
        )
        
        # Filter by date range
        if not df.empty:
            try:
                # Make sure the date index is sorted
                df = df.sort_index()
                
                # Use loc only if dates are within the available range
                start_date_obj = pd.to_datetime(start_date)
                end_date_obj = pd.to_datetime(end_date)
                
                # Find closest available dates
                if start_date_obj < df.index.min():
                    actual_start = df.index.min()
                    logger.warning(f"Requested start date {start_date} is before available {name} data. Using {actual_start} instead.")
                else:
                    actual_start = start_date_obj
                    
                if end_date_obj > df.index.max():
                    actual_end = df.index.max()
                    logger.warning(f"Requested end date {end_date} is after available {name} data. Using {actual_end} instead.")
                else:
                    actual_end = end_date_obj
                
                # Filter data within available range
                filtered_df = df.loc[actual_start:actual_end]
                data[name] = filtered_df
                
                # Save to CSV
                csv_path = f"data/{name}_prices.csv"
                filtered_df.to_csv(csv_path)
                logger.info(f"Saved {name} data to {csv_path}")
            except Exception as e:
                logger.error(f"Error processing {name} data: {str(e)}")
        else:
            logger.warning(f"No data retrieved for {name}")
    
    return data

def test_api_key():
    """Test if the Alpha Vantage API key is valid and working."""
    # Alpha Vantage API key - Get from environment variable or use a default for development
    API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if not API_KEY:
        logger.error("No Alpha Vantage API key provided. Set the ALPHA_VANTAGE_API_KEY environment variable.")
        return False
        
    test_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey={API_KEY}"
    try:
        r = requests.get(test_url)
        data = r.json()
        if "Error Message" in data:
            logger.error(f"API key test failed: {data['Error Message']}")
            return False
        if "Time Series (Daily)" in data:
            logger.info("API key test successful")
            return True
        return False
    except Exception as e:
        logger.error(f"API key test error: {str(e)}")
        return False

def update_asset_data(asset, start_date=None, end_date=None):
    """
    Check for missing data in the specified asset's price CSV and update only the missing dates.
    
    Args:
        asset (str): Asset name ('nasdaq', 'gold')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        tuple: (str, int) - Path to the updated CSV file and number of days added
    """
    # Validate asset name
    valid_assets = ['nasdaq', 'gold']
    if asset.lower() not in valid_assets:
        logger.error(f"Invalid asset: {asset}. Must be one of {valid_assets}")
        return None, 0
    
    asset = asset.lower()
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 1 month of data
        start = datetime.now() - timedelta(days=30)
        start_date = start.strftime('%Y-%m-%d')
    
    logger.info(f"Checking for missing {asset} data from {start_date} to {end_date}")
    
    # Convert dates to datetime objects
    start_date_obj = pd.to_datetime(start_date)
    end_date_obj = pd.to_datetime(end_date)
    
    # Generate complete date range (business days for stock market data)
    if asset == 'nasdaq' or asset == 'gold':
        # These assets only trade on business days
        date_range = pd.bdate_range(start=start_date_obj, end=end_date_obj)
    else:
        # Use calendar days for other assets like crypto
        date_range = pd.date_range(start=start_date_obj, end=end_date_obj, freq='D')
    
    expected_dates = pd.DataFrame(index=date_range)
    
    # Check if the data file exists
    csv_path = f"data/{asset}_prices.csv"
    if os.path.exists(csv_path):
        # Load existing data
        try:
            existing_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded existing {asset} data with {len(existing_data)} rows")
            
            # Find missing dates
            existing_dates = existing_data.index
            missing_dates_df = expected_dates.loc[~expected_dates.index.isin(existing_dates)]
            missing_dates = [d.strftime('%Y-%m-%d') for d in missing_dates_df.index]
            
            if not missing_dates:
                logger.info(f"No missing {asset} data in the specified range")
                return csv_path, 0
            
            logger.info(f"Found {len(missing_dates)} missing dates in {asset} data")
            
            # For Alpha Vantage, we need to collect all data again, but we'll only update the missing dates
            # Get the symbol configuration for the asset
            symbols = {
                'nasdaq': {
                    'symbol': 'QQQ',
                    'function': 'TIME_SERIES_DAILY',
                    'kwargs': {}
                },
                'gold': {
                    'symbol': 'GLD',
                    'function': 'TIME_SERIES_DAILY',
                    'kwargs': {}
                }
            }
            
            config = symbols.get(asset)
            
            if not config:
                logger.error(f"Configuration not found for asset: {asset}")
                return csv_path, 0
                
            # Check if we need to wait for API rate limiting
            logger.info(f"Waiting 15 seconds before API call to avoid rate limiting...")
            time.sleep(15)
            
            # Get updated data from Alpha Vantage
            df = get_alpha_vantage_data(
                symbol=config['symbol'],
                function=config['function'],
                **config.get('kwargs', {})
            )
            
            if df.empty:
                logger.warning(f"Failed to retrieve new {asset} data")
                return csv_path, 0
                
            # Filter to just the dates we need
            new_data = df[df.index.isin(pd.to_datetime(missing_dates))]
            
            if new_data.empty:
                logger.warning(f"No new {asset} data available for the missing dates")
                return csv_path, 0
                
            # Combine with existing data
            combined_data = pd.concat([existing_data, new_data])
            
            # Remove duplicates and sort
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            
            # Save to CSV
            combined_data.to_csv(csv_path)
            logger.info(f"Updated {asset} data saved to {csv_path} with {len(new_data)} new rows")
            
            return csv_path, len(new_data)
            
        except Exception as e:
            logger.error(f"Error processing existing {asset} data: {str(e)}")
            # If there's an error, we'll proceed with collecting all data
    
    # If file doesn't exist or there was an error, collect all data
    logger.info(f"{asset} data file not found or couldn't be processed, collecting all data in range")
    
    # Collect all data for the date range
    data = get_all_data(start_date=start_date, end_date=end_date)
    
    if asset in data and not data[asset].empty:
        return f"data/{asset}_prices.csv", len(data[asset])
    else:
        logger.warning(f"Failed to collect {asset} data")
        return None, 0

def update_missing_data(start_date=None, end_date=None, assets=None):
    """
    Update missing data for specified assets within a date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        assets (list): List of assets to update ('bitcoin', 'nasdaq', 'gold'). If None, updates all.
        
    Returns:
        dict: Dictionary with results for each asset
    """
    if assets is None:
        assets = ['bitcoin', 'nasdaq', 'gold']
    
    results = {}
    
    for asset in assets:
        logger.info(f"Updating {asset} data...")
        
        if asset.lower() == 'bitcoin':
            # Import here to avoid circular import
            import crypto_data_collection as crypto
            csv_path, rows_added = crypto.update_bitcoin_data(start_date=start_date, end_date=end_date)
        else:
            csv_path, rows_added = update_asset_data(asset, start_date=start_date, end_date=end_date)
        
        results[asset] = {
            'csv_path': csv_path,
            'rows_added': rows_added
        }
        
        logger.info(f"Added {rows_added} rows to {asset} data")
        
        # Add delay between API calls to avoid rate limiting
        if asset != assets[-1]:
            logger.info(f"Waiting 15 seconds before next API call...")
            time.sleep(15)
    
    return results

if __name__ == "__main__":
    # Test API key first
    if not API_KEY:
        logger.error("Please set your Alpha Vantage API key as an environment variable:")
        print("ERROR: Missing Alpha Vantage API key. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        print("Example: export ALPHA_VANTAGE_API_KEY=your_api_key_here")
    elif test_api_key():
        # If run directly, download data
        get_all_data()
    else:
        logger.error("Please check your Alpha Vantage API key")
        print("ERROR: Invalid Alpha Vantage API key. Please check the ALPHA_VANTAGE_API_KEY environment variable.") 