"""
Cryptocurrency Data Collection Module.

This module downloads historical cryptocurrency price data from Binance Data Vision,
specifically focused on Bitcoin (BTC) historical data.
"""

import os
import pandas as pd
import requests
import logging
import datetime
import io
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def get_binance_crypto_data(symbol="BTCUSDT", start_date=None, end_date=None):
    """
    Download historical cryptocurrency data from Binance Data Vision.
    
    Args:
        symbol (str): Symbol to download (e.g., "BTCUSDT")
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pandas.DataFrame: Historical price data
    """
    logger.info(f"Downloading {symbol} data from Binance Data Vision")
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 5 years of data
        start = datetime.datetime.now() - datetime.timedelta(days=5*365)
        start_date = start.strftime('%Y-%m-%d')
    
    # Convert dates to datetime objects
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Initialize empty DataFrame to hold all data
    all_data = pd.DataFrame()
    
    # Generate list of months to download
    current_date = start_date_obj
    while current_date <= end_date_obj:
        year = current_date.year
        month = current_date.month
        
        # Binance Data Vision URL format
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d/{symbol}-1d-{year}-{month:02d}.zip"
        
        logger.info(f"Requesting data from: {url}")
        
        try:
            # Download the zip file
            response = requests.get(url)
            
            if response.status_code == 200:
                # Extract CSV from the zip file
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    # Extract CSV filename - there should be only one file in the zip
                    csv_filename = zip_file.namelist()[0]
                    
                    # Read the CSV data
                    with zip_file.open(csv_filename) as csv_file:
                        # Binance CSV format has no header, so we define our own
                        columns = [
                            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                            'Close time', 'Quote asset volume', 'Number of trades',
                            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                        ]
                        
                        df = pd.read_csv(
                            csv_file, 
                            header=None, 
                            names=columns
                        )
                        
                        # Convert timestamps to datetime with automatic unit detection
                        # Binance can use either milliseconds or microseconds
                        def convert_timestamp_to_datetime(timestamp):
                            # Check timestamp length to determine if it's milliseconds or microseconds
                            # Typical unix timestamp in ms is 13 digits (until year ~2286)
                            # Typical unix timestamp in us is 16 digits
                            timestamp = int(timestamp)
                            if timestamp > 10**15:  # Likely microseconds
                                return pd.to_datetime(timestamp, unit='us')
                            else:  # Likely milliseconds
                                return pd.to_datetime(timestamp, unit='ms')
                        
                        # Apply conversion with validation
                        try:
                            df['Date'] = df['Open time'].apply(convert_timestamp_to_datetime)
                            
                            # Validate dates are within a reasonable range (1990-2050)
                            min_valid_date = pd.Timestamp('1990-01-01')
                            max_valid_date = pd.Timestamp('2050-12-31')
                            
                            invalid_dates = ((df['Date'] < min_valid_date) | (df['Date'] > max_valid_date)).sum()
                            if invalid_dates > 0:
                                logger.warning(f"Found {invalid_dates} dates outside the expected range. Check timestamp conversion.")
                                
                                # If more than 50% of dates are invalid, try the opposite unit
                                if invalid_dates > len(df) * 0.5:
                                    logger.warning("Majority of dates are invalid. Trying alternative conversion method...")
                                    if 'ms' in locals():
                                        df['Date'] = pd.to_datetime(df['Open time'], unit='us')
                                    else:
                                        df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
                        except Exception as e:
                            logger.error(f"Error converting timestamps: {str(e)}")
                            # Fallback to milliseconds which is most common for Binance
                            df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
                        
                        # Filter out unnecessary columns
                        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        
                        # Set Date as index
                        df.set_index('Date', inplace=True)
                        
                        # Convert prices to float
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                        # Append to the main DataFrame
                        all_data = pd.concat([all_data, df])
                        
                        logger.info(f"Successfully downloaded data for {year}-{month:02d}")
            else:
                logger.warning(f"Failed to download data for {year}-{month:02d}: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error downloading data for {year}-{month:02d}: {str(e)}")
        
        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        
        current_date = datetime.datetime(year, month, 1)
    
    if not all_data.empty:
        # Sort by date
        all_data = all_data.sort_index()
        
        # Filter to requested date range
        all_data = all_data.loc[start_date:end_date]
        
        logger.info(f"Downloaded a total of {len(all_data)} days of {symbol} data")
    else:
        logger.warning(f"No data downloaded for {symbol}")
    
    return all_data

def save_bitcoin_data(start_date=None, end_date=None):
    """
    Download and save Bitcoin price data to CSV.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        str: Path to the saved CSV file or None if unsuccessful
    """
    try:
        # Get Bitcoin data from Binance
        btc_data = get_binance_crypto_data(
            symbol="BTCUSDT", 
            start_date=start_date, 
            end_date=end_date
        )
        
        if btc_data.empty:
            logger.warning("No Bitcoin data retrieved, cannot save CSV")
            return None
            
        # Save to CSV
        csv_path = "data/bitcoin_prices.csv"
        btc_data.to_csv(csv_path)
        logger.info(f"Saved Bitcoin data to {csv_path}")
        
        return csv_path
        
    except Exception as e:
        logger.error(f"Error saving Bitcoin data: {str(e)}")
        return None

def update_bitcoin_data(start_date=None, end_date=None):
    """
    Check for missing data in the Bitcoin price CSV and update only the missing dates.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        tuple: (str, int) - Path to the updated CSV file and number of days added
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        # Default to 1 month of data
        start = datetime.datetime.now() - datetime.timedelta(days=30)
        start_date = start.strftime('%Y-%m-%d')
    
    logger.info(f"Checking for missing Bitcoin data from {start_date} to {end_date}")
    
    # Convert dates to datetime objects
    start_date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate complete date range
    date_range = pd.date_range(start=start_date_obj, end=end_date_obj, freq='D')
    expected_dates = pd.DataFrame(index=date_range)
    
    # Check if the data file exists
    csv_path = "data/bitcoin_prices.csv"
    if os.path.exists(csv_path):
        # Load existing data
        try:
            existing_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded existing Bitcoin data with {len(existing_data)} rows")
            
            # Find missing dates
            existing_dates = existing_data.index
            missing_dates_df = expected_dates.loc[~expected_dates.index.isin(existing_dates)]
            missing_dates = [d.strftime('%Y-%m-%d') for d in missing_dates_df.index]
            
            if not missing_dates:
                logger.info(f"No missing Bitcoin data in the specified range")
                return csv_path, 0
            
            logger.info(f"Found {len(missing_dates)} missing dates in Bitcoin data")
            
            # If the missing dates are not continuous, we'll collect data in chunks
            missing_chunks = []
            current_chunk = []
            
            for i, date_str in enumerate(missing_dates):
                if i == 0 or datetime.datetime.strptime(date_str, '%Y-%m-%d') - datetime.datetime.strptime(missing_dates[i-1], '%Y-%m-%d') > datetime.timedelta(days=1):
                    if current_chunk:
                        missing_chunks.append(current_chunk)
                    current_chunk = [date_str]
                else:
                    current_chunk.append(date_str)
                    
            if current_chunk:
                missing_chunks.append(current_chunk)
            
            # Collect data for missing chunks
            new_data_frames = []
            
            for chunk in missing_chunks:
                chunk_start = chunk[0]
                chunk_end = chunk[-1]
                logger.info(f"Collecting missing Bitcoin data from {chunk_start} to {chunk_end}")
                
                chunk_data = get_binance_crypto_data(
                    symbol="BTCUSDT", 
                    start_date=chunk_start, 
                    end_date=chunk_end
                )
                
                if not chunk_data.empty:
                    new_data_frames.append(chunk_data)
            
            if new_data_frames:
                # Concatenate all new data
                new_data = pd.concat(new_data_frames)
                
                # Combine with existing data
                combined_data = pd.concat([existing_data, new_data])
                
                # Remove duplicates and sort
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                combined_data = combined_data.sort_index()
                
                # Save to CSV
                combined_data.to_csv(csv_path)
                logger.info(f"Updated Bitcoin data saved to {csv_path} with {len(new_data)} new rows")
                
                return csv_path, len(new_data)
            else:
                logger.warning("No new Bitcoin data collected")
                return csv_path, 0
                
        except Exception as e:
            logger.error(f"Error processing existing Bitcoin data: {str(e)}")
            # If there's an error with the existing file, collect all data in the range
            return save_bitcoin_data(start_date=start_date, end_date=end_date), 0
    else:
        # If file doesn't exist, collect all data
        logger.info(f"Bitcoin data file not found, collecting all data in range")
        return save_bitcoin_data(start_date=start_date, end_date=end_date), 0

if __name__ == "__main__":
    # Default to downloading 5 years of Bitcoin data
    five_years_ago = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Downloading Bitcoin data from {five_years_ago} to {today}")
    csv_path = save_bitcoin_data(start_date=five_years_ago, end_date=today)
    
    if csv_path:
        logger.info(f"Successfully downloaded and saved Bitcoin data to {csv_path}")
    else:
        logger.error("Failed to download and save Bitcoin data") 