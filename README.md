# Bitcoin Correlation Analysis

This project analyzes the correlation between Bitcoin prices and those of gold and NASDAQ stocks.

## Analysis Workflow

1. **Data Collection**: Gathering historical price data for Bitcoin, gold, and NASDAQ.
2. **Data Preparation**: Cleaning and processing the collected data.
3. **Exploratory Analysis**: Initial exploration of price movements and patterns.
4. **Correlation Analysis**: Measuring the correlation between asset prices.
5. **Advanced Analysis**: Further statistical testing and time-series analysis.
6. **Visualization**: Creating visualizations to represent the findings.

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Alpha Vantage API key as an environment variable:
   ```bash
   # Linux/macOS
   export ALPHA_VANTAGE_API_KEY=your_api_key_here
   
   # Windows Command Prompt
   set ALPHA_VANTAGE_API_KEY=your_api_key_here
   
   # Windows PowerShell
   $env:ALPHA_VANTAGE_API_KEY="your_api_key_here"
   ```
   (Get a free key at https://www.alphavantage.co/support/#api-key)
4. Run the analysis using the Jupyter notebooks in the `notebooks` directory

## Project Structure

```
.
├── data_collection.py        # Module for collecting price data for gold and NASDAQ
├── crypto_data_collection.py # Dedicated module for Bitcoin data from Binance
├── data_preparation.py       # Module for cleaning and preparing data
├── correlation_analysis.py   # Module for correlation calculations
├── visualization.py          # Module for generating charts and graphs
├── advanced_analysis.py      # Module for advanced statistical analysis
├── notebooks/                # Jupyter notebooks for interactive analysis
├── data/                     # Directory where downloaded data is stored
└── requirements.txt          # Project dependencies
```

## Data Sources

### Alpha Vantage API
Used for retrieving historical price data for gold and NASDAQ:

- **Free Tier**: 5 API calls per minute, 500 per day
- **API Key**: Get yours for free at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- **Security Note**: Store your API key as an environment variable (ALPHA_VANTAGE_API_KEY) rather than hardcoding it
- **Data Retrieved**:
  - **Gold**: Time Series Daily API for GLD (SPDR Gold Shares ETF as a proxy for gold price)
  - **NASDAQ**: Time Series Daily API for QQQ (NASDAQ-100 Index ETF)

### Binance Data Vision
Used for retrieving historical Bitcoin price data:

- **Bitcoin**: Monthly klines data for BTCUSDT pair at daily interval
- **Advantages**:
  - No API key required
  - No rate limits
  - Complete historical data available

## Using the Code

### Collecting Data

```python
# For gold and NASDAQ data
from data_collection import get_all_data
data = get_all_data(start_date='2018-01-01', end_date='2023-01-01')

# For Bitcoin data specifically
from crypto_data_collection import save_bitcoin_data
bitcoin_csv = save_bitcoin_data(start_date='2018-01-01', end_date='2023-01-01')
```

### Performing Analysis

See the Jupyter notebooks in the `notebooks` directory for examples of how to perform the analysis.

## Troubleshooting

### Alpha Vantage API Issues
- Verify your API key is correct
- Check that you haven't exceeded the rate limits (5 calls per minute, 500 per day)
- If you get empty data, try again after a few minutes

### Binance Data Issues
- For very recent data (last few days), the monthly archives might not be updated yet
- Check your internet connection, as the zip files can be large
- Make sure the date range you requested is valid 