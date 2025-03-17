"""
Correlation Analysis Module for Bitcoin Correlation Analysis.

This module calculates various correlation metrics between Bitcoin, gold, and NASDAQ.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_theme(font_scale=1.2)

# Create necessary directories
os.makedirs('data', exist_ok=True)
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

def calculate_correlations(returns):
    """
    Calculate correlations between asset returns.
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
        
    Returns:
        dict: Dictionary containing correlation DataFrames
    """
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return') and not col.endswith('log_return')]
    ret_data = returns[return_columns]
    
    # Calculate Pearson correlation (linear relationship)
    pearson_corr = ret_data.corr(method='pearson')
    
    # Calculate Spearman rank correlation (monotonic relationship, not necessarily linear)
    spearman_corr = ret_data.corr(method='spearman')
    
    # Save correlation matrices to CSV
    pearson_corr.to_csv("data/pearson_correlation.csv")
    spearman_corr.to_csv("data/spearman_correlation.csv")
    
    logger.info("Calculated and saved correlation matrices")
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr
    }

def plot_correlation_heatmaps(correlations):
    """
    Plot correlation heatmaps.
    
    Args:
        correlations (dict): Dictionary containing correlation DataFrames
    """
    # Plot heatmaps for each correlation type
    for corr_type, corr_matrix in correlations.items():
        plt.figure(figsize=(10, 8))
        
        # Removing the mask to display the full matrix
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap=cmap, 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5, 
            fmt=".2f"
        )
        
        plt.title(f'{corr_type.capitalize()} Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'plots/{corr_type}_correlation_heatmap.png', dpi=300)
        plt.close()
        
        logger.info(f"Created {corr_type} correlation heatmap")

def calculate_rolling_correlations(returns, window=30):
    """
    Calculate rolling correlations over time.
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
        window (int): Rolling window size in days
        
    Returns:
        dict: Dictionary containing rolling correlation DataFrames
    """
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return') and not col.endswith('log_return')]
    ret_data = returns[return_columns]
    
    # Dictionary to store results
    rolling_corrs = {}
    
    # Calculate rolling correlations between Bitcoin and each asset
    bitcoin_col = 'bitcoin_return'
    for col in return_columns:
        if col != bitcoin_col:
            # Rolling Pearson correlation (default method)
            rolling_corr = ret_data[[bitcoin_col, col]].rolling(window=window).corr()
            
            # Let's print the column names after reset_index to debug
            logger.info(f"Creating rolling correlation between {bitcoin_col} and {col}")
            
            try:
                # The safer way to extract what we need - using index levels directly
                # First filter to just get the correlation values for Bitcoin vs. the other asset
                bitcoin_vs_asset = rolling_corr.xs(bitcoin_col, level=0)[col]
                
                # Convert to DataFrame with date as index
                result_df = pd.DataFrame({
                    f'btc_{col.split("_")[0]}_corr': bitcoin_vs_asset
                })
                
                rolling_corrs[col.split('_')[0]] = result_df
                
            except Exception as e:
                logger.error(f"Error processing rolling correlation: {str(e)}")
                logger.info(f"Rolling correlation index structure: {rolling_corr.index.names}")
                # Try alternative approach if the first one fails
                try:
                    # Reset the index and examine the structure
                    reset_corr = rolling_corr.reset_index()
                    logger.info(f"Columns after reset_index: {reset_corr.columns.tolist()}")
                    
                    # Filter using the correct column names from the actual DataFrame
                    level0_col = reset_corr.columns[0]  # First level of the original multi-index
                    level1_col = reset_corr.columns[1]  # Second level of the original multi-index
                    
                    filtered_corr = reset_corr[reset_corr[level0_col] == bitcoin_col]
                    
                    # Create the result DataFrame
                    result_df = pd.DataFrame({
                        f'btc_{col.split("_")[0]}_corr': filtered_corr[col]
                    }, index=filtered_corr[level1_col])
                    
                    rolling_corrs[col.split('_')[0]] = result_df
                    
                except Exception as e2:
                    logger.error(f"Alternative approach also failed: {str(e2)}")
                    # If all else fails, create an empty DataFrame
                    rolling_corrs[col.split('_')[0]] = pd.DataFrame()
    
    # Combine all rolling correlations
    if all(not df.empty for df in rolling_corrs.values()):
        combined = pd.concat(rolling_corrs.values(), axis=1)
        
        # Save to CSV
        combined.to_csv(f"data/rolling_{window}d_correlations.csv")
        logger.info(f"Calculated and saved {window}-day rolling correlations")
        
        return combined
    else:
        logger.error("Failed to calculate some rolling correlations, returning empty DataFrame")
        return pd.DataFrame()

def analyze_lead_lag_relationships(returns, max_lag=10):
    """
    Analyze lead-lag relationships between asset returns using multiple methods.
    
    This function implements several approaches to detect if one asset's movements tend to
    lead or lag another asset's movements:
    1. Cross-correlation analysis
    2. Granger causality tests
    3. Vector Autoregression (VAR) model impulse responses
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
        max_lag (int): Maximum number of lags to analyze (in days)
        
    Returns:
        dict: Dictionary containing lead-lag analysis results
    """
    logger.info(f"Analyzing lead-lag relationships with max lag of {max_lag} days")
    
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return') and not col.endswith('log_return')]
    ret_data = returns[return_columns].copy()
    
    # Check for stationarity (required for Granger causality)
    is_stationary = {}
    for col in return_columns:
        # Run Augmented Dickey-Fuller test
        result = adfuller(ret_data[col].dropna())
        is_stationary[col] = result[1] < 0.05  # p-value < 0.05 indicates stationarity
        logger.info(f"Stationarity test for {col}: p-value={result[1]:.4f}, {'stationary' if is_stationary[col] else 'non-stationary'}")
    
    # 1. Cross-correlation analysis
    logger.info("Performing cross-correlation analysis")
    ccf_results = {}
    
    for i, col1 in enumerate(return_columns):
        for col2 in return_columns[i+1:]:
            # Only analyze if both series are available
            if col1 in ret_data.columns and col2 in ret_data.columns:
                # Compute cross-correlation function
                series1 = ret_data[col1].dropna()
                series2 = ret_data[col2].dropna()
                
                # Align series
                aligned = pd.concat([series1, series2], axis=1).dropna()
                if len(aligned) > max_lag:
                    s1 = aligned[col1]
                    s2 = aligned[col2]
                    
                    # Calculate cross-correlation at different lags
                    cross_corr = {}
                    for lag in range(-max_lag, max_lag + 1):
                        if lag < 0:
                            # Negative lag: col1 lags behind col2 (col2 leads)
                            s1_shifted = s1.shift(-lag)
                            cross_corr[lag] = s1_shifted.corr(s2)
                        else:
                            # Positive lag: col1 leads col2
                            s2_shifted = s2.shift(lag)
                            cross_corr[lag] = s1.corr(s2_shifted)
                    
                    ccf_results[f"{col1}_vs_{col2}"] = cross_corr
                    
                    # Find the lag with the maximum correlation
                    max_corr_lag = max(cross_corr.items(), key=lambda x: abs(x[1]))
                    if max_corr_lag[0] < 0:
                        logger.info(f"{col2.split('_')[0]} leads {col1.split('_')[0]} by {abs(max_corr_lag[0])} days with correlation {max_corr_lag[1]:.3f}")
                    elif max_corr_lag[0] > 0:
                        logger.info(f"{col1.split('_')[0]} leads {col2.split('_')[0]} by {max_corr_lag[0]} days with correlation {max_corr_lag[1]:.3f}")
                    else:
                        logger.info(f"No lead-lag between {col1.split('_')[0]} and {col2.split('_')[0]}, max correlation at lag 0: {max_corr_lag[1]:.3f}")
    
    # 2. Granger causality tests
    logger.info("Performing Granger causality tests")
    granger_results = {}
    
    for i, col1 in enumerate(return_columns):
        for col2 in return_columns[i+1:]:
            # Only analyze if both series are stationary (required for Granger causality)
            if is_stationary.get(col1, False) and is_stationary.get(col2, False):
                # Prepare data
                data = ret_data[[col1, col2]].dropna()
                asset1 = col1.split('_')[0]
                asset2 = col2.split('_')[0]
                
                # Test if col1 Granger-causes col2
                try:
                    gc_1_to_2 = grangercausalitytests(data[[col2, col1]], maxlag=max_lag, verbose=False)
                    min_p_val_1_to_2 = min([gc_1_to_2[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag+1)])
                    best_lag_1_to_2 = [lag for lag in range(1, max_lag+1) if gc_1_to_2[lag][0]['ssr_chi2test'][1] == min_p_val_1_to_2][0]
                    
                    # Test if col2 Granger-causes col1
                    gc_2_to_1 = grangercausalitytests(data[[col1, col2]], maxlag=max_lag, verbose=False)
                    min_p_val_2_to_1 = min([gc_2_to_1[lag][0]['ssr_chi2test'][1] for lag in range(1, max_lag+1)])
                    best_lag_2_to_1 = [lag for lag in range(1, max_lag+1) if gc_2_to_1[lag][0]['ssr_chi2test'][1] == min_p_val_2_to_1][0]
                    
                    # Save results
                    granger_results[f"{asset1}_to_{asset2}"] = {
                        'p_value': min_p_val_1_to_2,
                        'lag': best_lag_1_to_2,
                        'significant': min_p_val_1_to_2 < 0.05
                    }
                    
                    granger_results[f"{asset2}_to_{asset1}"] = {
                        'p_value': min_p_val_2_to_1,
                        'lag': best_lag_2_to_1,
                        'significant': min_p_val_2_to_1 < 0.05
                    }
                    
                    # Log results
                    if min_p_val_1_to_2 < 0.05:
                        logger.info(f"Granger causality: {asset1} -> {asset2} is significant (p={min_p_val_1_to_2:.4f}) at lag {best_lag_1_to_2}")
                    else:
                        logger.info(f"Granger causality: {asset1} -> {asset2} is not significant (p={min_p_val_1_to_2:.4f})")
                        
                    if min_p_val_2_to_1 < 0.05:
                        logger.info(f"Granger causality: {asset2} -> {asset1} is significant (p={min_p_val_2_to_1:.4f}) at lag {best_lag_2_to_1}")
                    else:
                        logger.info(f"Granger causality: {asset2} -> {asset1} is not significant (p={min_p_val_2_to_1:.4f})")
                        
                except Exception as e:
                    logger.error(f"Error in Granger causality test for {col1} and {col2}: {str(e)}")
    
    # 3. VAR model for multivariate analysis
    logger.info("Fitting VAR model for multivariate lead-lag analysis")
    var_results = {}
    
    try:
        # Prepare data - use all returns for multivariate analysis
        var_data = ret_data[return_columns].dropna()
        
        # Determine optimal lag order
        from statsmodels.tsa.api import VAR
        model = VAR(var_data)
        # Get lag order from AIC
        lag_order = model.select_order(maxlags=max_lag).aic
        
        if lag_order > 0:
            # Fit VAR model with optimal lag
            var_fit = model.fit(lag_order)
            logger.info(f"VAR model fitted with lag order {lag_order}")
            
            # Perform Granger causality tests from VAR model
            asset_names = [col.split('_')[0] for col in return_columns]
            
            # Summarize VAR results
            var_results['lag_order'] = lag_order
            var_results['asset_names'] = asset_names
            
            # Extract coefficients to identify lead-lag relationships
            for i, asset1 in enumerate(asset_names):
                for j, asset2 in enumerate(asset_names):
                    if i != j:
                        # Check if asset1 impacts asset2
                        coef_sum = 0
                        for lag in range(1, lag_order+1):
                            coef_sum += abs(var_fit.coefs[lag-1][j, i])  # Effect of asset1 on asset2
                        
                        var_results[f"{asset1}_impact_on_{asset2}"] = coef_sum
                        
                        if coef_sum > 0.1:  # Arbitrary threshold
                            logger.info(f"VAR model: {asset1} has strong impact on {asset2} (coefficient sum: {coef_sum:.3f})")
        else:
            logger.warning("Optimal VAR lag order is 0, no lead-lag relationships found")
            
    except Exception as e:
        logger.error(f"Error in VAR analysis: {str(e)}")
    
    # Create summary of findings
    logger.info("Creating lead-lag relationship summary")
    
    lead_lag_summary = pd.DataFrame(columns=['Method', 'Relationship', 'Direction', 'Lag', 'Strength'])
    
    # Add cross-correlation findings
    row_index = 0
    for pair, corrs in ccf_results.items():
        assets = pair.split('_vs_')
        max_corr_lag = max(corrs.items(), key=lambda x: abs(x[1]))
        
        if max_corr_lag[0] != 0 and abs(max_corr_lag[1]) > 0.1:  # Only include non-zero lags with meaningful correlation
            if max_corr_lag[0] < 0:
                leading = assets[1].split('_')[0]
                lagging = assets[0].split('_')[0]
            else:
                leading = assets[0].split('_')[0]
                lagging = assets[1].split('_')[0]
                
            lead_lag_summary.loc[row_index] = [
                'Cross-correlation',
                f"{leading}-{lagging}",
                f"{leading} -> {lagging}",
                abs(max_corr_lag[0]),
                max_corr_lag[1]
            ]
            row_index += 1
    
    # Add Granger causality findings
    for relation, result in granger_results.items():
        if result['significant']:
            assets = relation.split('_to_')
            lead_lag_summary.loc[row_index] = [
                'Granger causality',
                f"{assets[0]}-{assets[1]}",
                f"{assets[0]} -> {assets[1]}",
                result['lag'],
                1 - result['p_value']  # Convert p-value to strength (1-p)
            ]
            row_index += 1
    
    # Add VAR model findings
    for key, value in var_results.items():
        if '_impact_on_' in key and value > 0.1:  # Only include strong impacts
            assets = key.split('_impact_on_')
            lead_lag_summary.loc[row_index] = [
                'VAR model',
                f"{assets[0]}-{assets[1]}",
                f"{assets[0]} -> {assets[1]}",
                var_results.get('lag_order', 'Unknown'),
                value
            ]
            row_index += 1
    
    # Save to CSV
    lead_lag_summary.to_csv("data/lead_lag_relationships.csv", index=False)
    
    # Plot cross-correlation functions
    for pair, corrs in ccf_results.items():
        assets = pair.split('_vs_')
        asset1 = assets[0].split('_')[0]
        asset2 = assets[1].split('_')[0]
        
        plt.figure(figsize=(12, 6))
        lags = sorted(corrs.keys())
        cors = [corrs[lag] for lag in lags]
        
        plt.bar(lags, cors, width=0.8)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f'Cross-Correlation: {asset1} vs {asset2}', fontsize=16)
        plt.xlabel('Lag (days)', fontsize=14)
        plt.ylabel('Correlation', fontsize=14)
        plt.xticks(lags[::2])  # Show every other lag on x-axis
        plt.grid(True, alpha=0.3)
        
        # Add directional explanations
        plt.figtext(0.15, 0.02, f"<-- {asset2} leads {asset1}", ha="center", fontsize=10)
        plt.figtext(0.85, 0.02, f"{asset1} leads {asset2} -->", ha="center", fontsize=10)
        
        # Add threshold reference lines
        plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.3)
        plt.axhline(y=-0.2, color='green', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/ccf_{asset1}_vs_{asset2}.png', dpi=300)
        plt.close()
    
    # Return complete analysis
    return {
        'cross_correlation': ccf_results,
        'granger_causality': granger_results,
        'var_model': var_results,
        'summary': lead_lag_summary
    }

def plot_rolling_correlations(rolling_corrs):
    """
    Plot rolling correlations over time.
    
    Args:
        rolling_corrs (pandas.DataFrame): DataFrame with rolling correlation data
    """
    plt.figure(figsize=(12, 8))
    
    # Plot each rolling correlation
    for col in rolling_corrs.columns:
        plt.plot(rolling_corrs.index, rolling_corrs[col], label=f'BTC-{col.split("_")[1]}')
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title('Rolling Correlation with Bitcoin', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Correlation Coefficient', fontsize=14)
    plt.legend(fontsize=12)
    plt.ylim(-1, 1)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/rolling_correlations.png', dpi=300)
    plt.close()
    
    logger.info("Created rolling correlation plot")

def analyze_correlation_significance(returns):
    """
    Analyze the statistical significance of correlations.
    
    Args:
        returns (pandas.DataFrame): DataFrame with return data
        
    Returns:
        pandas.DataFrame: DataFrame with correlation significance results
    """
    # Focus on daily percentage returns
    return_columns = [col for col in returns.columns if col.endswith('_return') and not col.endswith('log_return')]
    ret_data = returns[return_columns]
    
    # Create a DataFrame to store results
    results = pd.DataFrame(columns=[
        'Asset Pair', 
        'Pearson Correlation', 
        'Pearson p-value', 
        'Spearman Correlation', 
        'Spearman p-value'
    ])
    
    # Calculate correlations and p-values for Bitcoin with other assets
    bitcoin_col = 'bitcoin_return'
    row_idx = 0
    
    for col in return_columns:
        if col != bitcoin_col:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(ret_data[bitcoin_col], ret_data[col])
            
            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(ret_data[bitcoin_col], ret_data[col])
            
            # Add to results
            results.loc[row_idx] = [
                f'Bitcoin-{col.split("_")[0].capitalize()}',
                pearson_corr,
                pearson_p,
                spearman_corr,
                spearman_p
            ]
            
            row_idx += 1
    
    # Save results to CSV
    results.to_csv("data/correlation_significance.csv", index=False)
    logger.info("Analyzed and saved correlation significance results")
    
    return results

def run_correlation_analysis():
    """
    Run the complete correlation analysis.
    
    Returns:
        tuple: (correlations, rolling_corrs, significance) - The calculated correlation data
    """
    # Load processed data
    _, returns = load_processed_data()
    
    # Calculate correlations
    correlations = calculate_correlations(returns)
    
    # Plot correlation heatmaps
    plot_correlation_heatmaps(correlations)
    
    # Calculate rolling correlations (30-day window)
    rolling_corrs = calculate_rolling_correlations(returns, window=30)
    
    # Plot rolling correlations
    plot_rolling_correlations(rolling_corrs)
    
    # Analyze correlation significance
    significance = analyze_correlation_significance(returns)
    
    # Analyze lead-lag relationships with max lag of 10 days
    lead_lag = analyze_lead_lag_relationships(returns, max_lag=10)
    
    logger.info("Completed correlation analysis")
    
    return correlations, rolling_corrs, significance, lead_lag

if __name__ == "__main__":
    run_correlation_analysis() 