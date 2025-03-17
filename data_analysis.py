import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def analyze_distributions(target_df: pd.DataFrame, 
                         target_columns: List[str]) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Analyze distributions and calculate statistics for specified columns.
    
    Args:
        target_df: DataFrame containing the data
        target_columns: List of column names to analyze
        
    Returns:
        Tuple containing:
        - DataFrame with statistics for each column
        - Matplotlib figure with distribution plots
    """
    # Initialize results
    stats_df = pd.DataFrame()
    fig, axes = plt.subplots(len(target_columns), 2, 
                            figsize=(12, 3 * len(target_columns)))
    
    # If only one column, convert axes to 2D array for consistency
    if len(target_columns) == 1:
        axes = axes.reshape(1, -1)
    
    # Calculate statistics and create plots for each column
    for i, col in enumerate(target_columns):
        # Calculate statistics
        col_stats = target_df[col].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        col_stats['skewness'] = target_df[col].skew()
        col_stats['kurtosis'] = target_df[col].kurt()
        stats_df[col] = col_stats
        
        # Create distribution plots
        # Histogram with KDE
        sns.histplot(target_df[col], kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f'{col} Distribution')
        axes[i, 0].set_xlabel('Value')
        axes[i, 0].set_ylabel('Frequency')
        
        # Boxplot
        sns.boxplot(x=target_df[col], ax=axes[i, 1])
        axes[i, 1].set_title(f'{col} Boxplot')
        axes[i, 1].set_xlabel('Value')
    
    # Adjust layout
    plt.tight_layout()
    
    return stats_df, fig

# Example usage:
# stats, fig = analyze_distributions(returns, ['bitcoin_return', 'gold_return'])
# stats.to_csv('distribution_stats.csv')
# fig.savefig('distributions.png') 