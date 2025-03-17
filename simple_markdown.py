"""
Simple examples of using pandas DataFrame.to_markdown() for generating markdown output.
"""

import pandas as pd
from IPython.display import Markdown, display

def display_as_markdown(df, title=None):
    """
    Display a DataFrame as markdown in a Jupyter notebook.
    
    Args:
        df (pandas.DataFrame): DataFrame to display
        title (str, optional): Title to display above the table
    """
    md = ""
    if title:
        md += f"## {title}\n\n"
    
    md += df.to_markdown()
    display(Markdown(md))

# Example usage in a notebook:
"""
# Simple example for your notebook

# For a single asset
bitcoin_rises = top_movements['bitcoin'][top_movements['bitcoin']['Type'] == 'Rise']
bitcoin_drops = top_movements['bitcoin'][top_movements['bitcoin']['Type'] == 'Drop']

# Format the columns you want to display
display_cols = ['Type', 'bitcoin_pct_change']
bitcoin_rises = bitcoin_rises[display_cols].copy()
bitcoin_drops = bitcoin_drops[display_cols].copy()

# Rename columns for better display
bitcoin_rises = bitcoin_rises.rename(columns={'bitcoin_pct_change': 'Bitcoin Change (%)'})
bitcoin_drops = bitcoin_drops.rename(columns={'bitcoin_pct_change': 'Bitcoin Change (%)'})

# Round values if needed
bitcoin_rises['Bitcoin Change (%)'] = bitcoin_rises['Bitcoin Change (%)'].round(2)
bitcoin_drops['Bitcoin Change (%)'] = bitcoin_drops['Bitcoin Change (%)'].round(2)

# Display as markdown
display_as_markdown(bitcoin_rises, "Top Bitcoin Rises")
display_as_markdown(bitcoin_drops, "Top Bitcoin Drops")

# Or directly use to_markdown() and display
from IPython.display import Markdown, display

# For all assets in a loop
for asset, df in top_movements.items():
    # Format the DataFrame
    display_df = df[['Type', f'{asset}_pct_change']].copy()
    display_df = display_df.rename(columns={f'{asset}_pct_change': f'{asset.capitalize()} Change (%)'})
    display_df[f'{asset.capitalize()} Change (%)'] = display_df[f'{asset.capitalize()} Change (%)'].round(2)
    
    # Create markdown
    md = f"## Top Movements for {asset.capitalize()}\n\n"
    md += display_df.to_markdown()
    
    # Display in notebook
    display(Markdown(md))
    
    # Or save to file
    with open(f"{asset}_movements.md", "w") as f:
        f.write(md)
""" 