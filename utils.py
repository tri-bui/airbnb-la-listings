import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def get_token():
    
    '''
    Retrieve the mapbox access token.
    
    Args: None
    
    Returns: Mapbox access token (string)
    '''
    
    with open('mapbox_access_token.txt') as f:
        return f.read()
    
    
def hist_pct(df, ax, col='score_rating', filter_col='by_superhost', filter_val=None,  bins=range(0, 110, 10), title=None, xlab=None, ylab=None):
    
    '''
    Plot a histogram of the selected column in a dataframe on a relative scale (percentage).
    There are options to filter on another column and to bin values into intervals.
    
    Args: df (Pandas dataframe)
          col (string) - name of column to plot
          filter_col (string) - name of column to filter on
          filter_val - value to filter for on `filter_col`
          bins (integer or list/generator) - number of bins or list/generator of bin edges
          ax (Matplotlib axis object) - axis to plot on
          title (string) - plot title
          xlab (string) - label for x-axis
          ylab (string) - label for y-axis
        
    Returns: None
    '''
    
    if filter_val is not None:
        df = df[df[filter_col] == filter_val].copy()
    df = df[col].dropna()
    
    if bins:
        df = pd.cut(df, bins=bins)
        
    ct = df.value_counts().sort_index() # Absolute count
    pct = np.round(100 * ct / df.shape[0], 1) # Relative count
    pct.plot(kind='bar', ax=ax)
    
    # Plot settings
    ax.set_xticklabels(pct.index, rotation=45, ha='right')
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Label bars with percentages if greater than 0
    for i in range(0, 10):
        p = pct.iloc[i]
        if p > 0:
            ax.text(i, pct.iloc[i] + 2, f'{p}%', ha='center')