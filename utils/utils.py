import numpy as np
import pandas as pd


def get_token():
    
    '''
    Retrieve the mapbox access token.
    
    Args: None
    
    Returns: Mapbox access token (string)
    '''
    
    with open('mapbox_access_token.txt') as f:
        return f.read()
    
    
def desc_byhost(df, col, host_col='by_superhost'):
    
    '''
    Calculate descriptive statistics for a dataframe's column for regular hosts and superhosts.
    
    Args: df (Pandas dataframe)
          col (string) - name of column to describe
          host_col (string) - name of the superhost column
    '''
    
    frame = df[df[host_col] == 0][[col]].describe()
    frame.columns = ['regular_hosts']
    frame['superhosts'] = df.loc[df[host_col] == 1, col].describe()
    return frame