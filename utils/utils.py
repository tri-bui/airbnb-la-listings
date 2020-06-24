import numpy as np
import pandas as pd
import json


def get_token(path='mapbox_access_token.txt'):
    
    '''
    Retrieve the mapbox access token.
    
    Args: path (string) - path to the text file with the Mapbox access token
    
    Returns: Mapbox access token (string)
    '''
    
    with open(path) as key:
        return key.read()
    
    
def get_langeo(path='data/la-county-neighborhoods-current.geojson'):
    
    '''
    Load the geojson data of Los Angeles neighborhoods
    
    Args: path (string) - path to the json file of the geographic data
    
    Returns: Los Angeles neighborhood geographic data (json)
    '''
    
    with open(path) as geo:
        return json.load(geo)
    
    
def desc_byhost(df, col, host_col='by_superhost'):
    
    '''
    Calculate descriptive statistics for a dataframe's column for regular hosts 
    and superhosts.
    
    Args: df (Pandas dataframe)
          col (string) - name of column to describe
          host_col (string) - name of the superhost column
          
    Returns: descriptive statistics of the selected column separated by the host column (Pandas dataframe)
    '''
    
    frame = df[df[host_col] == 0][[col]].describe()
    frame.columns = ['regular_hosts']
    frame['superhosts'] = df.loc[df[host_col] == 1, col].describe()
    return frame


def sample_byinterval(df, col, lower, upper, step, size=100):
    
    '''
    Downsample a dataframe's column in intervals.
    
    Args: df (Pandas dataframe)
          col (string) - name of column to sample
          lower (integer) - lower bound of the first interval
          upper (integer) - upper bound of the last interval
          step (integer) - size of each interval
          size (integer) - size of each sample
    
    Returns: downsampled data (Pandas dataframe)
    '''
    
    samples = [df[df[col] < lower].copy()]
    for r in range(lower, upper, step):
        samp = df[(df[col] > r) & (df[col] <= r + step)].copy().sample(size, random_state=0)
        samples.append(samp)
    return pd.concat(samples)


def agg_to_2cols(df, agg_col, agg_col_name, groupby_col='neighborhood', groupby_col_name='Neighborhood', agg_by_mean=True, round_to=2):
    
    '''
    Aggregate a dataframe into 2 columns - grouping by 1 and aggregating the other 
    by either the mean or count.
    
    Args: df (Pandas dataframe)
          agg_col (string) - name of column to aggregate
          agg_col_name (string) - name of aggregated column in the resulting 
                                  dataframe
          groupby_col (string) - name of column to group by
          groupby_col_name (string) - name of grouped column in the resulting 
                                      dataframe
          agg_by_mean (boolean) - whether to aggregate using the mean, else use 
                                  the count
          round_to (integer) - number of digits to round to if aggregating using 
                               the mean
                                  
    Returns: Aggregated data in 2 columns (pandas dataframe)
    '''
    
    if agg_by_mean:
        frame = df.groupby(groupby_col)[[agg_col]].mean().round(round_to).sort_values(agg_col, ascending=False).reset_index()
    else:
        frame = df.groupby(groupby_col)[[agg_col]].count().sort_values(agg_col, ascending=False).reset_index()
        
    frame.columns = [groupby_col_name, agg_col_name]
    return frame