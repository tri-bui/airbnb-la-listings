import numpy as np
import pandas as pd


def get_token(path='mapbox_access_token.txt'):
    
    '''
    Retrieve the mapbox access token.
    
    Args: path (string) - path to the text file with the Mapbox access token
    
    Returns: Mapbox access token (string)
    '''
    
    with open(path) as key:
        return key.read()
    
    
def desc_byhost(df, col, host_col='by_superhost'):
    
    '''
    Calculate descriptive statistics for a dataframe's column for regular hosts 
    and superhosts.
    
    Args: df (Pandas dataframe)
          col (string) - name of column to describe
          host_col (string) - name of the superhost column
          
    Returns: descriptive statistics of the selected column separated by the host 
             column (Pandas dataframe)
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


def agg_to_2cols(df, agg_col, agg_col_name, groupby_col='neighborhood', 
                 groupby_col_name='Neighborhood', agg_by_mean=True, round_to=2):
    
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


def dup_feats(df):
    
    '''
    Find duplicated features in a dataframe.
    
    Args: df (Pandas dataframe)
    
    Returns: Pairs of duplicated features (list[list[string]: length 2])
    '''
    
    cols = df.columns.tolist()
    dup = []
    
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                dup.append([cols[i], cols[j]])
                
    return dup if len(dup) else None


def const_feats(df, threshold=0.05):
    
    '''
    Find constant and quasi-constant features. A feature is considered 
    quasi-constant if its variance is less than the defined threshold.
    
    Args: df (Pandas dataframe)
          threshold (float) - variance threshold to consider a feature 
                              quasi-constant
                              
    Returns: List of constant and quasi-constant features (list[string])
    '''
    
    const = df.var()[df.var() < threshold].index.tolist()
    return const if len(const) else None


def const_cat_feats(df, threshold=0.95):
    
    '''
    Find constant and quasi-constant categorical features. A feature is 
    considered quasi-constant if any of its values has a relative count 
    greater than the defined threshold.
    
    Args: df (Pandas dataframe)
          threshold (float) - relative count threshold to consider a feature 
                              quasi-constant
                              
    Returns: List of constant and quasi-constant features (list[string])
    '''
    
    cols = df.dtypes[df.dtypes == 'object'].index.tolist()
    const = []
    
    for col in cols:
        value_pct = df[col].value_counts() / df.shape[0]
        if value_pct[0] > threshold:
            const.append(col)
            
    return const if len(const) else None


def corr_feats(df, threshold=0.5):
    
    '''
    Find correlated features.
    
    Args: df (Pandas dataframe)
          threshold (float) - threshold for correlation coefficient
          
    Returns: Pairs of correlated features (list[list[string, string, float]])
    '''
    
    feats = df.columns.tolist()
    corr = []
    
    for i in range(len(feats) - 1):
        for j in range(i + 1, len(feats)):
            coef = df[feats[i]].corr(df[feats[j]])
            if coef > threshold:
                corr.append([feats[i], feats[j], coef])
    
    return corr if len(corr) else None