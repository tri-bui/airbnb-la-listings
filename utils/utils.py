import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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
            coef = abs(df[feats[i]].corr(df[feats[j]]))
            if coef > threshold:
                corr.append([feats[i], feats[j], coef])
    
    return corr if len(corr) else None


def check_nb(row, tier, nb_list, nb_col='neighborhood', tier_col='price_tier'):
    
    '''
    Helper function to assign a price tier to a listing if its
    neighborhood is in the given list. This helper function is to be passed
    into a Pandas dataframe's apply method with an axis argument of 1.
    
    Args: row (row in Pandas dataframe)
          tier (integer) - tier value to assign to the listing
          nb_list (list[string]) - list of neighborhoods in a certain tier
          nb_col (string) - name of neighborhood column
          tier_col (string) - name of price tier column
          
    Returns: Price tier value (integer)
    '''
    
    return tier if row[nb_col] in nb_list else row[tier_col]


def onehot_encoder(df, var_list, num_lab=2, drop_og=True):
    
    '''
    One-hot encode a given list of categorical variables. Only the most 
    frequent labels will be encoded for each variable. The number of labels
    to encode is 2 by default.
    
    Args: df (Pandas dataframe)
          var_list (list[string]) - list of categorical variables to 
                                    one-hot encode
          num_lab (integer) - number of labels to encode for each variable
          drop_og (boolean) - whether to drop the original categorical
                              column after encoding it
                            
    Returns: Data with categorical variables one-hot encoded (Pandas dataframe)
    '''
    
    df = df.copy()
    
    for var in var_list:
        labels = df[var].value_counts().index[:num_lab]
        
        for lab in labels:
            df[var + '_' + lab] = df[var].apply(lambda x: 1 if x == lab else 0)
            
    if drop_og:
        df.drop(var_list, axis=1, inplace=True)
        
    return df


def log_transform(series, inverse=False):
    
    '''
    Log transform a series or inverse-log transform.
    
    Args: series (Pandas series)
          inverse (boolean) - whether to inverse-log transform
          
    Return: transformed series (Pandas series)
    '''
    
    series = series.copy()
    if inverse:
        return np.exp(series)
    return np.log(series)


def feat_coefs(feats, coefs, feat_name='feat', coef_name='coef', sort=True):
    
    '''
    Show the coefficient of each feature.
    
    Args: feats (list[string]) - list of feature names
          coefs (list[float]) - list of coefficients
          feat_name (string) - name of feature column in resulting dataframe
          coef_name (string) - name of coefficient column in resulting dataframe
          sort (boolean) - whether to sort the coefficients in descending order
          
    Returns: Features and their corresponding coefficient (Pandas dataframe)
    '''
    
    df = pd.DataFrame(zip(feats, coefs), columns=[feat_name, coef_name])
    if sort:
        df.sort_values(coef_name, ascending=False, inplace=True)
    return df


def make_pred(model, df_test, df_train=None, inverse_transform=True):
    
    '''
    Make predictions using a specified model for the test set and optionally
    the train set. If only the test set is passed in, an array containing 1 will 
    be returned for the train predictions.
    
    Args: model (Sklearn model instance) - model to predict with
          df_test (Pandas dataframe) - test set
          df_train (Pandas dataframe) - train set
          inverse_transform (boolean) - whether to inverse-log-transform the
                                        predictions
                                        
    Returns: predictions for the test and train sets
             (list[Pandas series]: length 2)
    '''
    
    pred_test = model.predict(df_test)
    pred_train = [1] if df_train is None else model.predict(df_train)
    
    if inverse_transform:
        pred_test = log_transform(pred_test, inverse=True)
        pred_train = log_transform(pred_train, inverse=True)
        
    return pred_test, pred_train


def score_rmse(true_test, pred_test, true_train=[1], pred_train=[1], log_true=True, log_pred=False):
    
    '''
    Calculate the root mean squared error for the test set and optionally the
    train set. This function calculates the RMSE for true values, so if any
    input set(s) are log-transformed, they will be inverse-log-transformed
    before the RMSE is calculated. If only the test sets are passed in, the
    RMSE for the train set will be returned as 0.
    
    Args: true_test (Pandas series) - true target variable from test set
          pred_test (Pandas series) - predicted target variable from test set
          true_train (Pandas series) - true target variable from train set
          pred_train (Pandas series) - predicted target variable from train set
          log_true (boolean) - whether the input true set(s) are log-transformed
          log_pred (boolean) - whether the input predicted set(s) are log-transformed
    
    Returns: root mean squared error for the test and train sets 
             (list[float]: length 2)
    '''
    
    if log_true:
        true_test = log_transform(true_test, inverse=True)
        true_train = log_transform(true_train, inverse=True)
        
    if log_pred:
        pred_test = log_transform(pred_test, inverse=True)
        pred_train = log_transform(pred_train, inverse=True)
        
    rmse_test = np.sqrt(mean_squared_error(true_test, pred_test))
    rmse_train = np.sqrt(mean_squared_error(true_train, pred_train))
    return rmse_test, rmse_train


def score_model(model, xtest, ytest, xtrain=None, ytrain=[1], toprint=True):
    
    '''
    Score a model with the R2 score and root mean squared error for the test set
    and optionally the train set. If only the test sets are passed in, both the
    R2 score and RMSE for the train set will be returned as 0.
    
    Args: model (Sklearn model instance) - model to score
          xtest (Pandas dataframe) - test set
          ytest (Pandas series) - test target
          xtrain (Pandas dataframe) - train set
          yterain (Pandas series) - train target
          
    Returns: R2 score and root mean squared error for the test and train sets 
             (list[float]: length 4)
    '''
    
    # R2 score
    
    ptest, ptrain = make_pred(model=model, df_test=xtest, df_train=xtrain)
    rmse_test, rmse_train = score_rmse(true_test=ytest, pred_test=ptest, true_train=ytrain, pred_train=ptrain)
    
    r2_train = 0
    if xtrain is not None:
        r2_train = model.score(xtrain, ytrain)
        if toprint:
            print('Train R2 Score:', r2_train)
            print('Train RMSE:', rmse_train)
            print()
    
    r2_test = model.score(xtest, ytest)
    if toprint:
        print('R2 Score:', r2_test)
        print('RMSE:', rmse_test)
    
    return r2_test, rmse_test, r2_train, rmse_train


