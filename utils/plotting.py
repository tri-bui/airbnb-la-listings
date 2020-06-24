import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    
    
def hist_with_hue(df, col, bins, hue_col='by_superhost', height=5, aspect=2, alpha=0.75):
    
    '''
    Plot a histogram of a dataframe's column using another column as the color hue.
    
    Args: df (Pandas dataframe)
          col (string) - name of column to plot
          bins (integer or list/generator) - number of bins or list/generator of 
                                             bin edges
          hue_col (string) - name of column to use as hue
          height (integer or float) - plot height
          aspect (integer or float) - width-to-height ratio for plot
          alpha (float) - opacity of histogram bars
          
    Returns: None
    '''
    
    grid = sns.FacetGrid(data=df, hue=hue_col, height=height, aspect=aspect)
    grid.map(plt.hist, col, bins=bins, alpha=alpha)
    grid.add_legend()
    
    
def hist_pct(df, ax, col='score_rtg', filter_col='by_superhost', filter_val=None,  bins=range(0, 110, 10), title=None, xlab=None, ylab=None):
    
    '''
    Plot a histogram of the selected column in a dataframe on a relative scale 
    (percentage).
    There are options to filter on another column and to bin values into 
    intervals.
    
    Args: df (Pandas dataframe)
          col (string) - name of column to plot
          filter_col (string) - name of column to filter on
          filter_val - value to filter for on `filter_col`
          bins (integer or list/generator) - number of bins or list/generator of 
                                             bin edges
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
            
            
def labeled_barplot(df, x, y, title, upper, col_loc=1, horizontal=True):
    
    '''
    Plot a barplot labeled with bar values.
    
    Args: df (Pandas dataframe)
          x (string) - name of column to plot on the x-axis
          y (string) - name of column to plot on the y-axis
          title (string) - plot title
          upper (integer) - upper limit for x-axis if barplot is horizontal or 
                            y-axis if barplot is vertical
          col_loc (integer) - column index of the numeric column being plotted
          horizontal (boolean) - orientation of the barplot
          
    Returns: None
    '''
    
    sns.barplot(data=df, x=x, y=y)
    plt.title(title, fontsize=16)
    
    if horizontal:
        plt.xlim(0, upper)
        for i in range(df.shape[0]):
            n = df.iloc[i, col_loc]
            plt.text(n + 1, i, f'{n}', va='center')
            
    else:
        plt.ylim(0, upper)
        plt.xticks(rotation=15)
        for i in range(df.shape[0]):
            n = df.iloc[i, col_loc]
            plt.text(i, n + 1, f'{n}', ha='center')
            
            
def scatter_subplots(df, var_list, start_idx=2, figsize=(16, 16), subplot_rows=3, subplot_cols=3):
    
    '''
    Create a scatterplot of the first variable in a variable list against every 
    other variable in the list, starting with the start index, in subplots.
    
    Args: df (Pandas dataframe)
          var_list (list[string]) - list of variables to plot
          start_idx (integer) - list index of first variable to plot against
          figsize (tuple(integer): length 2) - figure size
          subplot_rows (integer) - number of rows for subplots
          subplot_cols (integer) - number of columns for subplots
          
    Returns: None
    '''
    
    plt.figure(figsize=figsize)
    for i in range(start_idx, len(var_list)):
        plt.subplot(subplot_rows, subplot_cols, i - start_idx + 1)
        plt.scatter(data=df, x=var_list[0], y=var_list[i])
        plt.xlabel(var_list[0])
        plt.ylabel(var_list[i])
        
        
def corr_heatmap(df, var_list):
    
    '''
    Create a heatmap for Pearson correlation and Spearman correlation.
    
    Args: df (Pandas dataframe)
          var_list (list[string]) - list of variables to plot
    '''
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Pearson correlation
    sns.heatmap(df[var_list].corr(), annot=True, ax=ax[0])
    ax[0].set_title('Pearson correlation')
    ax[0].set_xticklabels(var_list, rotation=45, ha='right')

    # Spearman correlation
    sns.heatmap(df[var_list].corr(method='spearman'), annot=True, ax=ax[1])
    ax[1].set_title('Spearman correlation')
    ax[1].set_xticklabels(var_list, rotation=45, ha='right')
    
    
def wordcloud(text, bg_color='white', stopwords=None):
    
    '''
    Generate a word cloud image of some text.
    
    Args: text (string) - text to generate
          bg_color (string) - background color of the image
          stopwords (list[string]) - list of words to ignore when generating the 
                                     word cloud
    '''
    
    cloud = WordCloud(stopwords=stopwords, background_color=bg_color).generate(text)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    
    
