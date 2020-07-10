import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    
    
def hist_with_hue(df, col, bins, hue_col='by_superhost', height=5, aspect=2, 
                  alpha=0.75):
    
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
    labs = ['0 - 10', '10 - 20', '20 - 30', '30 - 40', '40 - 50', '50 - 60', '60 - 70', '70 - 80', '80 - 90', '90 - 100']
    ax.set_xticklabels(labs, rotation=45, ha='right')
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Label bars with percentages if greater than 0
    for i in range(0, 10):
        p = pct.iloc[i]
        if p > 0:
            ax.text(i - 0.1, pct.iloc[i] + 2, f'{p}%', ha='center')
            
            
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
        
        
def corr_heatmap(df, var_list, show_spearman=False, palette=None, center=None):
    
    '''
    Create a heatmap for Pearson correlation and Spearman correlation.
    
    Args: df (Pandas dataframe)
          var_list (list[string]) - list of variables to plot
          show_spearman (boolean) - whether to show the Spearman correlation in
                                    a second subplot
          
    Returns: correlation table (Pandas dataframe)
    '''
    
    if show_spearman:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        # Pearson correlation
        sns.heatmap(df[var_list].corr(), annot=True, cmap=palette, center=center, ax=ax[0])
        ax[0].set_title('Pearson correlation')
        ax[0].set_xticklabels(var_list, rotation=45, ha='right')

        # Spearman correlation
        sns.heatmap(df[var_list].corr(method='spearman'), annot=True, cmap=palette, center=center, ax=ax[1])
        ax[1].set_title('Spearman correlation')
        ax[1].set_xticklabels(var_list, rotation=45, ha='right')
        
    else:
        sns.heatmap(df[var_list].corr(), cmap=palette, center=center)
        plt.title('Pearson correlation')
        plt.xticks(rotation=45, ha='right')
        
    plt.show()
    return df[var_list].corr()
    
    
def wordcloud(text, bg_color='white', stopwords=None):
    
    '''
    Generate a word cloud image of some text.
    
    Args: text (string) - text to generate
          bg_color (string) - background color of the image
          stopwords (list[string]) - list of words to ignore when generating the 
                                     word cloud
                                     
    Returns: None
    '''
    
    cloud = WordCloud(stopwords=stopwords, background_color=bg_color).generate(text)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    

def get_langeo(path='data/la-county-neighborhoods-current.geojson'):
    
    '''
    Load the geojson data of Los Angeles neighborhoods
    
    Args: path (string) - path to the json file of the geographic data
    
    Returns: Los Angeles neighborhood geographic data (json)
    '''
    
    with open(path) as geo:
        return json.load(geo)
    
    
def choropleth_map(df, color_col, loc_col='Neighborhood', center_lat=33.95, center_lon=-118.35, zoom=9, height=600, color_scale='Viridis', colorbar_range=None):
    
    '''
    Plot a choropleth map of the neighborhoods in LA. The neighborhoods are colored 
    based on the passed in column.
    
    Args: df (Pandas dataframe)
          color_col (string) - name of column that determines the color of each 
                               neighborhood on the map
          colorbar_range (tuple(integer): length 2) - lower and upper limits for the 
                                                      colorbar
          loc_col (string) - name of neighborhood column
          center_lat (float) - latitude at the center of the map
          center_lon (float) - longitude at the center of the map
          zoom (integer or float) - zoom ratio of the map
          height (integer) - height of the map
          color_scale (string) - name of the color palette to use
          
    Returns: Mapbox object with map (use the .show() method to show the map)
    '''
    
    # Geojson for LA neighborhoods
    la = get_langeo()
    return px.choropleth_mapbox(df, geojson=la, featureidkey='properties.name', 
                                color=color_col, locations=loc_col, 
                                center={'lat': center_lat, 'lon': center_lon},
                                zoom=zoom, height=height,
                                color_continuous_scale=color_scale,
                                range_color=colorbar_range)
