# Navigating AirBnB in Los Angeles
##### This is an analysis of AirBnB listings in Los Angeles County, the biggest county in the U.S and one of the world's top destinations. The purpose of this project is to help guide consumers in navigating AirBnB in Los Angeles. I do this by giving an overview of the listings from the consumer's standpoint, in order to give them a better idea of how to narrow their search. In my analysis, I answer the following questions:

1. Is there a difference with listings by superhosts?
2. Are higher ratings associated with higher costs?
3. What are reviewers saying?
4. Where are the best-rated listings?
5. Where can we find low-cost versus high-end listings?
6. Can we predict the price of a listing?
7. What factors have the most impact on price?

#### If in the future, you are ever looking for a place to stay in LA, then you are the target audience! Read about it on my [blog post](https://medium.com/@buitri/essential-guide-to-navigating-airbnb-in-los-angeles-cf8a932501b7).

## Data
The data used in this project was compiled on May 8, 2020, and can be found [here](http://insideairbnb.com/get-the-data.html) under "Los Angeles, California, United States". The main dataset used was `listings.csv`, which contains 37,048 AirBnb listings in Los Angeles County and 106 features for each listing. `reviews.csv` was also used, which contains 1,304,140 reviews that users left for these listings.

## Files
- `1-preprocessing.ipynb` - cleaning and preprocessing of the raw data
- `2-eda.ipynb` - main analysis with visualizations (questions 1 - 5)
- `3-modeling.ipynb` - feature engineering, feature selection, modeling with linear regression, K-neighbors, and random forest (questions 6 & 7)
- `utils/plotting.py` - user-defined functions for plotting
- `utils/utils.py` - user-defined functions for all other utilities

## To run the notebooks, the project directory must be set up

#### Setup
Note: The `notebooks/` directory was created to keep the repository clean.

1. Move all content in the `notebooks/` directory to the main directory.
2. Under "Los Angeles, California, United States" at the [data source](http://insideairbnb.com/get-the-data.html), download `listings.csv` and `reviews.csv` to a directory called `data/` and add this directory to the main directory.
3. Sign up for free at [MapBox](https://www.mapbox.com/) to get an access token. Store the access token string in a file named `mapbox_access_token.txt` and add it to the main directory.
4. Download the GeoJSON file from [here](http://boundaries.latimes.com/set/la-county-neighborhoods-current/) and add it to the `data/` directory.
5. \*Optional\* `visualizations.ipynb` was created to generate the figures for the Medium article, so it is not part of the project. However, if you want to run it, create a directory called `figs/` in the main directory.

Steps (3) and (4) are required for the Plotly-Mapbox plots to work. After the setup, the directory should look like below.

#### Structure
<pre>
airbnb_la_listings/
|-- data/
    |-- la-county-neighborhoods-current.geojson
    |-- listings.csv
    |-- reviews.csv
|-- figs/ *
|-- utils/
    |-- plotting.py
    |-- utils.py
|-- 1-preprocessing.ipynb
|-- 2-eda.ipynb
|-- 3-modeling.ipynb
|-- mapbox_access_token.txt
|-- visualizations.ipynb *
</pre>

#### Requirements
- Python 3
- Packages: Jupyter notebooks/lab, Uszipcode, Scipy, Numpy, Pandas, Matplotlib, Seaborn, Plotly, Wordcloud, Scikit-learn

## License
This repository is licensed under a [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).