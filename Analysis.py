import gmplot
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns; sns.set(color_codes=True)
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statistics import stdev, mean, median

df = pd.read_csv('schools_preprocessed.csv')

ratings = df['2017 RATING'].values

# --------------------------------------------------------------------
# -----------------------------Prediction-----------------------------
# --------------------------------------------------------------------

# One hot encoding
ordinal_columns = df[['INCOME_BINS', '2013 RATING', '2014 RATING',
                      '2015 RATING', '2016 RATING', '2017 RATING']]
categorical_columns = ['MUN', 'SCHOOL_TYPE', 'SCHOOL_BOARD']
categorical_columns = pd.get_dummies(df[categorical_columns])

ohe_df = pd.concat([categorical_columns, ordinal_columns], axis=1)


def LOOCV(ohe_df, use_previous_years=False):
    '''
    Leave one out cross validation to check performance on
    multiple regression models.
    '''
    models = {
     'RFR': RandomForestRegressor(n_estimators=50, random_state=0),
     'GBR': GradientBoostingRegressor(max_depth=1, random_state=0),
     'LIR': LinearRegression(),
     'SVR': SVR(kernel='linear')
    }
    if use_previous_years is False:
        ordinal_columns = df[['INCOME_BINS', '2017 RATING']]
        ohe_df = pd.concat([categorical_columns, ordinal_columns], axis=1)

    df_x = ohe_df.iloc[:, :-1]
    df_y = ohe_df.iloc[:, -1]

    scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']
    loo = LeaveOneOut.get_n_splits(df_x, df_y)

    for name, model in models.items():
        scores = cross_validate(model, df_x, df_y, cv=loo,
                                scoring=scoring)
        rmse = (-1*mean(scores['test_neg_mean_squared_error']))**0.5
        mae = -1*mean(scores['test_neg_mean_absolute_error'])
        print(f'{name} RMSE: {rmse: 0.4f}, MAE: {mae: 0.4f}')


def baseline_metrics(ohe_df):
    '''
    Establishes RMSE and MAE if mean of dataset is predicted for
    all inputs.
    '''
    df_x = ohe_df.iloc[:, :-1]
    df_y = ohe_df.iloc[:, -1]
    y_predicted = mean(df_y)
    rmse = (mean((y_predicted - df_y)**2))**0.5
    mae = mean(abs(y_predicted - df_y))

    print(f'RMSE: {rmse: 0.4f}, MAE: {mae: 0.4f}')


# --------------------------------------------------------------------
# ---------------------------Data Analysis----------------------------
# --------------------------------------------------------------------


def correlation_matrix(ohe_df):
    '''
    Correlation matrix is sorted by most correlated
    (absolute value so it does not matter if correlation is
    negative or positive), then put into a seaborn heatmap. Columns
    are renamed for size convenience.
    '''
    ohe_df = ohe_df.rename(columns={'INCOME_BINS': 'Income',
                                    'MUN_Etobicoke': 'Etobicoke',
                                    'MUN_North York': 'North York',
                                    'MUN_East York': 'East York',
                                    'MUN_former Toronto': 'Former Toronto',
                                    'MUN_York': 'York',
                                    'MUN_Scarborough': 'Scarborough',
                                    'SCHOOL_BOARD_ES': 'Catholic',
                                    'SCHOOL_BOARD_EP': 'Public',
                                    'SCHOOL_BOARD_PR': 'Private',
                                    'SCHOOL_TYPE_ELEMENTARY': 'Elementary',
                                    'SCHOOL_TYPE_SECONDARY': 'Secondary',
                                    '2013 RATING': '2013 Rating',
                                    '2014 RATING': '2014 Rating',
                                    '2015 RATING': '2015 Rating',
                                    '2016 RATING': '2016 Rating',
                                    '2017 RATING': '2017 Rating'})
    corr_sorted = abs(ohe_df.corr()['2017 Rating']).sort_values()
    ohe_df = ohe_df[list(corr_sorted.index)]
    corr = round(ohe_df.corr(), 2)

    fig = sns.heatmap(corr, annot=True, cmap='Blues',
                      xticklabels=corr.columns.values,
                      yticklabels=corr.columns.values,
                      cbar=False)
    plt.xticks(rotation=0)
    fig.xaxis.set_tick_params(labelsize=8)
    fig.yaxis.set_tick_params(labelsize=8)

    plt.show()


def one_way_anova():
    '''
    Each categorical column is separated into its categories. Next,
    ratings for each category is extracted then put into a list.
    These lists are then compared with each other for ANOVA. This
    process is repeated for each column.
    '''
    columns_to_analyze = ['MUN', 'SCHOOL_BOARD', 'SCHOOL_TYPE']
    for column in columns_to_analyze:
        grouped_dfs = []
        for group in df.groupby(column).groups:
            grouped_df = df.groupby(column).get_group(group)
            grouped_df = grouped_df.reset_index()['2017 RATING']
            grouped_dfs.append(list(grouped_df.dropna()))
        F, p = stats.f_oneway(*grouped_dfs)
        print(f'{column}: {p: 0.2e}')


# --------------------------------------------------------------------
# -------------------------Data Visualization-------------------------
# --------------------------------------------------------------------


def boxplot_incomes():
    incomes_binned = df['INCOME_BINS'].values
    labels = ['0', '20', '40', '60', '80', '100',
              '120', '140', '160', '180', '200+']
    fig = sns.boxplot(x=incomes_binned, y=ratings)
    plt.title('Distribution of 2017 Ratings by Median Neighbourhood Income')
    fig.set(xlabel='Neighbourhood Median Income (Thousands)',
            ylabel='School Rating in 2017')
    fig.set_xticklabels(labels)
    fig.xaxis.set_tick_params(labelsize=10)
    plt.show()


def boxplot_school_type():
    fig = sns.boxplot(x=df['SCHOOL_TYPE'], y=df['2017 RATING'])
    plt.title('Distribution of 2017 Ratings by School Type')
    plt.xlabel('School Type')
    plt.ylabel('School Rating Out of 10')
    labels = ['Secondary', 'Elementary']
    fig.set_xticklabels(labels)
    plt.show()


def boxplot_school_boards():
    fig = sns.boxplot(x=df['SCHOOL_BOARD'], y=ratings)
    plt.title('Distribution of 2017 Ratings in Each Toronto School Board')
    plt.xlabel('School Board')
    plt.ylabel('School Rating Out of 10')
    labels = ['Catholic', 'Public', 'Private']
    fig.set_xticklabels(labels)
    plt.show()


def boxplot_municipalities():
    fig = sns.boxplot(x=df['MUN'], y=ratings)
    plt.title('Distribution of 2017 Ratings in Each Municipality')
    plt.xlabel('Municipality')
    plt.ylabel('School Rating Out of 10')
    labels = ['North York', 'Scarborough', 'Former Toronto', 'Etobicoke',
              'East York', 'York']
    fig.set_xticklabels(labels)
    fig.xaxis.set_tick_params(labelsize=8)
    plt.show()


def distplot_2017_ratings():
    sns.distplot(df['2017 RATING'])
    plt.title('Distribution of 2017 Ratings in Toronto')
    plt.xlabel('School Rating Out of 10')
    plt.ylabel('Proportion of Schools')
    plt.show()

    print(mean(ratings), stdev(ratings))


def scatter_2017_ratings():
    latitudes = df['LATITUDE'].values
    longitudes = df['LONGITUDE'].values
    minlat = min(latitudes)
    maxlat = max(latitudes)
    minlong = min(longitudes)
    maxlong = max(longitudes)

    fig = plt.figure()
    sc = plt.scatter(longitudes, latitudes, c=ratings, cmap='hot')

    plt.colorbar(sc)
    plt.xlim(minlong, maxlong)
    plt.ylim(minlat, maxlat)
    plt.show()


def plot_on_gmaps(latitudes, longitudes):
    '''
    Plotting on Google Mapls with gmplot
    gmplot might throw an IndexError, it is an open issue on
    the GitHub page for gmplot.
    '''
    gmap = gmplot.GoogleMapPlotter.from_geocode("Toronto, Canada")
    gmap.scatter(latitudes, longitudes, weights=df['INCOME'].values,
                 cmap='hot', size=500, marker=False)
    gmap.draw("map2.html")