import pandas as pd
import numpy as np
import re
import math
import os
from geopy.distance import vincenty


def proper_round(number, d_places=1):
    'Used to avoid floating point errors'
    return math.floor((number+0.5/(10**d_places))*10)/10


def fill_missing_values(ratings_with_na):
    '''
    This function fills missing values with the mean of the ratings
    that are not missing. If all ratings (2013-2016) are n/a, the function
    returns the original list inputted.
    '''

    ratings_with_na = [np.nan if rating == 'n/a'
                       else float(rating)
                       for rating in ratings_with_na]
    if all([rating in [np.nan] for rating in ratings_with_na[0:-1]]):
        return ratings_with_na

    mean_rating = proper_round(np.nanmean(ratings_with_na[0:-1]))

    filled_missing_values = []
    for rating in ratings_with_na:
        if rating in [np.nan]:
            filled_missing_values.append(mean_rating)
        else:
            filled_missing_values.append(rating)

    return filled_missing_values


def find_bins(df, column, num_bins, maxbin):
    '''
    Divide the values dataframe column into intervals.
    If the value is higher than the highest bin, it will
    go into the highest bin.
    '''
    intervals = maxbin / num_bins
    values = []
    for _, row in df.iterrows():
        bin = row[column]/intervals
        bin = min(bin, num_bins)
        bin = (math.floor(bin))
        values.append(bin)

    df[column+'_BINS'] = values

    return df


def find_incomes_of_areas(original_df, income_table):
    '''
    This function is to minimize the distance between the latitudes
    and longitudes in one dataframe with the latitudes and longitudes
    of another. Used to find the dissemination area each school belongs to.
    Income table will have a list of incomes for each dissemination area
    along with the central latitude and longitude for that area.
    '''
    lats = original_df['LATITUDE'].values
    longs = original_df['LONGITUDE'].values
    school_coords = list(zip(lats, longs))

    income_lats = income_table['latitude'].values
    income_longs = income_table['longitude'].values
    income_coords = list(zip(income_lats, income_longs))
    incomes = []
    ids = []

    for school_coord in school_coords:
        minimum_distance = 9999
        for i, income_coord in enumerate(income_coords):
            distance = vincenty(school_coord, income_coord)
            if distance < minimum_distance:
                minimum_distance = distance
                row = income_table.index[i]
        income_for_school_area = float(income_table.loc[row, 'Median_Household_Income'])
        incomes.append(income_for_school_area)

    return incomes


def DA_to_lat_long(statscan_database, cnsmpr_database):
    '''
    Used to merge the database of dissemination areas given by
    CensusMapper.ca with the database of the central latitudes and
    longitudes of every dissemination area given by Statscan.
    statscan_df = statcan database
    cnsmpr_df = CensusMapper database
    '''
    statscan_df = pd.read_csv(statscan_database)
    statscan_df = statscan_df.rename(columns={'DAuid/ADidu': 'DA_ID',
                                            'DArplat/ADlat': 'latitude',
                                            'DArplong/ADlong': 'longitude'})
    statscan_df = statscan_df.drop_duplicates(subset=['DA_ID'])

    cnsmpr_df = pd.read_csv(cnsmpr_database)
    cnsmpr_df = cnsmpr_df.rename(columns={'GeoUID': 'DA_ID',
                'v_CA16_2397: Median total income of households in 2015 ($)':
                'Median_Household_Income'})
    cnsmpr_df = cnsmpr_df[['DA_ID', 'Median_Household_Income']]
    cnsmpr_df = cnsmpr_df[pd.to_numeric(cnsmpr_df['Median_Household_Income'],
                                        errors='coerce').notnull()]

    result = pd.merge(cnsmpr_df, statscan_df, how='inner', on=['DA_ID'])
    return result


def preprocess(text_files, school_database, cnsmpr_database,
               statscan_database, data_folder_name):
    '''
    Main preprocessing function.
    Reads the school locations database, incomes in each DA,
    ratings in each school text file, and combines them.

    Step 1: Drop unnecessary columns.
    Step 2: Merge Statscan and CensusMapper databases
    Step 3: Read text files using REGEX and extract ratings.
    Step 4: Fill missing ratings with mean rating of the school
    Step 5: Merge ratings into original database.
    Step 6: If all ratings are the exact same for multiple rows
            (unlikely unless duplicate),
            drop the duplicates.
    Step 7: Drop if school board is FP (there is only one).
    Step 8: Find the dissemination area income for each school
    Step 9: Put annual incomes into bins in intervals of 20000
    Step 10: Remove all rows that are still n/a
    '''
    database = os.path.join(data_folder_name, school_database)
    cnsmpr_csv = os.path.join(data_folder_name, cnsmpr_database)
    statscan_csv = os.path.join(data_folder_name, statscan_database)
    files = map(lambda file: os.path.join(data_folder_name, file), text_files)
    output_csv = os.path.join(data_folder_name, 'schools_preprocessed.csv')

    df = pd.read_excel(database)
    df = df.rename(columns={'SCL_TP': 'SCHOOL_BOARD'})
    df = df.drop(columns=['SCL_LVL', 'ADD_PT_ID', 'ADD_NUM', 'LN_NAM_FUL',
                          'ADD_FULL', 'BRD_NAME', 'SCL_TP_DSC', 'CITY',
                          'GEN_USE_CD', 'CNTL_ID', 'LO_NUM', 'LO_NUM_SUF',
                          'HI_NUM', 'HI_NUM_SUF', 'LN_NAM_ID', 'X', 'Y',
                          'OBJECTID', 'RID', 'POSTAL_CD'])  # Step 1

    income_df = DA_to_lat_long(cnsmpr_csv, statscan_csv)  # Step 2

    years = [2013, 2014, 2015, 2016, 2017]
    school_is_in_toronto = False
    for file in files:
        with open(file) as f:
            for line in f:
                if '[' in line:
                    city = re.search('] (.*?) ', line).group(1)
                    if city == 'Toronto':
                        school_is_in_toronto = True
                        school_name = re.search('(.*) \[', line)
                        school_name = school_name.group(1).upper()
                if 'Overall' in line:
                    if school_is_in_toronto:
                        ratings = re.search('10 (.*)', line)
                        ratings = ratings.group(1).split()[0:-1]  # Step 3
                        row = df['NAME'].str.startswith(school_name)

                        if 'Elementary Schools 2017.txt' in file:
                            df.loc[row, 'SCHOOL_TYPE'] = 'ELEMENTARY'
                        else:
                            df.loc[row, 'SCHOOL_TYPE'] = 'SECONDARY'

                        if 'n/a' in ratings:
                            ratings = fill_missing_values(ratings)  # Step 4
                        for i in range(len(ratings)):
                            column = str(years[i]) + ' RATING'
                            df.loc[row, column] = ratings[i]  # Step 5

                        school_is_in_toronto = False

    df = df.drop_duplicates(subset=['2013 RATING', '2014 RATING',
                                    '2015 RATING', '2016 RATING',
                                    '2017 RATING'])  # Step 6
    df = df.loc[df['SCHOOL_BOARD'] != 'FP']  # Step 7
    df = df.reset_index(drop=True)
    df['INCOME'] = find_incomes_of_areas(df, income_df)  # Step 8
    df = find_bins(df, 'INCOME', 10, 200000)  # Step 9

    df = df[pd.notnull(df['2013 RATING'])]  # Step 10
    df.to_csv(output_csv, index=False)

data_file_names = {
    'files': ['Elementary Schools 2017.txt', 'Secondary Schools 2017.txt'],
    'school_database': 'SCHOOL.xlsx',
    'cnsmpr_database': 'DA_to_lat_long.csv',
    'statscan_database': 'income_by_DA.csv',
    'data_folder_name': 'Data'
}

preprocess(**data_file_names)
