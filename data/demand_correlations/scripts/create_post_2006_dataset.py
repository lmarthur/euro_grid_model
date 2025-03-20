'''
Script to read in the hourly_demand_2006-2023.csv file (which is our processed demand file from the hourly ENTSO-E files) and create a new dataset containing only the countries with high-quality data.
Specifically only want countries with high quality data all the way from 2006. Currently the daily average generation is saved.
NOTE: the .npz files that are read in by this script are too large to be stored in the repo, and therefore this script will not run.
However, the output file is saved in the repo as supply_and_demand_post_2006.csv
'''

import numpy as np
import pandas as pd
import sys
import pickle as pkl
import matplotlib.pyplot as plt

# read in the text file with the good countries
filepath = "./data/demand_correlations/misc_files/good_countries_post_2006.txt"
with open(filepath, "r", encoding="utf-8") as file:
    countries_with_good_demand_data = [line.strip() for line in file]

hourly_demand_data = pd.read_csv('./data/demand_correlations/euro_grid/data/demand_correlations/hourly_demand_2006-2023.csv')
code_to_country_name_dict = pkl.load(open('./data/demand_correlations/misc_files/country_codes_dictionary.pkl', 'rb'))

# Create a dictionary to convert country names to country codes
country_name_to_code_dict = {}
for key, value in code_to_country_name_dict.items():
    country_name_to_code_dict[value] = key


# New dataset to store the hourly demand data for the countries with high-quality data
new_hourly_demand_data = pd.DataFrame()
new_hourly_demand_data['Date'] = hourly_demand_data['Date']
new_hourly_demand_data['Date'] = pd.to_datetime(new_hourly_demand_data['Date']) # put into DateTime format
new_hourly_demand_data['Hour_of_day'] = hourly_demand_data['Hour_of_day']

# Filter out rows where the year is 2023 because there is no supply data for 2023
new_hourly_demand_data = new_hourly_demand_data[new_hourly_demand_data['Date'].dt.year != 2023]

# Here is all the supply data
solar_generation_array = np.load('./data/data_preprocessing_files/solar_generation.npz')
wind_generation_array = np.load('./data/data_preprocessing_files/wind_generation.npz')
countries_array = np.load('./data/data_preprocessing_files/countries.npz')
solar_generation = solar_generation_array['arr_0']
wind_generation = wind_generation_array['arr_0']
countries_on_map = countries_array['arr_0']

#  quick calculation to see how many grid squares are covered by the good countries.
countries_count = 0
for country in countries_with_good_demand_data:
    masked_country = np.isin(countries_on_map, country)
    print(country, np.sum(masked_country))
    countries_count += np.sum(masked_country)

print(countries_count)

for country in countries_with_good_demand_data:
    print(country)
    # First, add in all the demand data for each country
    code = country_name_to_code_dict[country]
    country_ds = hourly_demand_data[code]
    heading = country + ' Demand'
    new_hourly_demand_data[heading] = country_ds

    # Now add in all the supply data for each country
    countries_mask = np.isin(countries_on_map, country)

    solar_generation_country = solar_generation[countries_mask]
    wind_generation_country = wind_generation[countries_mask]
    total_solar_generation_country = np.sum(solar_generation_country, axis=0)
    total_wind_generation_country = np.sum(wind_generation_country, axis=0)

    # no demand data prior to 2006
    years_mask = []
    list_of_years = np.arange(1980, 2023)
    years_to_remove = np.arange(1980, 2006)
    for year in list_of_years:
        if year in years_to_remove:
            years_mask.extend([False]*365*24)
        else:
            years_mask.extend([True]*365*24)


    total_solar_generation_country = np.array(total_solar_generation_country)
    total_solar_generation_country = total_solar_generation_country[years_mask]

    total_wind_generation_country = np.array(total_wind_generation_country)
    total_wind_generation_country = total_wind_generation_country[years_mask]

    solar_key = country + ' Solar'
    wind_key = country + ' Wind'

    new_hourly_demand_data[solar_key] = total_solar_generation_country
    new_hourly_demand_data[wind_key] = total_wind_generation_country


# Create a 'total' column for each thing
new_hourly_demand_data['Total Demand'] = new_hourly_demand_data.loc[:, new_hourly_demand_data.columns.str.endswith('Demand')].sum(axis=1, skipna=False)
new_hourly_demand_data['Total Solar'] = new_hourly_demand_data.loc[:, new_hourly_demand_data.columns.str.endswith('Solar')].sum(axis=1, skipna=False)
new_hourly_demand_data['Total Wind'] = new_hourly_demand_data.loc[:, new_hourly_demand_data.columns.str.endswith('Wind')].sum(axis=1, skipna=False)



### NOTE ###
# COULD SAVE THE DATASET HERE AND IT WILL BE IN HOURLY FORMAT NOT DAILY FORMAT
#new_hourly_demand_data.to_csv('./data/demand_correlations/good_hourly_demand_data_post_2006.csv', index=False)

# Instead, let's create a daily average
daily_avg = new_hourly_demand_data.groupby('Date').agg(lambda x: x.mean(skipna=False)).reset_index()
daily_avg = daily_avg.drop(columns=['Hour_of_day'])

# Add columns for year, month, day of week, and day of year. NOTE that the leap years will have 366 days not 365
daily_avg.insert(1, 'year', pd.to_datetime(daily_avg['Date']).dt.year)
daily_avg.insert(2, 'month', pd.to_datetime(daily_avg['Date']).dt.strftime('%B'))
daily_avg.insert(3, 'day_of_week', pd.to_datetime(daily_avg['Date']).dt.strftime('%A'))
daily_avg.insert(4, 'day_of_year', pd.to_datetime(daily_avg['Date']).dt.dayofyear)


print(daily_avg.head())

# save the daily average dataset
daily_avg.to_csv('./data/demand_correlations/supply_and_demand_post_2006.csv', index=False)

