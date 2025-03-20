'''
Script for processing the raw files from renewables.ninja and the world bank and some csv files on installed renewable generation from IRENA.

Any output files are saved to the raw_country_subset directory.
The exception to this is the ninja_total_supply_and_demand.csv file which is saved to the data directory.
(ninja_total_supply_and_demand.csv is the equivalent to the old total_supply_and_demand data file.)

'''

import pandas as pd 
import numpy as np
from datetime import datetime, timedelta



#list of country codes
list_of_country_codes = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IT', 'LT', 'LU', 'LV', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']

#list of country names (as they appear in the world bank database of country areas)
list_of_country_names = ['Austria', 'Belgium', 'Bulgaria', 'Switzerland', 'Czechia', 'Germany', 'Denmark', 'Estonia', 'Spain', 'Finland', 'France', 'United Kingdom', 'Greece', 'Croatia', 'Hungary', 'Italy', 'Lithuania', 'Luxembourg', 'Latvia', 'North Macedonia', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Sweden', 'Slovenia', 'Slovak Republic']




# SOLAR DATA PROCESSING FROM RENEWABLES.NINJA
solar_data = pd.read_csv('./data/data_processing_files/ninja_data_processing/raw_renewables_ninja_downloads/ninja_pv_europe_v1.1_merra2.csv')
columns_to_include = []
for col in solar_data.columns:
    if col in list_of_country_codes or col == 'time':
        columns_to_include.append(col)

#sorted solar data. Only countries I want and 1985-2016
solar_data  =  solar_data[columns_to_include]



# WIND DATA PROCESSING FROM RENEWABLES.NINJA
wind_data = pd.read_csv('./data/data_processing_files/ninja_data_processing/raw_renewables_ninja_downloads/ninja_wind_europe_v1.1_current_on-offshore.csv')

#only grab onshore wind
columns_to_include = [col for col in wind_data.columns if 'OFF' not in col]
wind_data = wind_data[columns_to_include]

# only grab countries I want
columns_to_include = []
for col in wind_data.columns:
    col_name = col[:2]
    if col_name in list_of_country_codes or col == 'time':
        columns_to_include.append(col)
wind_data = wind_data[columns_to_include]

# renames the wind columns to get rid of the '_ON' suffix.
wind_data.columns = solar_data.columns # I've checked that the columns are the same for solar and wind

# remove all data before 1985 in the wind data, since there is no solar data then
wind_data = wind_data[wind_data['time'] >= '1985-01-01']


# remove all 29th Feb days on leap years from the wind and solar data
leap_day = '02-29'
solar_data = solar_data[~solar_data['time'].str.contains(leap_day)]
wind_data = wind_data[~wind_data['time'].str.contains(leap_day)]



# save the data
solar_data.to_csv('./data/data_processing_files/ninja_data_processing/raw_country_subset/wind_capacity_factors.csv', index=False)

# excellent, both arrays are in the same format, and have the same length (32 years = 280320 hours of data)


# COUNTRY AREA DATA PROCESSING FROM WORLD BANK
area_data = pd.read_csv('./data/data_processing_files/ninja_data_processing/raw_renewables_ninja_downloads/world_countries_by_area_from_world_bank.csv')
# area is in square km

#only want the most recent available data
cols_of_interest = ['Country Name', '2021']
area_data = area_data[cols_of_interest]

# only select rows of countries I want
area_data = area_data[area_data['Country Name'].isin(list_of_country_names)]

# save the data
area_data.to_csv('raw_country_subset/area_data.csv', index=False)

# Remove countries outside of renewables.ninja database (Serbia and Montenegro)
demand_data = pd.read_csv('./data/data_processing_files/demand.csv')

# Reduced to the 28 countries of interest
demand_data = demand_data[demand_data['CountryCode'].isin(list_of_country_codes)]

# add more rows for the new total demand
total_demand = []
for hour in range(0,8760):
    all_countries = demand_data[demand_data['Hour of Year'] == hour]
    demand_at_hour = sum(all_countries['Demand'])
    total_demand.append(demand_at_hour)

# Create new demand data frame for the total demand using the German demand dataframe as a template.
germany_hour_of_year = demand_data[demand_data['CountryCode'] == 'DE']['Hour of Year'].tolist()
all_countries_df = pd.DataFrame({'Hour of Year': germany_hour_of_year})
all_countries_df['Demand'] = total_demand
all_countries_df['Date'] = demand_data[demand_data['CountryCode'] == 'DE']['Date'].tolist()
all_countries_df['Hour of Day'] = demand_data[demand_data['CountryCode'] == 'DE']['Hour of Day'].tolist()
all_countries_df['CountryCode'] = 'Total'

#now just add this df with all total demand data to the full European one
total_df = pd.concat([all_countries_df, demand_data], ignore_index=True)

# save the data
total_df.to_csv('./data/data_processing_files/ninja_data_processing/raw_country_subset/total_demand_data.csv', index=False)

# NOW CALCULATE EUROPE-WIDE SOLAR GENERATION IN THREE WAYS
# 1. Area weighted average of capacity factors
# 2. Weighted average of capacity factors by 2016 installed capacity
# 3. Weighted average of capacity factors by 2022 installed capacity

# ninja supply and demand file will have a column associated with each of these

total_area = sum(area_data['2021'])

# initialise the arrays
area_weighted_solar_capacity_factors = np.zeros(len(solar_data))
area_weighted_wind_capacity_factors = np.zeros(len(wind_data))

# get area weighted capacity factors
for country, code in zip(list_of_country_names, list_of_country_codes):
    country_area = area_data[area_data['Country Name'] == country]['2021'].values[0]
    print(country_area)
    #solar
    area_weighted_solar_capacity_factors = area_weighted_solar_capacity_factors + (solar_data[code] * country_area / total_area)
    #wind
    area_weighted_wind_capacity_factors = area_weighted_wind_capacity_factors + (wind_data[code] * country_area / total_area)

# now weight by the 2016 and 2022 capacity factors

# Load the solar data
solar_installation_data = pd.read_csv('./data/ninja_data_processing/raw_renewables_ninja_downloads/europe_installed_solar.csv')

# subset the columns
columns_to_include = ['Country', '2016', '2022']
solar_installation_data = solar_installation_data[columns_to_include]

solar_capacity_factors_weighted_by_2016_installed_capacity = np.zeros(len(solar_data))
solar_capacity_factors_weighted_by_2022_installed_capacity = np.zeros(len(solar_data))

total_solar_capacity_2016 = sum(solar_installation_data['2016'])
total_solar_capacity_2022 = sum(solar_installation_data['2022'])

for country, code in zip(list_of_country_names, list_of_country_codes):
    country_2016_installed_capacity = solar_installation_data[solar_installation_data['Country'] == country]['2016'].values[0]
    country_2022_installed_capacity = solar_installation_data[solar_installation_data['Country'] == country]['2022'].values[0]
    solar_capacity_factors_weighted_by_2016_installed_capacity = solar_capacity_factors_weighted_by_2016_installed_capacity + (solar_data[code] * country_2016_installed_capacity / total_solar_capacity_2016)
    solar_capacity_factors_weighted_by_2022_installed_capacity = solar_capacity_factors_weighted_by_2022_installed_capacity + (solar_data[code] * country_2022_installed_capacity / total_solar_capacity_2022)

# Repeat for the wind data
wind_installation_data = pd.read_csv('./data/ninja_data_processing/raw_renewables_ninja_downloads/europe_installed_onshore_wind.csv')

# subset the columns
columns_to_include = ['Country', '2016', '2022']
wind_installation_data = wind_installation_data[columns_to_include]

wind_capacity_factors_weighted_by_2016_installed_capacity = np.zeros(len(wind_data))
wind_capacity_factors_weighted_by_2022_installed_capacity = np.zeros(len(wind_data))

total_wind_capacity_2016 = sum(wind_installation_data['2016'])
total_wind_capacity_2022 = sum(wind_installation_data['2022'])

for country, code in zip(list_of_country_names, list_of_country_codes):
    country_2016_installed_capacity = wind_installation_data[wind_installation_data['Country'] == country]['2016'].values[0]
    country_2022_installed_capacity = wind_installation_data[wind_installation_data['Country'] == country]['2022'].values[0]
    wind_capacity_factors_weighted_by_2016_installed_capacity = wind_capacity_factors_weighted_by_2016_installed_capacity + (wind_data[code] * country_2016_installed_capacity / total_wind_capacity_2016)
    wind_capacity_factors_weighted_by_2022_installed_capacity = wind_capacity_factors_weighted_by_2022_installed_capacity + (wind_data[code] * country_2022_installed_capacity / total_wind_capacity_2022)


# NOW CREATE THE SUPPLY AND DEMAND ARRAY IN THE SAME FORMAT


# Extend the demand data to 32 years
repeated_array = np.tile(total_demand, 32)
total_demand_gw = repeated_array / 1000 #converts to GW
peak_demand_gw = max(total_demand_gw) # 480 ish GW


# get all the numbers
solar_supply_area_weighted = peak_demand_gw * area_weighted_solar_capacity_factors
wind_supply_area_weighted = peak_demand_gw * area_weighted_wind_capacity_factors

solar_supply_2016_weighted = peak_demand_gw * solar_capacity_factors_weighted_by_2016_installed_capacity
solar_supply_2022_weighted = peak_demand_gw * solar_capacity_factors_weighted_by_2022_installed_capacity

wind_supply_2016_weighted = peak_demand_gw * wind_capacity_factors_weighted_by_2016_installed_capacity
wind_supply_2022_weighted = peak_demand_gw * wind_capacity_factors_weighted_by_2022_installed_capacity

hour = np.arange(0, 280320)
def generate_date_range(start_date, end_date):
    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime('%d/%m/%Y'))
        current_date += timedelta(days=1)

    return date_list

# Define start and end dates
start_date = datetime(1985, 1, 1)
end_date = datetime(2016, 12, 31)

# Generate the list of dates
date_strings = generate_date_range(start_date, end_date)
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# Remove dates corresponding to February 29 in leap years
filtered_dates = [date_str for date_str in date_strings if not (int(date_str[-4:]) % 4 == 0 and date_str[:5] == '29/02')]

# Function to extract day from a date string and convert it to an integer
def extract_day(date_str):
    date_object = datetime.strptime(date_str, '%d/%m/%Y')
    return date_object.day

# Extract days and convert to integers
day_list = [extract_day(date_str) for date_str in filtered_dates]

def extract_month(date_str):
    date_object = datetime.strptime(date_str, '%d/%m/%Y')
    return date_object.month

# Extract months and convert to integers
month_list = [extract_month(date_str) for date_str in filtered_dates]

def extract_year(date_str):
    date_object = datetime.strptime(date_str, '%d/%m/%Y')
    return date_object.year

year_list = [extract_year(date_str) for date_str in filtered_dates]


repeated_year_list = [item for item in year_list for _ in range(24)]
repeated_month_list = [item for item in month_list for _ in range(24)]
repeated_day_list = [item for item in day_list for _ in range(24)]
repeated_filtered_dates = [item for item in filtered_dates for _ in range(24)]


list_of_hours = np.arange(24)
all_hours = np.tile(list_of_hours, 11680) # 365 days x 32 years = 11680 days

solar_supply_area_weighted = solar_supply_area_weighted.tolist()
solar_supply_2016_weighted = solar_supply_2016_weighted.tolist()
solar_supply_2022_weighted = solar_supply_2022_weighted.tolist()
wind_supply_area_weighted = wind_supply_area_weighted.tolist()
wind_supply_2016_weighted = wind_supply_2016_weighted.tolist()
wind_supply_2022_weighted = wind_supply_2022_weighted.tolist()


# Create the ninja_total_supply_and_demand file which can be used in the rest of the analysis!
all_data = {
    'Total Hour' : hour,
    'Date' : repeated_filtered_dates,
    'Hour of Day' : all_hours,
    'Day of Month' : repeated_day_list,
    'Month of Year' : repeated_month_list,
    'Year' : repeated_year_list,
    'Total Demand (GW)' : total_demand_gw,
    'Solar Area Weighted (GW)' : solar_supply_area_weighted,
    'Solar 2016 Weighted (GW)' : solar_supply_2016_weighted,
    'Solar 2022 Weighted (GW)' : solar_supply_2022_weighted,
    'Wind Area Weighted (GW)' : wind_supply_area_weighted,
    'Wind 2016 Weighted (GW)' : wind_supply_2016_weighted,
    'Wind 2022 Weighted (GW)' : wind_supply_2022_weighted,
}

big_df = pd.DataFrame(all_data)

#save to csv file
big_df.to_csv('./data/ninja_total_supply_and_demand.csv', index=False)







