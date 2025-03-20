'''
This script takes the final production data (full_production_data.nc) and demand data (demand.csv).
Normalises supply to demand and returns the final dataset of solar and wind production over time.
Final information contained within

total_supply_and_demand.csv
'''
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


ds_production = xr.open_dataset('./data/data_processing_files/full_production_data.nc')
df_demand = pd.read_csv('./data/data_processing_files/demand.csv')

solar_power = ds_production['Solar Power']
wind_power = ds_production['Wind Power']

solar_supply = solar_power.values
wind_supply = wind_power.values

# Extract the total European electricity demand from the demand dataset
total_demand = df_demand[df_demand['CountryCode']=='Total']
total_demand_array = total_demand['Demand']

# Extend this over the 43 year period.
repeated_array = np.tile(total_demand_array, 43)
total_demand_gw = repeated_array / 1000 # converts to GW
peak_demand_gw = max(total_demand_gw) # 485.8 GW



solar_supply_gw = solar_supply / 1e9 # converts to GW
wind_supply_gw = wind_supply / 1e9 # converts to GW

#################
### IMPORTANT ###
### NORMALISATION TO PEAK DEMAND ###

# Normalise to Peak Demand. This is the required NAMEPLATE CAPACITY needed to meet peak demand in Europe.

# Actually has nothing to do with the weather data because it is all about NAMEPLATE CAPACITY.

# Rated (nameplate) capacity per unit area for the solar panels is simply 21% x 1000 W/m^2 = 210 W/m^2 (illumination at 1000W/m^2 is the test)

A_eff = peak_demand_gw / (210 * 1e-9) # this is the installed solar NAMEPLATE CAPACITY required to meet peak European demand

# Each turbine has a rated capacity of 4.1 MW.
N_eff = peak_demand_gw / (4.1 * 1e-3) # this is the installed wind NAMEPLATE CAPACITY required to meet peak European demand

print('EFFECTIVE NUMBERS')
print(A_eff)
print(N_eff)


solar_supply_gw_norm = solar_supply_gw * A_eff
wind_supply_gw_norm = wind_supply_gw * N_eff

hour = np.arange(0, 376680)
def generate_date_range(start_date, end_date):
    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime('%d/%m/%Y'))
        current_date += timedelta(days=1)

    return date_list

# Define start and end dates
start_date = datetime(1980, 1, 1)
end_date = datetime(2022, 12, 31)

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
all_hours = np.tile(list_of_hours, 15695)

# Now we create the total_supply_and_demand file which can be used in the rest of the analysis!
all_data = {
    'Total Hour' : hour,
    'Date' : repeated_filtered_dates,
    'Hour of Day' : all_hours,
    'Day of Month' : repeated_day_list,
    'Month of Year' : repeated_month_list,
    'Year' : repeated_year_list,
    'Total Demand (GW)' : total_demand_gw,
    'Norm Solar (GW)' : solar_supply_gw_norm,
    'Norm Wind (GW)' : wind_supply_gw_norm,
}

big_df = pd.DataFrame(all_data)

# save to csv file
big_df.to_csv('./data/total_supply_and_demand.csv', index=False)
