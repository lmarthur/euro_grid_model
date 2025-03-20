'''
Script to take in the raw input files from ENTSO-E and create a giant dataset called hourly_demand_2006-2023.csv
This seeks to combine all the ENTSO-E datasets into a single csv file.

NOTE
The 2006-2015 data has several entries where no coverage ratio value is given. We have ignored for the moment but reached out to ENTSO-E to see if they can provide this information.
UPDATE: they can't provide the information, so just ignore the data with no coverage ratio.

NOTE
The 2015-2019 dataset actually only contains 2016-2017 data for most countries. Only Germany has 2015 data and NO countries have 2018 data.
The 2015 values for Germany from the 2006-2015 and 2015-2019 datasets appear to contradict each other. 
UPDATE: ENTSO-E can't provide any info on pre-2019 datasets.

NOTE
There is no data for any country at one of the hours on the 26th March 2006. 
'''

import pandas as pd
import pycountry
import numpy as np
import datetime
import pickle

list_of_demand_files = [
    'hourly_load_values_2006-2015.csv',
    'hourly_load_values_2015-2019.csv',
    'hourly_load_values_2019.csv',
    'hourly_load_values_2020.csv',
    'hourly_load_values_2021.csv',
    'hourly_load_values_2022.csv',
    'hourly_load_values_2023.csv',
    #'hourly_load_values_2024.csv',
]


file_1 = pd.read_csv(f'./data/demand_correlations/ENTSO_E_data_files/{list_of_demand_files[0]}')

# remove rows 0,1 and 2

file_1 = file_1.drop([0,1,2])

# cycle through columns and name them to make the data easier to process

file_1 = file_1.rename(columns={file_1.columns[0]: 'Country'})
file_1 = file_1.rename(columns={file_1.columns[1]: 'Year'})
file_1 = file_1.rename(columns={file_1.columns[2]: 'Month'})
file_1 = file_1.rename(columns={file_1.columns[3]: 'Day_of_month'})
file_1 = file_1.rename(columns={file_1.columns[4]: 'Coverage_ratio'})

for i in range(5, len(file_1.columns)):
    file_1 = file_1.rename(columns={file_1.columns[i]: f'hour_{i-5}'})

# cycle through all the countries in the country column

country_codes = list(file_1['Country'].unique())

dict_of_country_codes = {}

for country_code in country_codes:
    if country_code == 'DK_W':
        dict_of_country_codes[country_code] = 'Denmark (West)'
    elif country_code == 'UA_W':
        dict_of_country_codes[country_code] = 'Ukraine (West)'
    elif country_code == 'CS':
        dict_of_country_codes[country_code] = 'Serbia and Montenegro'
    else:
        dict_of_country_codes[country_code] = pycountry.countries.get(alpha_2=country_code).name

print(dict_of_country_codes)

# we now have a dictionary
for country_code in dict_of_country_codes:
    print(country_code)
    print(dict_of_country_codes[country_code])

# cycle through the rest of the files

for file in list_of_demand_files[1:]:
    print('FILE:', file)
    file = pd.read_csv(f'./data/demand_correlations/ENTSO_E_data_files/{file}')

    new_country_codes = list(file['CountryCode'].unique())

    for new_country_code in new_country_codes:
        if new_country_code not in dict_of_country_codes:
            try:
                if new_country_code == 'XK':
                    country_name = 'Kosovo'
                else:
                    country_name = pycountry.countries.get(alpha_2=new_country_code).name
                print(new_country_code)
                print(country_name)
                dict_of_country_codes[new_country_code] = country_name
            except:
                print(new_country_code, ': country not found')


print(len(dict_of_country_codes))

# date, hour of day, and 44 countries -- this gives 46 columns.
num_columns = 46

# hourly data for 18 years, neglecting 29th February. This gives 365 * 24 * 18 rows.
num_rows = 365 * 24 * 18

# create the output dataframe
output_df = pd.DataFrame(np.nan, index=range(num_rows), columns=range(num_columns))
column_labels = ['Date', 'Hour_of_day'] + list(dict_of_country_codes.keys())
output_df.columns = column_labels

# fill out the dates with YY-MM-DD format, exluding 29th February
list_of_dates = []
list_of_hours_of_day = []

date = datetime.datetime(2006, 1, 1)
for i in range(num_rows):
    if date.month == 2 and date.day == 29:
        date += datetime.timedelta(days=1)
    list_of_dates.append(date.strftime('%Y-%m-%d'))
    list_of_hours_of_day.append(date.strftime('%H'))
    date += datetime.timedelta(hours=1)

output_df['Date'] = list_of_dates
output_df['Hour_of_day'] = list_of_hours_of_day
output_df['Hour_of_day'] = output_df['Hour_of_day'].astype(int)

# Iterate through all the files and and add the data to the output dataframe

# File 1...
list_of_country_codes_in_file_1 = list(file_1['Country'].unique())
for code in dict_of_country_codes:
    if code in list_of_country_codes_in_file_1:
        print(code)

        country_df = file_1[file_1['Country'] == code]

        # Rename the Day_of_month column as Day because this is required for the pandas to_datetime function to work
        country_df = country_df.rename(columns={'Day_of_month': 'Day'})

        country_df['Date'] = pd.to_datetime(country_df[['Year', 'Month', 'Day']])
        country_df['Date'] = country_df['Date'].dt.strftime('%Y-%m-%d')

        # Use melt() to reshape the DataFrame to the form that I want
        hour_columns = [f'hour_{h}' for h in range(24)]
        melted_df = country_df.melt(
            id_vars=['Date', 'Coverage_ratio'],  # Keep 'Date' as is
            value_vars=hour_columns,  # The columns to melt
            var_name='Hour_of_day',  # Name for the new 'hour_of_day' column
            value_name=code  # Name for the new 'demand' column
        )

        melted_df['Hour_of_day'] = melted_df['Hour_of_day'].str.extract('(\d+)').astype(int)
        melted_df = melted_df.sort_values(by=['Date', 'Hour_of_day']).reset_index(drop=True)
        melted_df['Coverage_ratio'] = pd.to_numeric(melted_df['Coverage_ratio'])
        melted_df[code] = melted_df['Coverage_ratio'] * melted_df[code] / 100

        #drop the coverage ratio column
        melted_df = melted_df.drop(columns=['Coverage_ratio'])


        output_df.set_index(['Date', 'Hour_of_day'], inplace=True)
        melted_df.set_index(['Date', 'Hour_of_day'], inplace=True)
        output_df.update(melted_df[code])
        output_df.reset_index(inplace=True)


# Now cycle through the remainder of the files
for filename in list_of_demand_files[1:]:
    print('FILE:', filename)
    file = pd.read_csv(f'./data/demand_correlations/{filename}')

    new_country_codes = list(file['CountryCode'].unique())

    # Cycle through every country
    for code in dict_of_country_codes:
        if code in new_country_codes:
            print(code)

            country_df = file[file['CountryCode'] == code].copy()

            # Extract the hour as an integer
            country_df['Hour_of_day'] = country_df['TimeFrom'].str.split(':').str[0].astype(int)

            # Note that the date format is different for the 2015-2019 file

            if filename == 'hourly_load_values_2015-2019.csv' or filename == 'hourly_load_values_2021.csv':
                country_df['Date'] = pd.to_datetime(country_df['DateShort'], format='%m-%d-%y').dt.strftime('%Y-%m-%d')
            else:
                country_df['Date'] = pd.to_datetime(country_df['DateShort'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')

            true_load = country_df['Cov_ratio'] * country_df['Value'] / 100
            country_df[code] = true_load

            output_df.set_index(['Date', 'Hour_of_day'], inplace=True)
            country_df.set_index(['Date', 'Hour_of_day'], inplace=True)
            output_df.update(country_df[code])
            output_df.reset_index(inplace=True)

# save as a csv file
output_df.to_csv('./data/demand_correlations/hourly_demand_2006-2023.csv', index=False)