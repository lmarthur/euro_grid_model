'''
Script to check the hourly demand figures against the annual EMBER ones to verify reliability.
Basically a cross-check between EMBER and ENTSO-E
'''

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

hourly_demand_data = pd.read_csv('./data/demand_correlations/euro_grid/data/demand_correlations/hourly_demand_2006-2023.csv')
annual_demand_data = pd.read_csv('./data/demand_correlations/EMBER_data_files/annual_energy_data.csv')

dict_of_country_codes = pickle.load(open('./data/demand_correlations/misc_files/country_codes_dictionary.pkl', 'rb'))


for country_code in dict_of_country_codes:
    country_code = 'DE'
    #print('Country code:', country_code)
    country_name = dict_of_country_codes[country_code]
    print('Country name:', country_name)

    # France plot
    list_of_ember_results = []
    list_of_entsoe_results = []
    list_of_entsoe_years = []
    list_of_ember_years = []
    list_of_hours_per_year = []


    try:
        for year in range(2006, 2024):
            #print('YEAR:', year)
            hourly_data_for_specific_year = hourly_demand_data[hourly_demand_data['Date'].str[:4]==str(year)]
            list_of_hourly_values = hourly_data_for_specific_year[country_code].values
            #print('no of hourly values:', len(list_of_hourly_values))

            real_values = list_of_hourly_values[~np.isnan(list_of_hourly_values)]
            #print('Number of real values:', len(real_values))

            sum_of_real_values = np.sum(real_values)
            #print('ENTSO-E:', sum_of_real_values/1e6)

            country_annual_data = annual_demand_data[annual_demand_data['country']==country_name]
            country_annual_data_specific_year = country_annual_data[country_annual_data['year']==year]
            total_demand = country_annual_data_specific_year['electricity_demand'].values[0]

            #print('Ember: ', total_demand)

            if sum_of_real_values/1e6 > 0:
                list_of_entsoe_results.append(sum_of_real_values/1e6)
                list_of_entsoe_years.append(year)
                list_of_hours_per_year.append(len(real_values))

            if total_demand > 0:
                list_of_ember_results.append(total_demand)
                list_of_ember_years.append(year)


        # Creating the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))  # Larger figure size for better clarity

        # Plotting demand data on the primary y-axis
        ax1.plot(list_of_ember_years, list_of_ember_results, label='Ember', color='#32CD32', linestyle='--', linewidth=2.5, marker='o', markersize=8)  # Bright Lime Green
        ax1.plot(list_of_entsoe_years, list_of_entsoe_results, label='ENTSO-E', color='#8A2BE2', linestyle='--', linewidth=2.5, marker='o', markersize=8)  # Vivid Blue Violet   

        # Customizing the primary y-axis
        ax1.set_xlabel('Year', fontsize=14, weight='bold', labelpad=15)  # Add padding with labelpad
        ax1.set_ylabel('Annual Demand (TWh)', fontsize=14, weight='bold', labelpad=15)
        ax1.set_title(country_name, fontsize=16, weight='bold', pad=20)  # Professional title
        ax1.set_xticks(list_of_ember_years)
        ax1.set_xticklabels(list_of_ember_years, fontsize=12, rotation=90)  # Larger x-axis tick labels
        ax1.tick_params(axis='y', labelsize=12)
        ax1.set_ylim(0, max(list_of_ember_results) * 1.5)  # Setting the y-axis limits

        # Adding gridlines
        ax1.grid(visible=True, linestyle='--', alpha=0.7)

        # Adding legend for the primary y-axis
        ax1.legend(fontsize=12, frameon=True, loc='upper left')  # Professional-looking legend

        # Creating a twin y-axis for hours per year
        ax2 = ax1.twinx()
        ax2.axhline(y=8760, color='black', linestyle='-', linewidth=1, alpha=1, label = '8760 hours / year')  # Adding a horizontal line for 8760 hours
        ax2.plot(list_of_entsoe_years, list_of_hours_per_year, label='Hours per Year', color='grey', alpha=0.3, linestyle='--', linewidth=2.5, marker='o', markersize=8)
        
        # Customizing the secondary y-axis
        ax2.set_ylabel('Hours per Year with Coverage', fontsize=14, weight='bold', labelpad=15)
        ax2.tick_params(axis='y', labelsize=12, colors='black')
        ax2.set_ylim(0, max(list_of_hours_per_year) * 1.2)  # Adjusting the y-axis limits for hours

        # Adding legend for the secondary y-axis
        ax2.legend(fontsize=12, frameon=True, loc='upper right')

        # Tight layout for cleaner appearance
        plt.tight_layout()

        # Saving the plot
        #plt.savefig(f'./data/demand_correlations/validating_hourly_datasets/{country_name}.png')
        plt.show()
        plt.close()

    except:
        print('Failed for country:', country_name)
