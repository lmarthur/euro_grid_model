'''
Script to plot correlations between wind/solar supply and electricity demand in each month across the whole dataset.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Normalisation
no_of_grid_squares = 924

# Read dataset
daily_avg = pd.read_csv('./data/demand_correlations/supply_and_demand_post_2006.csv')

# Filter relevant years
list_of_years = np.arange(2006, 2024)
years_to_remove = [2018, 2020, 2021, 2023]
list_of_years = list_of_years[~np.isin(list_of_years, years_to_remove)]
daily_avg = daily_avg[daily_avg['year'].isin(list_of_years)]

# Normalize data
daily_avg['Total Solar'] /= no_of_grid_squares
daily_avg['Total Wind'] /= no_of_grid_squares
daily_avg['Total Demand GW'] = daily_avg['Total Demand'] / 1e3
daily_avg['Total Wind MW'] = daily_avg['Total Wind']/1e6

daily_avg['Total Wind Capacity Factor'] = 100 * daily_avg['Total Wind MW'] / 4.1 #normalise to rated turbine capacity to get capacity factor
daily_avg['Total Solar Capacity Factor'] = 100 * daily_avg['Total Solar'] / 210 #normalise to rated solar per m^2 to get capacity factor

# Define months and days
list_of_months = daily_avg['month'].unique()
list_of_week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
list_of_weekend_days = ['Saturday', 'Sunday']

# Plot solar and wind side by side with three linear fits for each subset
for m in list_of_months:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    
    daily_avg_for_specific_month = daily_avg[daily_avg['month'] == m]
    
    for ax, energy_type, label in zip(axes, ['Total Solar Capacity Factor', 'Total Wind Capacity Factor'], ['Solar', 'Wind']):
        daily_avg_specific_month_weekdays = daily_avg_for_specific_month[daily_avg_for_specific_month['day_of_week'].isin(list_of_week_days)]
        #daily_avg_specific_month_saturday = daily_avg_for_specific_month[daily_avg_for_specific_month['day_of_week'] == 'Saturday']
        #daily_avg_specific_month_sunday = daily_avg_for_specific_month[daily_avg_for_specific_month['day_of_week'] == 'Sunday']

        daily_avg_specific_month_weekend = daily_avg_for_specific_month[daily_avg_for_specific_month['day_of_week'].isin(list_of_weekend_days)]

        for data, category, color in zip([daily_avg_specific_month_weekdays, daily_avg_specific_month_weekend], 
                                         ['Weekdays', 'Weekends'], 
                                         ['tab:blue', 'tab:orange']):
            xvalues = data[energy_type].values
            yvalues = data['Total Demand GW'].values
            
            # remove NaN values from xvalues and yvalues (want to remove from both when either is nan)
            xnan_mask = np.isnan(xvalues)
            ynan_mask = np.isnan(yvalues)
            nan_mask = np.logical_or(xnan_mask, ynan_mask)
            xvalues = xvalues[~nan_mask]
            yvalues = yvalues[~nan_mask]


            ax.scatter(xvalues, yvalues, color=color, edgecolors='black', linewidth=0.5, alpha=0.7)
            
            coeffs = np.polyfit(xvalues, yvalues, 1)
            ax.plot(xvalues, np.polyval(coeffs, xvalues), color=color, linewidth=1.5, label = f'{category}: y = {coeffs[0]:.2f}x + {coeffs[1]:.0f}')


        ax.legend()
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=10)

    axes[0].set_title(f'{m}: Demand vs Solar Generation', fontsize=14,)

    axes[0].set_xlabel(rf'Average Daily Solar Capacity Factor ($\%$)', fontsize=12)
    axes[0].set_ylabel('Total Demand (GW)', fontsize=12,)
    #axes[0].set_xlabel('Average Daily Solar Generation per m²', fontsize=12, fontweight='bold')
    axes[0].set_ylim(140,270)
    #axes[0].set_xlim(10, 70)
    axes[1].set_title(f'{m}: Demand vs Wind Generation', fontsize=14,)

    axes[1].set_ylim(140,270)
    #axes[1].set_xlim(0, 2.5e6)
    axes[1].set_ylabel('Total Demand (GW)', fontsize=12,)
    axes[1].set_xlabel(rf'Average Daily Wind Capacity Factor ($\%$)', fontsize=12,)
    plt.tight_layout()
    plt.savefig(f'./data/demand_correlations/correlation_plots/{m}.pdf', dpi=300)
    plt.show()

# Now for the annual plot
    
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, energy_type, label in zip(axes, ['Total Solar Capacity Factor', 'Total Wind Capacity Factor'], ['Solar', 'Wind']):

    daily_avg_weekdays = daily_avg[daily_avg['day_of_week'].isin(list_of_week_days)]
    daily_avg_weekends = daily_avg[daily_avg['day_of_week'].isin(list_of_weekend_days)]

    for data, category, color in zip([daily_avg_weekdays, daily_avg_weekends],
                                        ['Weekdays', 'Weekends'],
                                        ['tab:blue', 'tab:orange']):

        xvalues = data[energy_type].values
        yvalues = data['Total Demand GW'].values

        # remove NaN values from xvalues and yvalues (want to remove from both when either is nan)
        xnan_mask = np.isnan(xvalues)
        ynan_mask = np.isnan(yvalues)
        nan_mask = np.logical_or(xnan_mask, ynan_mask)
        xvalues = xvalues[~nan_mask]
        yvalues = yvalues[~nan_mask]


        ax.scatter(xvalues, yvalues, color=color, edgecolors='black', linewidth=0.5, alpha=0.7)

        coeffs = np.polyfit(xvalues, yvalues, 1)
        ax.plot(xvalues, np.polyval(coeffs, xvalues), color=color, linewidth=1.5, label = f'{category}: y = {coeffs[0]:.2f}x + {coeffs[1]:.0f}')


    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=10)

axes[0].set_title(f'Annual: Demand vs Solar Generation', fontsize=14,)
axes[0].set_xlabel(rf'Average Daily Solar Capacity Factor ($\%$)', fontsize=12)
axes[0].set_ylabel('Total Demand (GW)', fontsize=12,)
axes[0].set_ylim(140,270)
axes[1].set_title(f'Annual: Demand vs Wind Generation', fontsize=14,)
axes[1].set_ylim(140,270)
axes[1].set_ylabel('Total Demand (GW)', fontsize=12,)
axes[1].set_xlabel(rf'Average Daily Wind Capacity Factor ($\%$)', fontsize=12,)
plt.tight_layout()
plt.savefig(f'./data/demand_correlations/correlation_plots/annual.pdf')
