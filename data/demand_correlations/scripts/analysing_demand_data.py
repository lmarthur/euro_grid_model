'''
File to analyse the post-2006 electricity demand data. In particular, plotting time-traces of demand patterns over the course of a year.
'''

import numpy as np
import pandas as pd
import sys
import pickle as pkl
import matplotlib.pyplot as plt


# read in the post-2006 daily average dataset
daily_avg = pd.read_csv('./data/demand_correlations/supply_and_demand_post_2006.csv')

for key in daily_avg.keys():
    print(key)



# 2018 is missing data, 2020/2021 are COVID years, and 2023 has no corresponding demand data in our analysis
list_of_years = np.arange(2006, 2024)
years_to_remove = [2018, 2020, 2021, 2023]
list_of_years = list_of_years[~np.isin(list_of_years, years_to_remove)]
list_of_colors = plt.cm.plasma(np.linspace(0, 1, len(list_of_years)))


# PLOT 1: simple plot with no modifications
# NOTE: the leap years go to 366 not 365 days (and day 60 is skipped)
for year, color in zip(list_of_years, list_of_colors):
    print(year)
    daily_avg_for_specific_year = daily_avg[daily_avg['year'] == year]
    daily_avg_for_specific_year['Total Demand GW'] = daily_avg_for_specific_year['Total Demand']/1e3
    plt.plot(daily_avg_for_specific_year['day_of_year'], daily_avg_for_specific_year['Total Demand GW'], label = str(year), color=color)
    plt.xlabel('Day of year')
    plt.ylabel('Daily Average Demand (GW)')
    plt.ylim(0, 300)
    plt.legend()
#plt.show()


# PLOT 2: line-up by weekdays and weekends
for year, color in zip(list_of_years, list_of_colors):
    # cut down from 3rd Monday of the year to the 50th Monday of the year for every year
    mondays = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="W-MON")
    date_of_third_monday = mondays[2] 
    date_of_fiftieth_monday = mondays[49]
    day_of_third_monday = date_of_third_monday.dayofyear
    day_of_fiftieth_monday = date_of_fiftieth_monday.dayofyear

    daily_avg_for_specific_year = daily_avg[daily_avg['year'] == year]
    daily_avg_for_specific_year = daily_avg_for_specific_year[(daily_avg_for_specific_year['day_of_year'] >= day_of_third_monday) & (daily_avg_for_specific_year['day_of_year'] <= day_of_fiftieth_monday)]
    daily_avg_for_specific_year['days_after_third_monday'] = daily_avg_for_specific_year['day_of_year'] - day_of_third_monday

    daily_avg_for_specific_year['Total Demand GW'] = daily_avg_for_specific_year['Total Demand']/1e3

    plt.plot(daily_avg_for_specific_year['days_after_third_monday'], daily_avg_for_specific_year['Total Demand GW'], label = str(year), color=color)
    plt.xlabel('Days after 3rd Monday of the year')
    plt.ylabel('Daily Average Demand (GW)')
    plt.ylim(0, 300)
    plt.legend()
#plt.show()


# Assuming `daily_avg` and other required variables are defined.

# Identify "bad days" (days missing in any year)
list_of_bad_days = []
for day in range(1, 366):
    for year in list_of_years:
        daily_avg_specific = daily_avg[(daily_avg['year'] == year) & (daily_avg['day_of_year'] == day)]
        if daily_avg_specific.empty or np.isnan(daily_avg_specific['Total Demand'].values[0]):
            list_of_bad_days.append(day)
            break

list_of_good_days = [day for day in range(1, 366) if day not in list_of_bad_days]

# Get 2006 total annual demand (only for good days)
daily_avg_for_2006 = daily_avg[daily_avg['year'] == 2006]
daily_avg_for_2006 = daily_avg_for_2006[daily_avg_for_2006['day_of_year'].isin(list_of_good_days)]
total_demand_for_2006 = np.sum(daily_avg_for_2006['Total Demand'])

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

for year, color in zip(list_of_years, list_of_colors):
    mondays = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="W-MON")
    date_of_third_monday = mondays[2]
    date_of_fiftieth_monday = mondays[49]
    day_of_third_monday = date_of_third_monday.dayofyear
    day_of_fiftieth_monday = date_of_fiftieth_monday.dayofyear

    daily_avg_for_specific_year = daily_avg[daily_avg['year'] == year]
    daily_avg_for_specific_year = daily_avg_for_specific_year[daily_avg_for_specific_year['day_of_year'].isin(list_of_good_days)]
    scale_factor = total_demand_for_2006 / np.sum(daily_avg_for_specific_year['Total Demand'])

    daily_avg_for_specific_year = daily_avg_for_specific_year[
        (daily_avg_for_specific_year['day_of_year'] >= day_of_third_monday) & 
        (daily_avg_for_specific_year['day_of_year'] <= day_of_fiftieth_monday)
    ]
    daily_avg_for_specific_year['days_after_third_monday'] = (
        daily_avg_for_specific_year['day_of_year'] - day_of_third_monday
    )
    daily_avg_for_specific_year['Total Demand GW'] = daily_avg_for_specific_year['Total Demand'] / 1e3

    daily_avg_for_specific_year = daily_avg_for_specific_year.sort_values('days_after_third_monday')
    daily_avg_for_specific_year['gap'] = daily_avg_for_specific_year['days_after_third_monday'].diff() > 1

    days = daily_avg_for_specific_year['days_after_third_monday'].values
    demand = daily_avg_for_specific_year['Total Demand GW'].values * scale_factor

    days_with_gaps, demand_with_gaps = [], []
    for i in range(len(days)):
        days_with_gaps.append(days[i])
        demand_with_gaps.append(demand[i])
        if i < len(days) - 1 and daily_avg_for_specific_year['gap'].iloc[i + 1]:
            days_with_gaps.append(np.nan)
            demand_with_gaps.append(np.nan)

    ax.plot(days_with_gaps, demand_with_gaps, label=str(year), color=color, alpha=0.8, linewidth=1.2)

# Customizing the plot for publication quality
ax.set_xlabel('Days after 3rd Monday of the Year', fontsize=14)
ax.set_ylabel('Normalised Daily Average Demand (GW)', fontsize=14)
ax.set_ylim(0, 300)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(title="Year", fontsize=10, title_fontsize=12, loc='upper right', bbox_to_anchor=(1.15, 1))

# Grid adjustments
ax.grid(True, linestyle="--", alpha=0.5)

# Display the improved plot
plt.tight_layout()
plt.savefig('./data/demand_correlations/correlation_plots/annual_demand_patterns.pdf')
plt.show()
