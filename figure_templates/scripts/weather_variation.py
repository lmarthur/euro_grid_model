'''
Recreate the Fig 3 plots from my initial 22.16 paper.
To switch between ninja data and original data:
1) change the supply/demand file that is being read
2) change the year from 2016 rather than 2022 (because 2022 is out of range for the ninja data).
3) change the save_file paths 
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fsize = 18

# 1) CHANGE THE SUPPLY/DEMAND FILE HERE.
#    ALSO NEED TO SPECIFY WHICH NORMALISATION TO USE IF USING NINJA DATA.
#    I.E WEIGHTING BY 2016 INSTALLED CAPACITY, 2022 INSTALLED CAPACITY OR AREA-WEIGHTED
data_type = 'area_avg' # 'ninja' or 'area_avg'

if data_type == 'ninja':
    df = pd.read_csv('./data/ninja_total_supply_and_demand.csv')

    df['Norm Wind (GW)'] = df['Wind 2022 Weighted (GW)']
    df['Norm Solar (GW)'] = df['Solar 2022 Weighted (GW)']
    save_folder = 'ninja_2022_weighted'

elif data_type == 'area_avg':
    df = pd.read_csv('./data/final_solar_total_supply_and_demand.csv')
    save_folder = 'area_avg'

else:
    raise ValueError("Invalid data type specified. Please choose 'ninja' or 'area_avg'.")

# 2) CHANGE THE YEAR/DATE HERE

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Get unique dates for 2016
unique_dates = df[df['Date'].dt.year == 2016]['Date'].unique()

# daily subplot
fig = plt.figure(figsize=(8, 9))  # Adjusted figure size
ax0 = plt.subplot(2, 1, 1)

alpha = 0.1

# Loop through each unique date and plot the data
for date in unique_dates:
    df_day = df[df['Date'] == date]
    ax0.plot(df_day['Hour of Day'], df_day['Total Demand (GW)'], linestyle='-', alpha=alpha, color='green')
    ax0.plot(df_day['Hour of Day'], df_day['Norm Wind (GW)'], linestyle='-', alpha=alpha, color='tab:blue')
    ax0.plot(df_day['Hour of Day'], df_day['Norm Solar (GW)'], linestyle='-', alpha=alpha, color='orange')
    

# Add a single legend entry for each data component
ax0.plot([], [], label='Demand', color='green')
ax0.plot([], [], label='Wind', color='tab:blue')
ax0.plot([], [], label='Solar', color='orange')


# Set labels and title with adjusted font sizes
ax0.set_xlabel('Hour of Day (2022)', fontsize=fsize)
ax0.set_ylabel('Power (GW)', fontsize=fsize)
# set the font size for the ticks
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)


plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
# 
# # Add grid for better readability
ax0.grid(True, linestyle='--', alpha=0.7)

# Add legend and specify its location with adjusted font size
ax0.legend(loc='upper left', fontsize=fsize)

# Customize the appearance of the plot
# plt.tight_layout()  # Adjust layout for better spacing

# Set the start date for the year
start_year = 1980 # df['Year'].max()
end_year = 2022 # df['Year'].max()
year = start_year

ax1 = plt.subplot(2, 1, 2)

for year in range(start_year, end_year+1):
    df_year = df[df['Date'].dt.year == year]
    unique_dates = df_year['Date'].unique()
    # print the leap days for each year

    days = np.arange(len(unique_dates)+1)

    mean_demand = np.zeros(len(unique_dates)+1)
    mean_solar = np.zeros(len(unique_dates)+1)
    mean_wind = np.zeros(len(unique_dates)+1)


    for date in unique_dates:
        df_day = df_year[df_year['Date'] == date]
        mean_demand[date.dayofyear-1] = df_day['Total Demand (GW)'].mean()
        mean_solar[date.dayofyear-1] = df_day['Norm Solar (GW)'].mean()
        mean_wind[date.dayofyear-1] = df_day['Norm Wind (GW)'].mean()

    # remove zero values from the data
    mean_demand = mean_demand[mean_demand != 0]
    mean_solar = mean_solar[mean_solar != 0]
    mean_wind = mean_wind[mean_wind != 0]

    # plot the mean demand, solar and wind for each unique date
    ax1.plot(mean_wind, linestyle='-', alpha=alpha, color='tab:blue')
    ax1.plot(mean_solar, linestyle='-', alpha=alpha, color='orange')
    

ax1.plot(mean_demand, linestyle='-', color='green')

# Set labels and title with adjusted font sizes
ax1.set_xlabel('Day of Year', fontsize=fsize)
ax1.set_ylabel('Power (GW)', fontsize=fsize)

# ax1.legend(loc='upper left', fontsize=14)

# set the font size for the ticks
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()  # Adjust layout for better spacing
plt.subplots_adjust(hspace=0.3)  # Adjust the spacing between subplots

plt.savefig(f'./figure_templates/figures/{save_folder}/weather_variation.pdf')  # Save the plot

plt.show()