# This script plots how the cheapest scenario changes with differing constraints on the grid reliability.

from scipy.integrate import quad
import numpy as np
import time
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots

# set the plotting style
#plt.style.use(['nature', 'vibrant'])
colors = plt.cm.plasma(np.linspace(0, 1, 5))
# 1) CHANGE THE SUPPLY/DEMAND FILE HERE. (ALSO NEED TO SPECIFY WHICH NORMALISATION TO USE IF USING NINJA DATA)
data_type = 'area_avg' # 'ninja' or 'area_avg'
grid_name = 'dense_grid_2_final_solar' # Define the input path (This should be changed for each run)

da = xr.open_dataarray(f'./output/{grid_name}/results_with_costs.nc')

if data_type == 'ninja':
    df = pd.read_csv('./data/ninja_total_supply_and_demand.csv')

    norm_wind = df['Wind 2022 Weighted (GW)'].values
    norm_solar = df['Solar 2022 Weighted (GW)'].values
    demand = df['Total Demand (GW)'].values
    peak_demand = np.max(demand)

    da.loc[{'variable': 'backup_used'}] = 32 * da.sel(variable='backup_used') / np.sum(demand) # 32 years of data

elif data_type == 'area_avg':
    df = pd.read_csv('./data/total_supply_and_demand.csv')

    norm_wind = df['Norm Wind (GW)'].values
    norm_solar = df['Norm Solar (GW)'].values
    demand = df['Total Demand (GW)'].values
    peak_demand = np.max(demand)

    da.loc[{'variable': 'backup_used'}] = 43 * da.sel(variable='backup_used') / np.sum(demand)

else:
    raise ValueError("data_type must be either 'ninja' or 'area_avg'")


# remove transmission contribution to total costs
da.loc[{'variable': 'total_cost'}] = da.sel(variable='wind_cost') + da.sel(variable='solar_cost') + da.sel(variable='storage_cost') + da.sel(variable='gas_cost')

# Separate out the lost_hours and total_cost DataArrays
lost_hours_da = da.sel(variable='lost_hours')
total_cost_da = da.sel(variable='total_cost')
backup_used_da = da.sel(variable='backup_used')

list_of_gas_allowed = np.array([0,0.01,0.02])
list_of_lost_hours = np.linspace(0, 100, 51)
#list_of_colors_for_gas = ['#FF0000', '#00FF00', '#0000FF']

fsize = 14
plt.figure(figsize=(8, 5))

for idx in range(len(list_of_gas_allowed)):
    amount_of_gas_allowed = list_of_gas_allowed[idx]
    percentage_of_gas_allowed = 100 * amount_of_gas_allowed

    list_of_total_costs  = []

    for lost_hours_allowed in list_of_lost_hours:


        # Filter based on nat gas and reliability constraints
        filtered_da = total_cost_da.where(lost_hours_da <= lost_hours_allowed, drop=True)
        filtered_da = filtered_da.where(backup_used_da <= amount_of_gas_allowed, drop=True)

        # Find the minimum cost among the filtered entries
        min_cost = filtered_da.min().item()
        min_cost_coords = da.where(da == min_cost, drop=True).coords

        # JUST TAKE THE COORDS OF THE FIRST VALUE. NOTE THAT THERE MAY BE MULTIPLE MINIMUM VALUES WHICH ARE BEING IGNORED HERE
        coord_dict = {dim: min_cost_coords[dim].values[0] for dim in min_cost_coords if dim != 'variable' and dim != 'all_coords'}


        # Extract the gas_backup and lost_hours values corresponding to the minimum cost coordinates
        gas_backup_value = da.sel(coord_dict).sel(variable='backup_used').item()
        lost_hours_value = da.sel(coord_dict).sel(variable='lost_hours').item()

        wind_cost_value = da.sel(coord_dict).sel(variable='wind_cost').item()
        solar_cost_value = da.sel(coord_dict).sel(variable='solar_cost').item()
        storage_cost_value = da.sel(coord_dict).sel(variable='storage_cost').item()
        gas_cost_value = da.sel(coord_dict).sel(variable='gas_cost').item()

        # Just print out the grid configuration for the cheapest solution with 100% reliability
        if lost_hours_allowed == list_of_lost_hours[0]:
            print('Cheapest at start')
            for dim in min_cost_coords:
                if dim != 'variable' and dim != 'all_coords':
                    print(f"{dim}: {min_cost_coords[dim].values}")
            print(f"\nValue of 'gas_backup' at the minimum cost entry: {gas_backup_value}")
            print(f"Value of 'lost_hours' at the minimum cost entry: {lost_hours_value}")

            print(f"Value of 'wind_cost' at the minimum cost entry: {int(wind_cost_value)}")
            print(f"Value of 'solar_cost' at the minimum cost entry: {int(solar_cost_value)}")
            print(f"Value of 'storage_cost' at the minimum cost entry: {int(storage_cost_value)}")
            print(f"Value of 'gas_cost' at the minimum cost entry: {int(gas_cost_value)}")
            print(f"Value of 'total_cost' at the minimum cost entry: {int(min_cost)}\n")

        # Print out the grid configuration for the cheapest solution with the lowest level of reliability in the list_of_lost_hours array. This corresponds to the far right of the plot

        if lost_hours_allowed == list_of_lost_hours[-1]:
            print('Cheapest at end')
            for dim in min_cost_coords:
                if dim != 'variable' and dim != 'all_coords':
                    print(f"{dim}: {min_cost_coords[dim].values}")
            print(f"\nValue of 'gas_backup' at the minimum cost entry: {gas_backup_value}")
            print(f"Value of 'lost_hours' at the minimum cost entry: {lost_hours_value}")

            print(f"Value of 'wind_cost' at the minimum cost entry: {int(wind_cost_value)}")
            print(f"Value of 'solar_cost' at the minimum cost entry: {int(solar_cost_value)}")
            print(f"Value of 'storage_cost' at the minimum cost entry: {int(storage_cost_value)}")
            print(f"Value of 'gas_cost' at the minimum cost entry: {int(gas_cost_value)}")
            print(f"Value of 'total_cost' at the minimum cost entry: {int(min_cost)}\n")

        list_of_total_costs.append(min_cost)

    list_of_total_costs = np.array(list_of_total_costs)
    plt.plot(list_of_lost_hours, list_of_total_costs, label = str(percentage_of_gas_allowed) + '% Nat Gas', linewidth=3)


horizontal_line_colors = plt.cm.plasma(np.linspace(0, 1, 5))

plt.axvline(x=2.63, linestyle='-', linewidth = 4, color='k', label='99.97% Reliability')
plt.axvline(x=8.76, linestyle='--', linewidth = 4, color='k', label='99.9% Reliability')
plt.axvline(x=87.6, linestyle=':', linewidth = 4, color='k', label='99% Reliability')

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)


plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
#plt.xlim([0.9,3.4])


plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=fsize, loc = 'upper right')
plt.xlabel('Permitted Loss of Load Expectation (Hours / year)', fontsize=fsize)
plt.ylabel(r'Annual System Cost (billions of $)', fontsize=fsize)
plt.tight_layout()

# may need to create the figures folder if it doens't exist
plt.savefig(f'./output/{grid_name}/figures/system_cost_vs_reliability.pdf')  # Save the plot if needed
#plt.show()