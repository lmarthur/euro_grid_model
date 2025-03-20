# PLOTTING SCRIPT
# for how the cheapest scenario changes with different assumptions about the storage capital cost.
from scipy.integrate import quad
import numpy as np
import time
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import os
import scienceplots

# set the plotting style
plt.style.use(['nature', 'vibrant'])

# 1) CHANGE THE SUPPLY/DEMAND FILE HERE. NOTE: ALSO NEED TO SPECIFY WHICH NORMALISATION TO USE IF USING NINJA DATA (2016 capacity, 2022 capcity or area-weighted)

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
    df = pd.read_csv('./data/final_solar_total_supply_and_demand.csv')

    norm_wind = df['Norm Wind (GW)'].values
    norm_solar = df['Norm Solar (GW)'].values
    demand = df['Total Demand (GW)'].values
    peak_demand = np.max(demand)

    da.loc[{'variable': 'backup_used'}] = 43 * da.sel(variable='backup_used') / np.sum(demand)

else:
    raise ValueError("data_type must be either 'ninja' or 'area_avg'")


def correct_storage_cost(old_total_cost, total_capacity_in_kW, storage_o_and_m, old_overnight_capital_cost, new_overnight_capital_cost):
    '''
    Function to take in old storage costs, and return updated storage costs when the overnight capital cost is changed.
    Function relies on the assumption that the annual loan repayment is proportional to the overnight capital cost.

    INPUTS:
    -----------

    old_total_cost in billions of dollars
    total_capacity_in_kW in kW (NOTE NOT GW so a scaling factor may have to be applied to storage capacities in the xr object)
    storage_o_and_m in dollars per kW-year
    old_overnight_capital_cost in dollars per kW
    new_overnight_capital_cost in dollars per kW

    RETURNS:
    -----------
    new_total_cost in billions of dollars

    NOTE: the function currently returns nan values for the zero storage/zero capacity case due to a division by zero.
    This should probably be fixed in future but for the moment I just set all nan values to zero (and hope
    that there aren't any other cases which are leading to nan values).

    NOTE: the storage_o_and_m value which is passed to the function must be the same as the value that was used in the original cost calculations.
    Otherwise the function won't return the correct results.
    
    '''

    old_total_cost_in_dollars = old_total_cost * 1e9 # convert from billions of dollars to dollars
    old_cost_per_kW = old_total_cost_in_dollars / total_capacity_in_kW


    old_annual_loan_payment = old_cost_per_kW - storage_o_and_m #in $ per kW-year

    # apply the correction factor to the annual construction loan repayment. Annual loan repayment is proportional to the overnight capital cost
    new_annual_loan_payment = old_annual_loan_payment * (new_overnight_capital_cost / old_overnight_capital_cost)
    
    new_cost_per_kW = new_annual_loan_payment + storage_o_and_m
    new_total_cost_in_dollars = new_cost_per_kW * total_capacity_in_kW
    new_total_cost = new_total_cost_in_dollars / 1e9 # convert from dollars to billions of dollars

    new_total_cost = new_total_cost.where(~np.isnan(new_total_cost), 0) # set the nans to zero

    return new_total_cost


# remove transmission contribution to total costs
da.loc[{'variable': 'total_cost'}] = da.sel(variable='wind_cost') + da.sel(variable='solar_cost') + da.sel(variable='storage_cost') + da.sel(variable='gas_cost')


array_of_storage_capital_costs = np.arange(10, 401, 10)
list_of_gas_allowed = [0.00, 0.01, 0.02]
lost_hours_allowed = 3.26 # 99.97% reliability

list_of_colors_for_gas = ['#FF0000', '#00FF00', '#0000FF']


fsize = 14
plt.figure(figsize=(8, 5))


for idx in range(len(list_of_gas_allowed)):
    amount_of_gas_allowed = list_of_gas_allowed[idx]
    percentage_of_gas_allowed = 100 * amount_of_gas_allowed

    list_of_total_costs  = []

    for storage_capital_cost in array_of_storage_capital_costs:

        da_copy = da.copy()
        da_copy.loc[{'variable': 'storage_cost'}] = correct_storage_cost(da_copy.sel(variable='storage_cost'), da_copy['storage_capacity']*1e6, 10, 200, storage_capital_cost)
        da_copy.loc[{'variable': 'total_cost'}] = da_copy.sel(variable='wind_cost') + da_copy.sel(variable='solar_cost') + da_copy.sel(variable='storage_cost') + da_copy.sel(variable='gas_cost')

        # Separate out the lost_hours and total_cost DataArrays
        lost_hours_da = da_copy.sel(variable='lost_hours')
        total_cost_da = da_copy.sel(variable='total_cost')
        backup_used_da = da_copy.sel(variable='backup_used')


        # Filter based on nat gas and reliability constraints
        filtered_da = total_cost_da.where(lost_hours_da <= lost_hours_allowed, drop=True)
        filtered_da = filtered_da.where(backup_used_da <= amount_of_gas_allowed, drop=True)

        # Find the minimum cost among the filtered entries
        min_cost = filtered_da.min().item()

        min_cost_coords = filtered_da.where(filtered_da == min_cost, drop=True).coords

        # JUST TAKE THE COORDS OF THE FIRST VALUE. NOTE THAT THERE MAY BE MULTIPLE MINIMUM VALUES WHICH ARE BEING IGNORED HERE
        coord_dict = {dim: min_cost_coords[dim].values[-1] for dim in min_cost_coords if dim != 'variable' and dim != 'all_coords'}


        # Extract the gas_backup and lost_hours values corresponding to the minimum cost coordinates
        gas_backup_value = da_copy.sel(coord_dict).sel(variable='backup_used').item()
        lost_hours_value = da_copy.sel(coord_dict).sel(variable='lost_hours').item()

        wind_cost_value = da_copy.sel(coord_dict).sel(variable='wind_cost').item()
        solar_cost_value = da_copy.sel(coord_dict).sel(variable='solar_cost').item()
        storage_cost_value = da_copy.sel(coord_dict).sel(variable='storage_cost').item()
        gas_cost_value = da_copy.sel(coord_dict).sel(variable='gas_cost').item()


        # Just print out the grid configuration for the cheapest solution with the LOWEST storage capital cost
        if storage_capital_cost == array_of_storage_capital_costs[0]:
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

        # And print out the grid configuration for the cheapest solution with the HIGHEST storage capital cost
        if storage_capital_cost == array_of_storage_capital_costs[-1]:
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
    plt.plot(array_of_storage_capital_costs, list_of_total_costs, label = str(percentage_of_gas_allowed) + '% Nat Gas', linewidth=3)


plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)


plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
#plt.xlim([0.9,3.4])


plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Storage Capital Cost ($/kWh)', fontsize=fsize)
plt.ylabel('Annual System Cost (billions of $)', fontsize=fsize)
#plt.title('Annual System Cost vs Storage Capital Cost (at 99.97% Reliability)')
plt.legend(fontsize=fsize, loc = 'upper left')
plt.axvline(x=200, linestyle='--', color='k', label='200 $/kWh', linewidth=3)
plt.tight_layout()

# may need to create the figures folder if it doens't exist
plt.savefig(f'./output/{grid_name}/figures/system_cost_vs_storage_capital_cost.pdf')  # Save the plot if needed
#plt.show()