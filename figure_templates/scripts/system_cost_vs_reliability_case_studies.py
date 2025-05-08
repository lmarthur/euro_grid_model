# PLOTTING SCRIPT
# for chosen scenarios, plot the outage durations, the % of demand met and the costs.
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
#plt.style.use(['vibrant'])
colors = plt.cm.plasma(np.linspace(0, 1, 5))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.sim import run_timestep
fsize = 20


### USER INPUTS ###
# 1) Load in the simulation
simulation_name = 'dense_grid_2_final_solar'

# 2) choose 'area_avg' or 'ninja'
run_type = 'area_avg'

da = xr.open_dataarray(f'./output/{simulation_name}/results_with_costs.nc')

folder_to_save = f'./output/{simulation_name}/figures/reliability_scan'
os.makedirs(folder_to_save, exist_ok=True)

# area_avg branch
if run_type == 'area_avg':
    df = pd.read_csv('./data/total_supply_and_demand.csv')
    # Constant arrays
    demand = df['Total Demand (GW)'].values
    peak_demand = np.max(demand)
    norm_solar = df['Norm Solar (GW)'].values
    norm_wind = df['Norm Wind (GW)'].values

    da.loc[{'variable': 'backup_used'}] = 43 * da.sel(variable='backup_used') / np.sum(demand)

# ninja branch (2022 normalisation)
elif run_type == 'ninja':
    df = pd.read_csv('./data/ninja_total_supply_and_demand.csv')
    # Constant arrays
    demand = df['Total Demand (GW)'].values
    peak_demand = np.max(demand)
    norm_solar = df['Solar 2022 Weighted (GW)'].values
    norm_wind = df['Wind 2022 Weighted (GW)'].values

    da.loc[{'variable': 'backup_used'}] = 32 * da.sel(variable='backup_used') / np.sum(demand) # scale the backup to fraction of total electricity produced

else:
    raise ValueError("Invalid runtype specified. Please choose 'ninja' or 'area_avg'.")





amounts_of_LOLE_allowed = [2.86, 87.6] # 99.97% and 99% reliability
amoounts_of_gas_allowed = [0.00, 0.01]

# cycle through the different constraint scenarios, find the cheapest mix and plot the results.
for lost_hours_allowed in amounts_of_LOLE_allowed:
    for amount_of_gas_allowed in amoounts_of_gas_allowed:

        percentage_of_gas_allowed = amount_of_gas_allowed * 100

        da_copy = da.copy()

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


        # NOW RUN A PYTHON SIMULATION WITH THESE PARAMETERS TO GET THE OUTAGE INFORMATION
        backup_capacity = coord_dict['backup_capacity']
        storage_capacity = coord_dict['storage_capacity']
        overbuild_factor = coord_dict['overbuild_factor']
        threshold = coord_dict['threshold']
        prop_wind = coord_dict['prop_wind']


        prop_solar = 1-prop_wind
        solar_supply = norm_solar * prop_solar * overbuild_factor
        wind_supply = norm_wind * prop_wind * overbuild_factor
        renewable_supply = solar_supply + wind_supply
        number_of_timesteps = len(demand)


        list_of_storage_levels = np.zeros(number_of_timesteps + 1)
        list_of_total_supply = np.zeros(number_of_timesteps)
        list_of_backup_used = np.zeros(number_of_timesteps)

        list_of_storage_levels[0] = storage_capacity / 2

        for i in range(number_of_timesteps):
            storage_end, total_supply, backup_used = run_timestep(renewable_supply[i], list_of_storage_levels[i], threshold, backup_capacity * peak_demand, storage_capacity, demand[i])
            list_of_storage_levels[i+1] = storage_end
            list_of_total_supply[i] = total_supply
            list_of_backup_used[i] = backup_used

        list_of_storage_levels = list_of_storage_levels[:-1]


        xvalues = np.arange(0, number_of_timesteps, 1)
        no_of_years = number_of_timesteps / 8760
        supply_demand_delta = list_of_total_supply - demand

        count=0
        list_of_interruptions = []
        for entry in supply_demand_delta:
            if entry < -0.001:
                count+=1
            else:
                list_of_interruptions.append(count)
                count = 0
        filtered_list = [x for x in list_of_interruptions if x != 0]

        # Plot Outage Lengths
        plt.figure(figsize=(4, 3))
        plt.grid(True, linestyle='--', alpha=0.7)
        outage_bins = np.arange(0, 55, 5)  # Bins will be [0, 5, 10, ..., 50]
        plt.hist(filtered_list, bins=outage_bins, edgecolor='black', color=colors[0])


        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)

        plt.xticks(fontsize=fsize)
        # Divide x-ticks by 340
        plt.yticks(fontsize=fsize)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))  # Apply max 5 ticks



        #plt.ylim([0, 15])
        plt.xlim([0, 50])
        plt.axvline(x=24, linestyle='--', color='black')
        plt.axvline(x=24*2, linestyle='--', color='black')
        plt.xlabel('Outage Duration (hrs.)', fontsize = fsize)
        plt.ylabel('# of Outages', fontsize=fsize)
        plt.tight_layout(pad=1.5)  # Adjust the pad value
        plt.savefig(f'{folder_to_save}/gas_pc_{int(percentage_of_gas_allowed)}_LOLE_{int(lost_hours_allowed*100)}_outage_length.pdf', dpi=300)
        #plt.show()

        supply_demand_percent = (list_of_total_supply / demand) * 100 # fraction of demand that is met
        supply_demand_percent = supply_demand_percent[supply_demand_percent < 100] # unmet demand

        # Plot % of demand met
        plt.figure(figsize=(4, 3))
        plt.grid(True, linestyle='--', alpha=0.7)
        prop_demand_bins = np.arange(50, 105, 5)
        plt.hist(supply_demand_percent, bins=prop_demand_bins, edgecolor='black', color=colors[1])

        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))  # Apply max 5 ticks


        plt.xticks(fontsize=fsize)
        # Divide x-ticks by 340
        plt.yticks(fontsize=fsize)
        #plt.ylim([0, 15])
        plt.xlabel('% of Demand Met\n During Outage', fontsize = fsize)
        plt.xlim([45, 100])
        plt.ylabel('# of Hours', fontsize=fsize)
        plt.tight_layout(pad=1.5)  # Adjust the pad value
        plt.savefig(f'{folder_to_save}/gas_pc_{int(percentage_of_gas_allowed)}_LOLE_{int(lost_hours_allowed*100)}_prop_of_demand_met.pdf')
        # plt.show()

        # Bar chart for cost data

        wind_capacity = peak_demand * prop_wind * overbuild_factor
        solar_capacity = peak_demand * prop_solar * overbuild_factor
        gas_capacity = peak_demand * backup_capacity

        wind_capacity = int(wind_capacity)
        solar_capacity = int(solar_capacity)
        gas_capacity = int(gas_capacity)
        storage_capacity = int(storage_capacity)

        # Creating dynamic labels
        cost_labels = [
            f'Wind ({wind_capacity} GW)',
            f'Solar ({solar_capacity} GW)',
            f'Storage ({storage_capacity} GWh)',
            f'Gas ({gas_capacity} GW)'
        ]

        cost_values = [wind_cost_value, solar_cost_value, storage_cost_value, gas_cost_value] # these are from the original C simulation

        plt.figure(figsize=(5, 6.5))  # Adjusted width for clarity
        plt.grid(True, linestyle='--', alpha=0.7)

        # Create bar chart
        plt.bar(cost_labels, cost_values, edgecolor='black', color=colors)

        # Set the properties of the plot to match the histograms
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)

        # Rotate x-axis labels and ensure they are properly aligned
        plt.xticks(fontsize=fsize, rotation=45, ha="right")
        plt.yticks(range(0, 401, 50), fontsize=fsize)

        plt.ylabel('Annual Cost (billions of $)', fontsize=fsize)

        plt.title('$' + str(int(sum(cost_values))) + ' billion', fontsize=fsize)

        plt.tight_layout(pad=1.5)  # Adjust the pad value
        plt.savefig(f'{folder_to_save}/gas_pc_{int(percentage_of_gas_allowed)}_LOLE_{int(lost_hours_allowed*100)}_costs.pdf')
        #plt.show()




