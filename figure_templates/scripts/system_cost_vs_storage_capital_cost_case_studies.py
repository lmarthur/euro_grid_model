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
from matplotlib.ticker import MaxNLocator
import scienceplots

# set the plotting style
#plt.style.use(['nature', 'vibrant'])
colors = plt.cm.plasma(np.linspace(0, 1, 5))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.sim import run_timestep
fsize = 20


### USER INPUTS ###
# 1) Load in the simulation
simulation_name = 'dense_grid_2_final_solar'

# 2) choose 'area_avg' or 'ninja'
run_type = 'area_avg'


folder_to_save = f'./output/{simulation_name}/figures/storage_cost_scan'
os.makedirs(folder_to_save, exist_ok=True)

da = xr.open_dataarray(f'./output/{simulation_name}/results_with_costs.nc')

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



storage_capital_costs = [10, 400]
amoounts_of_gas_allowed = [0.00, 0.01]

# cycle through the different constraint scenarios, find the cheapest mix and plot the results.
for storage_capital_cost in storage_capital_costs:
    for amount_of_gas_allowed in amoounts_of_gas_allowed:

        percentage_of_gas_allowed = amount_of_gas_allowed * 100
        lost_hours_allowed = 3.26 # 99.97% reliability

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

        print('min_cost_coords:', min_cost_coords)


        # JUST TAKE THE COORDS OF THE FIRST VALUE. NOTE THAT THERE MAY BE MULTIPLE MINIMUM VALUES WHICH ARE BEING IGNORED HERE
        coord_dict = {dim: min_cost_coords[dim].values[-1] for dim in min_cost_coords if dim != 'variable' and dim != 'all_coords'}

        print(da_copy.sel(coord_dict, method='nearest').sel(variable='storage_cost').item())


        # Extract the gas_backup and lost_hours values corresponding to the minimum cost coordinates
        gas_backup_value = da_copy.sel(coord_dict).sel(variable='backup_used').item()
        lost_hours_value = da_copy.sel(coord_dict).sel(variable='lost_hours').item()


        wind_cost_value = da_copy.sel(coord_dict).sel(variable='wind_cost').item()
        solar_cost_value = da_copy.sel(coord_dict).sel(variable='solar_cost').item()
        storage_cost_value = da_copy.sel(coord_dict).sel(variable='storage_cost').item()
        gas_cost_value = da_copy.sel(coord_dict).sel(variable='gas_cost').item()

        print('Wind cost:', wind_cost_value)
        print('Solar cost:', solar_cost_value)
        print('Storage cost:', storage_cost_value)
        print('Gas cost:', gas_cost_value)

        # NOW RUN A PYTHON SIMULATION WITH THESE PARAMETERS TO GET THE OUTAGE INFORMATION
        backup_capacity = coord_dict['backup_capacity']
        storage_capacity = coord_dict['storage_capacity']
        overbuild_factor = coord_dict['overbuild_factor']
        threshold = coord_dict['threshold']
        prop_wind = coord_dict['prop_wind']

        print('backup_capacity:', backup_capacity)
        print('storage_capacity:', storage_capacity)
        print('overbuild_factor:', overbuild_factor)
        print('threshold:', threshold)
        print('prop_wind:', prop_wind)


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

        # Plot outage lengths
        plt.figure(figsize=(4, 3))
        plt.grid(True, linestyle='--', alpha=0.7)
        outage_bins = np.arange(0, 55, 5)  # Bins will be [0, 5, 10, ..., 50]
        plt.hist(filtered_list, bins=outage_bins, edgecolor='black', color=colors[0])


        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        # Force integer ticks on y-axis
        plt.gca().yaxis.set_major_locator(MaxNLocator(5, integer=True))

        plt.xticks(fontsize=fsize)
        # Divide x-ticks by 340
        plt.yticks(fontsize=fsize)

        #plt.ylim([0, 15])
        plt.xlim([0, 50])
        plt.axvline(x=24, linestyle='--', color='black')
        plt.axvline(x=24*2, linestyle='--', color='black')
        plt.xlabel('Outage Duration (hrs.)', fontsize = fsize)
        plt.ylabel('# of Outages', fontsize=fsize)
        #plt.title('Histogram of Length of Repeated Zeros')
        plt.tight_layout(pad=1.5)
        plt.savefig(f'{folder_to_save}/storage_cost_{int(storage_capital_cost)}_gas_pc_{int(percentage_of_gas_allowed)}_outage_length.pdf', dpi=300)
        #plt.show()


        supply_demand_percent = (list_of_total_supply / demand) * 100 #what fraction of demand is met
        supply_demand_percent = supply_demand_percent[supply_demand_percent < 100] # only ineteested in unmet demand

        plt.figure(figsize=(4, 3))
        plt.grid(True, linestyle='--', alpha=0.7)
        prop_demand_bins = np.arange(50, 105, 5)
        plt.hist(supply_demand_percent, bins=prop_demand_bins, edgecolor='black')

        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)

        plt.xticks(fontsize=fsize)
        # Divide x-ticks by 340
        plt.yticks(fontsize=fsize)
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))  # Apply max 5 ticks


        #plt.ylim([0, 15])
        plt.xlabel('% of Demand Met\n During Outage', fontsize = fsize)
        plt.xlim([45, 105])
        plt.ylabel('# of Hours', fontsize=fsize)
        #plt.title('Histogram of Length of Repeated Zeros')
        plt.tight_layout(pad=1.5)
        plt.savefig(f'{folder_to_save}/storage_cost_{int(storage_capital_cost)}_gas_pc_{int(percentage_of_gas_allowed)}_prop_of_demand_met.pdf', dpi=300)
        #plt.show()


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

        cost_values = [wind_cost_value, solar_cost_value, storage_cost_value, gas_cost_value] # from original C simulation


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
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))  # Apply max 5 ticks


        plt.ylabel('Annual Cost (billions of $)', fontsize=fsize)

        plt.title('$' + str(int(sum(cost_values))) + ' billion', fontsize=fsize)

        plt.tight_layout(pad=1.5)
        plt.savefig(f'{folder_to_save}/storage_cost_{int(storage_capital_cost)}_gas_pc_{int(percentage_of_gas_allowed)}_costs.pdf', dpi=300)
        #plt.show()





