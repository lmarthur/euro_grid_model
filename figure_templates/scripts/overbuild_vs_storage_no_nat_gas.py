'''
Plot the required overbuild factor as a function of storage capacity to reach a given reliability.
'''

# Imports
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.sim import run_timestep

# Import custom C library
from ctypes import *
so_file = "./src/eurogridsim.so"
eurogridsim = CDLL(so_file)


# 1) CHANGE THE SUPPLY/DEMAND FILE HERE. (ALSO NEED TO SPECIFY WHICH NORMALISATION TO USE IF USING NINJA DATA)

data_type = 'area_avg' # 'ninja' or 'area_avg' or 'area_avg'

if data_type == 'ninja':
    df = pd.read_csv('./data/ninja_total_supply_and_demand.csv')
    norm_wind = df['Wind 2022 Weighted (GW)'].values
    norm_solar = df['Solar 2022 Weighted (GW)'].values
    save_folder = 'ninja_2022_weighted'

elif data_type == 'area_avg':
    df = pd.read_csv('./data/final_solar_total_supply_and_demand.csv')
    norm_wind = df['Norm Wind (GW)'].values
    norm_solar = df['Norm Solar (GW)'].values
    save_folder = 'area_avg'

else:
    raise ValueError("Invalid data type specified. Please choose 'ninja' or 'area_avg' or 'area_avg.")


demand = df['Total Demand (GW)'].values
peak_demand = np.max(demand)


# Define the struct for the return value
class RunResult(Structure):
    _fields_ = [
        ("lost_hours", c_double),
        ("backup_used", c_double),
    ]

# Set the return type of the simulation function to the struct
eurogridsim.run_simulation.restype = RunResult


# Initialize the xarray data structure
print("Creating data array...")

# Note that this is now a percentage of peak demand
backup_capacities = np.zeros(1)
storage_capacities = np.arange(0, peak_demand*24.5, peak_demand/5) # GWh
#overbuild_factors = np.arange(4.4, 12, 0.2)
overbuild_factors = np.arange(4.4, 12, 0.02)
thresholds = np.zeros(1)
prop_winds = np.linspace(0.7, 1, 16) # as a fraction

# RELIABILITY STANDARD
LOLE_standards  = np.array([2.6, 8.76, 26.3]) # hours per year (99.97, 99.9, 99.7, 99.0, 95.0 per cent reliability)
reliability_standards = np.array([99.97, 99.9, 99])
line_colors = plt.cm.plasma(np.linspace(0, 0.6, 3))
fsize = 16

data = np.zeros((len(backup_capacities), len(storage_capacities), len(overbuild_factors), len(thresholds), len(prop_winds), 2))
coords = {'backup_capacity': backup_capacities, 'storage_capacity': storage_capacities, 'overbuild_factor': overbuild_factors, 'threshold': thresholds, 'prop_wind': prop_winds, 'variable': ['lost_hours', 'backup_used']}
dims = ('backup_capacity', 'storage_capacity', 'overbuild_factor', 'threshold', 'prop_wind', 'variable')
da = xr.DataArray(data=data, coords=coords, dims=dims)

print("Data array constructed!")

length = len(demand)

list_of_best_overbuilds = []
list_of_best_prop_winds = []

start_time = datetime.now() # Record the start time

# Cycle through every storage value and work out the minimum overbuild factor needed to meet the LOLE standard
# Optimise over solar/wind mix, but once a solution is found break the loop and move onto the next storage value
print("Running simulations...")
for LOLE_standard in LOLE_standards:
    print('LOLE Standard:', LOLE_standard)
    best_overbuilds = []
    for j in range(len(storage_capacities)):
        for k in range(len(overbuild_factors)):
            for m in range(len(prop_winds)):
                prop_wind = prop_winds[m]
                prop_solar = 1-prop_wind
                solar_supply = norm_solar * prop_solar * overbuild_factors[k]
                wind_supply = norm_wind * prop_wind * overbuild_factors[k]
                renewable_supply = solar_supply + wind_supply
                initial_storage = storage_capacities[j] / 2

                result = eurogridsim.run_simulation(renewable_supply.ctypes.data_as(POINTER(c_double)), c_double(initial_storage), c_double(thresholds[0]), c_double(backup_capacities[0]*peak_demand), c_double(storage_capacities[j]), demand.ctypes.data_as(POINTER(c_double)), c_int(length))

                if result.lost_hours < LOLE_standard: # We have reached the required reliability standard! Save this overbuild factor
                    print('SUCCESS')
                    print(f'Storage: {storage_capacities[j]}')
                    print(f'Overbuild: {overbuild_factors[k]}')
                    print(f'Prop Wind: {prop_winds[m]}')
                    best_overbuilds.append(overbuild_factors[k])
                    success = True
                    
                else:
                    success = False

                if success == True:
                    break
            if success == True:
                break
    list_of_best_overbuilds.append(best_overbuilds)


# Plot the results
plt.figure(figsize=(8, 6))

for idx in range(len(list_of_best_overbuilds)):
    plt.plot(storage_capacities/peak_demand, list_of_best_overbuilds[idx], label = f'{reliability_standards[idx]}% Reliability', color=line_colors[idx] ,linewidth=3)

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)


plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=fsize, loc = 'upper right')
plt.xlabel('Storage Capacity (hours of peak demand)', fontsize=fsize)
plt.ylabel(r'Renewable Overbuild Factor', fontsize=fsize)
#plt.gca().set_facecolor('#f4f4f4')
yticks = plt.yticks()[0]  
plt.yticks(yticks, [f"$\\times${int(y)}" for y in yticks])


plt.legend(fontsize=fsize, loc = 'upper right')
plt.tight_layout()
plt.savefig(f'./figure_templates/figures/{save_folder}/overbuild_vs_storage.pdf', dpi=300)  # Save the plot
plt.show()

'''
#####################
#####################
# Now do the storage trace plot for a specific case

# Define the parameters
storage_capacity = peak_demand * 24
overbuild_factor = 5
prop_wind = 0.78 #optimum proportion of wind is currently 0.78 for new solar area avg and 0.76 for ninja. 
threshold = 0
backup_capacity = 0


prop_solar = 1-prop_wind
solar_supply = norm_solar * prop_solar * overbuild_factor
wind_supply = norm_wind * prop_wind * overbuild_factor
renewable_supply = solar_supply + wind_supply
number_of_timesteps = len(demand)
number_of_years = number_of_timesteps / 8760


list_of_storage_levels = np.zeros(number_of_timesteps + 1)
list_of_total_supply = np.zeros(number_of_timesteps)
list_of_backup_used = np.zeros(number_of_timesteps)

# run python version of the simulation because speed doesn't matter and I want to extract storage levels over time
list_of_storage_levels[0] = storage_capacity / 2
for i in range(number_of_timesteps):
    storage_end, total_supply, backup_used = run_timestep(renewable_supply[i], list_of_storage_levels[i], threshold, backup_capacity * peak_demand, storage_capacity, demand[i])
    list_of_storage_levels[i+1] = storage_end
    list_of_total_supply[i] = total_supply
    list_of_backup_used[i] = backup_used

list_of_storage_levels = list_of_storage_levels[:-1]



plt.figure(figsize=(8, 5))

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)

# Add a background color for better contrast
# plt.gca().set_facecolor('#f4f4f4')

xvalues = np.arange(0, number_of_timesteps, 1)
plt.plot(xvalues, list_of_storage_levels/peak_demand, label='Storage Level', color='blue')
for i in range(len(xvalues)+1):
    if i%8760 == 0:
        plt.axvline(x=i, color='black', linestyle='--', alpha=0.5)

plt.axhline(y=0, color='Red', linestyle='-', alpha=0.5)


custom_xlabel_positions = [(8760/2)*i for i in range(0, int(2*number_of_years))]
custom_xlabels = []
for i in range(int(number_of_years)):
    custom_xlabels.append('Jan')
    custom_xlabels.append('July')

custom_xlabel_positions.append(8760)
custom_xlabels.append('Jan')



plt.xticks(custom_xlabel_positions, custom_xlabels, fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlim([8760*39.5, 8760*43.1])
#plt.ylim([-0.5, 16])

plt.ylabel('Hours of Storage Remaining', fontsize=fsize)

plt.legend(fontsize=fsize, loc = 'lower left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'./figure_templates/figures/{save_folder}/storage_trace_plot.pdf', dpi=300)  # Save the plot
plt.show()

'''

