'''
Recreate the Fig 5 plot from my initial 22.16 paper.
To switch between ninja data and original data:
1) change the supply/demand file that is being read
3) change the save_file paths 
'''

# Imports
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Import custom C library
from ctypes import *
so_file = "./src/eurogridsim.so"
eurogridsim = CDLL(so_file)


# 1) CHANGE THE SUPPLY/DEMAND FILE HERE.
#    ALSO NEED TO SPECIFY WHICH NORMALISATION TO USE IF USING NINJA DATA.
#    I.E WEIGHTING BY 2016 INSTALLED CAPACITY, 2022 INSTALLED CAPACITY OR AREA-WEIGHTED

data_type = 'area_avg' # 'ninja' or 'area_avg'

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
    raise ValueError("Invalid data type specified. Please choose 'ninja' or 'area_avg'.")


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
storage_capacities = np.zeros(1)
overbuild_factors = np.linspace(1, 11, 6)
overbuild_factors = np.array([1, 3, 5, 7, 10])
thresholds = np.zeros(1)
prop_winds = np.linspace(0, 1, 101) # as a fraction

data = np.zeros((len(backup_capacities), len(storage_capacities), len(overbuild_factors), len(thresholds), len(prop_winds), 2))

coords = {'backup_capacity': backup_capacities, 'storage_capacity': storage_capacities, 'overbuild_factor': overbuild_factors, 'threshold': thresholds, 'prop_wind': prop_winds, 'variable': ['lost_hours', 'backup_used']}

dims = ('backup_capacity', 'storage_capacity', 'overbuild_factor', 'threshold', 'prop_wind', 'variable')

da = xr.DataArray(data=data, coords=coords, dims=dims)

print("Data array constructed!")

length = len(demand)

start_time = datetime.now() # Record the start time

# Next we perform the iteration over the parameters and run the simulation for each combination
# This can be accelerated and parallelized in the future (I got fed up with type conversion with ctypes and Ufuncs and decided to keep it simple for now)
print("Running simulations...")
for i in range(len(backup_capacities)):
    for j in range(len(storage_capacities)):
        for k in range(len(overbuild_factors)):
            for l in range(len(thresholds)):
                for m in range(len(prop_winds)):
                    print(m)
                    prop_wind = prop_winds[m]
                    prop_solar = 1-prop_wind
                    solar_supply = norm_solar * prop_solar * overbuild_factors[k]
                    wind_supply = norm_wind * prop_wind * overbuild_factors[k]
                    renewable_supply = solar_supply + wind_supply
                    initial_storage = storage_capacities[j] / 2

                    result = eurogridsim.run_simulation(renewable_supply.ctypes.data_as(POINTER(c_double)), c_double(initial_storage), c_double(thresholds[l]), c_double(backup_capacities[i]*peak_demand), c_double(storage_capacities[j]), demand.ctypes.data_as(POINTER(c_double)), c_int(length))

                    da[i, j, k, l, m, 0] = result.lost_hours
                    da[i, j, k, l, m, 1] = result.backup_used

# Record the end time
end_time = datetime.now()
time_taken = end_time - start_time
print(f"Simulations complete!")
print(f"Runtime: {time_taken}")

# plot reliability vs % of wind in the generation mix
overbuilds = da.overbuild_factor.values
proportion_of_wind = da.prop_wind.values
backup_capacity = 0
storage_capacity = 0
threshold = 0

fsize = 16

plt.figure(figsize=(8, 6))

# Define a color palette for the lines
colors = plt.cm.plasma(np.linspace(0, 0.8, len(overbuilds)))

for idx, overbuild in enumerate(overbuilds):
    lost_hours = da.sel(backup_capacity=backup_capacity, storage_capacity=storage_capacity, threshold=threshold, overbuild_factor=overbuild, variable='lost_hours').values
    reliability = 100 * (8760 - lost_hours) / 8760     # I think lost hours are reported in hours per year
    plt.plot(proportion_of_wind*100, reliability, label=f' x{int(overbuild)}', color=colors[idx], linewidth=3)

#plt.axvline(x=77, linestyle='--', color='gray', label='77% Wind', linewidth=3)

# Customize the plot
plt.xlabel(r'% of Wind in Generation Mix', fontsize=fsize)
plt.ylabel('Grid Reliability (%)', fontsize=fsize)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)
plt.xlim([0,100])

plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title = 'Overbuild Factor', title_fontsize = fsize, fontsize=fsize, loc='upper left')

# Add a background color for better contrast
# plt.gca().set_facecolor('#f4f4f4')

plt.tight_layout()

plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines
plt.savefig(f'./figure_templates/figures/{save_folder}/reliability_vs_prop_wind.pdf')  # Save the plot
plt.show()


