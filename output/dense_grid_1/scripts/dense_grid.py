# This script is exactly the same as main.py in the src directory, except that the input 
# ranges to the simulation have been changed so that there is a higher density of configurations
# around the minimum cost solutions.

# Imports
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr

# Import custom C library
from ctypes import *
so_file = "../src/eurogridsim.so"
eurogridsim = CDLL(so_file)

# Load the dataset
df = pd.read_csv('../../../data/total_supply_and_demand.csv')

# Define the struct for the return value
class RunResult(Structure):
    _fields_ = [
        ("lost_hours", c_double),
        ("backup_used", c_double),
    ]

# Set the return type of the simulation function to the struct
eurogridsim.run_simulation.restype = RunResult

# Constant arrays
demand = df['Total Demand (GW)'].values
peak_demand = np.max(demand)
norm_solar = df['Norm Solar (GW)'].values
norm_wind = df['Norm Wind (GW)'].values

# Initialize the xarray data structure
print("Creating data array...")

# Note that this is now a percentage of peak demand
backup_capacities = np.linspace(0, 1, 21)
storage_capacities = np.linspace(0, 15000, 31)
overbuild_factors = np.linspace(3, 9, 31)
thresholds = np.linspace(0, 0.5, 11)
prop_winds = np.linspace(0.5, 0.9, 9) # used to have 21 points, but just for speed here

data = np.zeros((len(backup_capacities), len(storage_capacities), len(overbuild_factors), len(thresholds), len(prop_winds), 2))

coords = {'backup_capacity': backup_capacities, 'storage_capacity': storage_capacities, 'overbuild_factor': overbuild_factors, 'threshold': thresholds, 'prop_wind': prop_winds, 'variable': ['lost_hours', 'backup_used']}

dims = ('backup_capacity', 'storage_capacity', 'overbuild_factor', 'threshold', 'prop_wind', 'variable')

da = xr.DataArray(data=data, coords=coords, dims=dims)

print("Data array constructed!")

length = len(demand)

start_time = datetime.now() # Record the start time

# Next we perform the iteration over the parameters and run the simulation for each combination
# This could be accelerated and parallelised in the future
print("Running simulations...")
for i in range(len(backup_capacities)):
    print(f"Percentage complete: {i/len(backup_capacities)}")
    for j in range(len(storage_capacities)):
        print(f"Percentage complete 2: {j/len(storage_capacities)}")
        for k in range(len(overbuild_factors)):
            print(f"Percentage complete 3: {k/len(overbuild_factors)}")
            for l in range(len(thresholds)):
                for m in range(len(prop_winds)):
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

# Save the results to a netCDF file
da.to_netcdf('../dense_grid_results.nc')
print("Results saved to dense_grid_results.nc")
