# This code runs a simulation of a hypothetical European electrical grid with renewables, storage, and backup.
# The script is based on an original Python script by Jamie Dunsmore, but has been modified to call a C function through the ctypes library, and to use xarray for data storage.
# The C function is compiled from the sim.c script.

# Imports
import numpy as np
from datetime import datetime
import pandas as pd
import xarray as xr
import os

# Import custom C library
from ctypes import *
so_file = "./src/eurogridsim.so"
eurogridsim = CDLL(so_file)

############################################################################################
# THIS SECTION SHOULD BE MODIFIED FOR EACH RUN
runtype = 'area_avg' # 'ninja' or 'area_avg'
grid_name = 'dense_grid_2_final_solar' # Define the output path (This should be changed for each run)

# Note that this is now a percentage of peak demand
renewable_allocation = False # This is a flag to toggle whether we allocate the renewable generation to the sunniest and windiest locations
backup_capacities = np.linspace(0, 0.7, 15) # every 0.05
storage_capacities = np.linspace(0, 7000, 36) # every 200
overbuild_factors = np.linspace(3, 9, 31) # every 0.2
thresholds = np.linspace(0, 0.5, 11) # every 0.05
prop_winds = np.linspace(0.5, 0.9, 21) # every 0.02

############################################################################################

# area_avg branch
if runtype == 'area_avg':
    df = pd.read_csv('./data/final_solar_total_supply_and_demand.csv')
    # Constant arrays
    demand = df['Total Demand (GW)'].values
    peak_demand = np.max(demand)
    norm_solar = df['Norm Solar (GW)'].values
    norm_wind = df['Norm Wind (GW)'].values

# ninja branch (2022 normalisation)
elif runtype == 'ninja':
    df = pd.read_csv('./data/ninja_total_supply_and_demand.csv')
    # Constant arrays
    demand = df['Total Demand (GW)'].values
    peak_demand = np.max(demand)
    norm_solar = df['Solar 2022 Weighted (GW)'].values
    norm_wind = df['Wind 2022 Weighted (GW)'].values

else:
    raise ValueError("Invalid runtype specified. Please choose 'ninja' or 'area_avg'.")

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

# Create the output directory if it doesn't exist
output_path = './output/' + grid_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(output_path + 'figures')
    os.makedirs(output_path + 'scripts')

if renewable_allocation:
    # If we have allocated the renewable generation to the sunniest and windiest locations, we scale up the generation
    norm_solar = norm_solar * 1.57
    norm_wind = norm_wind * 2.45

data = np.zeros((len(backup_capacities), len(storage_capacities), len(overbuild_factors), len(thresholds), len(prop_winds), 2))

coords = {'backup_capacity': backup_capacities, 'storage_capacity': storage_capacities, 'overbuild_factor': overbuild_factors, 'threshold': thresholds, 'prop_wind': prop_winds, 'variable': ['lost_hours', 'backup_used']}

dims = ('backup_capacity', 'storage_capacity', 'overbuild_factor', 'threshold', 'prop_wind', 'variable')

da = xr.DataArray(data=data, coords=coords, dims=dims)

print("Data array constructed!")

length = len(demand)

start_time = datetime.now() # Record the start time

# Next we perform the iteration over the parameters and run the simulation for each combination
print("Running simulations...")
for i in range(len(backup_capacities)):
    print('Percentage Complete: ', i/len(backup_capacities)*100)
    for j in range(len(storage_capacities)):
        for k in range(len(overbuild_factors)):
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
da.to_netcdf(output_path + 'results.nc')
print("Results saved to output directory!")