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
#adding ../eurogrid/ to the system path allows functions from other directories to be imported 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.cost_functions import calculate_total_annual_cost


da = xr.open_dataarray('../ninja_dense_grid_results_with_costs.nc')
supply_and_demand = pd.read_csv('../../../data/ninja_total_supply_and_demand.csv')

demand = supply_and_demand['Total Demand (GW)'].values

#norm_solar = df['Solar 2022 Weighted (GW)'].values
#norm_wind = df['Wind 2022 Weighted (GW)'].values


#2.63 is 99.97%
#8.76 is 99.9%
#87.6 is 99%


def correct_storage_cost(old_total_cost, total_capacity_in_kW, storage_o_and_m, old_overnight_capital_cost, new_overnight_capital_cost):
    '''
    Function to take in old storage costs, and return updated storage costs when the overnight capital cost is changed.
    Function relies on the assumption that the annual loan repayment is proportional to the overnight capital cost.

    INPUTS:
    -----------

    old_total_cost in billions of dollars
    total_capacity_in_kW in kW (NOTE NOT GW so a correction factor may have to be applied to storage capacities in the xr object)
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


# For testing the correction factors
value_by_coords = da.sel(backup_capacity=0.1, storage_capacity=1000, overbuild_factor=3.2, threshold=0.1, prop_wind=0.7, variable = 'storage_cost').values
print(value_by_coords)

da.loc[{'variable': 'storage_cost'}] = correct_storage_cost(da.sel(variable='storage_cost'), da['storage_capacity']*1e6, 0, 100, 50)

value_by_coords = da.sel(backup_capacity=0.1, storage_capacity=1000, overbuild_factor=3.2, threshold=0.1, prop_wind=0.6, variable = 'storage_cost').values
print(value_by_coords)


# recalculate the total cost based on the new storage costs
da.loc[{'variable': 'total_cost'}] = da.sel(variable='wind_cost') + da.sel(variable='solar_cost') + da.sel(variable='storage_cost') + da.sel(variable='gas_cost')


# TODO: finish writing this script so that it gives a similar plot to the reliability/gas ones,
# except with overnight storage capital cost on the x-axis.