# Contains functions to calculate the cost of a given grid configuration.
# Can take in a previously run simulation and generate a new .nc file with the costs added.

from scipy.integrate import quad
import numpy as np
import time
import pandas as pd
import xarray as xr

# background assumptions about costs

#Transmission: EU has said 637 billion dollars needed by 2030. Assume 1 trillion by 2050. 

### SOLAR ###
solar_lifetime = 25 
# in years.
# From IEA. https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf  page 202.

solar_construction_time = 2 
# in years.
# No citation but in the ballpark of most online reports.

solar_overnight_cost = 790 
# dollars per kW. 
# From IEA. https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf  page 201. 

solar_o_and_m = 10 
# dollars per kW-year.
# IRENA document lit review: https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2021/Jun/IRENA_Power_Generation_Costs_2020.pdf?rev=c9e8dfcd1b2048e2b4d30fef671a5b84. p.81


### STORAGE ###
storage_lifetime = 15 
# in years.
# Highly cited NREL report. #https://www.nrel.gov/docs/fy23osti/85332.pdf page 8.

storage_construction_time = 2 
# in years
# No citation but in the ballpark of most online reports.

storage_overnight_cost = 200 
# dollars per kWh.
# Same NREL report as above. https://www.nrel.gov/docs/fy23osti/85332.pdf page iv.
# This shows their mid-cost projections for 4-hour lithium-ion battery capital costs for 2050.
# Current costs in the same report are about 400 dollars per kWh.

storage_o_and_m = 10
# dollars per kWh-year.
# Same NREL report as above. https://www.nrel.gov/docs/fy23osti/85332.pdf page 8.
# They have about 50 dollars per kW-year for a 4-hour system, which I think is about 10 dollars per kWh-year for a 4-hour system (to 1sf).
# They note that this is a high-end figure, which assumes some capacity additions/replacements to address degradation and maintain performance.


### WIND ###
wind_lifetime = 25 
# in years. 
# From IEA. https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf  page 202.

wind_construction_time = 2 
# in years.
# No citation but in the ballpark of things online. I don't think our calculations are that sensitive to the construction time anyway

wind_overnight_cost = 1540 
# dollars per kW. 
# From IEA. https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf  page 201.

wind_o_and_m = 40
# dollars per kW-year.
# From IRENA 202 report https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2021/Jun/IRENA_Power_Generation_Costs_2020.pdf?rev=c9e8dfcd1b2048e2b4d30fef671a5b84 p.81 Figure 2.8.
# 40 seems a reasonable estiamte from the graph on this page and the text.
# From IEA. https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf  page 201. 


### GAS ###
gas_lifetime = 30 
# in years
# Old NREL report (2000) https://www.nrel.gov/docs/fy00osti/27715.pdf p.8

gas_construction_time = 2 
# in years.
# Same old NREL report https://www.nrel.gov/docs/fy00osti/27715.pdf p.8

gas_overnight_cost = 1000 
# dollars per kW. 
# From IEA. https://iea.blob.core.windows.net/assets/deebef5d-0c34-4539-9d0c-10b13d840027/NetZeroby2050-ARoadmapfortheGlobalEnergySector_CORR.pdf  page 201.

gas_o_and_m = 20 #
# dollars per kW-year. 
# From recent EIA report. https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2025.pdf p.III Table 1-2.
# They report $ 15.51 per kW-year for a Combined Cycle 1x1x1 Turbine. $ 20 to 1sf.


### Transmission ###
# Keep these costs in here but remove them in the plotting scripts so that the paper plots just show generation costs.

transmission_lifetime = 50 
# in years.

transmission_construction_time = 10 
# in years.

transmission_overnight_cost = 1e12 
# dollars (1 trillion).


### FINANCE ASSUMPTIONS ###
inflation_rate = 0.04
nominal_interest_rate = 0.08
real_interest_rate = nominal_interest_rate - inflation_rate

### FUNCTIONS TO PERFORM THE COST CALCULATIONS ###

def correction_factor_integrand(t, construction_time, inflation_rate, nominal_interest_rate):
    '''
    returns the integral part of the overnight cost correction factor. Formula from 22.16 economics pset.
    '''

    return t * (construction_time - t) * np.exp(t * inflation_rate) * np.exp(nominal_interest_rate * (construction_time - t))

def correction_factor_to_overnight_cost(construction_time, inflation_rate, nominal_interest_rate):
    '''
    Calculates the correction factor to the overnight capital costs due to the fact that the facility cannot be built overnight.
    Just uses the formula from the 22.16 economics PSET.
    '''

    # Do the integration
    lower_limit = 0
    upper_limit = construction_time

    correction_factor_integral, error = quad(correction_factor_integrand, lower_limit, upper_limit, 
                         args=(construction_time, inflation_rate, nominal_interest_rate))
    

    correction_to_overnight = correction_factor_integral * 6 / (construction_time**3)
    return correction_to_overnight

def loan_required_for_construction(overnight_cost, construction_time, inflation_rate, nominal_interest_rate):
    '''
    Accounts for the time-value of money to calculate the initial loan required in REAL DOLLARS (i.e YEAR 0)
    to finance the construction of the project.
    '''

    correction_to_overnight = correction_factor_to_overnight_cost(construction_time, inflation_rate, nominal_interest_rate)
    mixed_dollar_cost = overnight_cost * correction_to_overnight
    real_dollar_cost = mixed_dollar_cost * np.exp( -(real_interest_rate) * construction_time)

    return real_dollar_cost


def loan_payments_per_year(total_loan, real_interest_rate, infrastructure_lifetime):
    '''
    Formula to calculate the payment needed every year to pay off the initial loan for construction, in real dollars.
    Takes in the total amount of money loaned, the real interest rate, and the lifetime of the infrastructure.
    '''

    annual_loan_payment = total_loan * \
                            ( 
                                (np.exp(real_interest_rate * infrastructure_lifetime) * (np.exp(real_interest_rate) - 1)) /
                                (np.exp(real_interest_rate * infrastructure_lifetime) - 1)
                                )
    return annual_loan_payment

def calculate_total_annual_cost(total_capacity, infrastructure_lifetime, construction_time, overnight_cost, o_and_m_costs,
                                inflation_rate, real_interest_rate, nominal_interest_rate):
    '''
    Main function to calculate the total annual cost for a given technology. Calls the other functions.
    NOTE
    total capacity should be in GW
    Infrastructure lifetime is in years
    Construction time is in years
    Overnight cost is in dollars / KW
    O and m costs are in dollars / kW
    Variable costs are assumed to be negligible
    '''

    construction_loan = loan_required_for_construction(overnight_cost, construction_time, inflation_rate, nominal_interest_rate)
    annual_loan_payment = loan_payments_per_year(construction_loan, real_interest_rate, infrastructure_lifetime)

    total_capacity_in_kW = total_capacity * 1e6 # convert from GW to kW to be consistent with cost calculations
    total_cost_per_year = (annual_loan_payment + o_and_m_costs) * total_capacity_in_kW # in dollars
    return total_cost_per_year / 1e9 # returns the annual cost in billions of dollars


### CHOOSE THE FILES TO READ AND WRITE ###

# 1) SUPPLY AND DEMAND DATA (TO GET PEAK DEMAND)
df = pd.read_csv('./data/final_solar_total_supply_and_demand.csv')
demand = df['Total Demand (GW)'].values
peak_demand = np.max(demand)

# 2) ORIGINAL SIMULATION RESULTS
da = xr.open_dataarray('./output/dense_grid_2_final_solar/results.nc')

backup_capacities = da.backup_capacity.values
storage_capacities = da.storage_capacity.values
overbuild_factors = da.overbuild_factor.values
thresholds = da.threshold.values
prop_winds = da.prop_wind.values

backup_capacities_GWh = peak_demand * backup_capacities  #GWh

# 3) PATH TO SAVE THE DATA
path_to_save = './output/dense_grid_2_final_solar/results_with_costs.nc'

# Initialise Data Array to store the cost results
data = np.zeros((len(backup_capacities), len(storage_capacities), len(overbuild_factors), len(prop_winds), len(thresholds), 6))
coords = {'backup_capacity': backup_capacities, 'storage_capacity': storage_capacities, 'overbuild_factor': overbuild_factors, 'prop_wind': prop_winds, 'threshold': thresholds, 'variable': ['total_cost', 'wind_cost', 'solar_cost', 'storage_cost', 'gas_cost', 'transmission_cost']}
dims = ('backup_capacity', 'storage_capacity', 'overbuild_factor', 'prop_wind', 'threshold', 'variable')
costs_da = xr.DataArray(data=data, coords=coords, dims=dims)

start = time.time()

#transmission costs (same for every scenario)
transmission_loan_required = loan_required_for_construction(transmission_overnight_cost, transmission_construction_time, 
                                                            inflation_rate, nominal_interest_rate)

transmission_annual_costs = loan_payments_per_year(transmission_loan_required, real_interest_rate, transmission_lifetime) / 1e9 #conert into billions

# cycle through parameters
for i in range(len(backup_capacities_GWh)):
    print('percentage complete: ', i/len(backup_capacities_GWh))
    gas_costs = calculate_total_annual_cost(backup_capacities_GWh[i], gas_lifetime, gas_construction_time, gas_overnight_cost, gas_o_and_m, 
                                         inflation_rate, real_interest_rate, nominal_interest_rate)

    for j in range(len(storage_capacities)):
        storage_costs = calculate_total_annual_cost(storage_capacities[j], storage_lifetime, storage_construction_time, storage_overnight_cost, storage_o_and_m, 
                                         inflation_rate, real_interest_rate, nominal_interest_rate)

        for k in range(len(overbuild_factors)):

            for l in range(len(prop_winds)):
                wind_capacity = peak_demand * overbuild_factors[k] * prop_winds[l]   #GWh
                solar_capacity = peak_demand * overbuild_factors[k] * (1 - prop_winds[l])  # GWh


                wind_costs = calculate_total_annual_cost(wind_capacity, wind_lifetime, wind_construction_time, wind_overnight_cost, wind_o_and_m, 
                                        inflation_rate, real_interest_rate, nominal_interest_rate)

                solar_costs = calculate_total_annual_cost(solar_capacity, solar_lifetime, solar_construction_time, solar_overnight_cost, solar_o_and_m, 
                                        inflation_rate, real_interest_rate, nominal_interest_rate)
                

                total_system_cost = wind_costs + solar_costs + storage_costs + gas_costs + transmission_annual_costs

                costs_da[i, j, k, l, :, 0] = total_system_cost
                costs_da[i, j, k, l, :, 1] = wind_costs
                costs_da[i, j, k, l, :, 2] = solar_costs
                costs_da[i, j, k, l, :, 3] = storage_costs
                costs_da[i, j, k, l, :, 4] = gas_costs
                costs_da[i, j, k, l, :, 5] = transmission_annual_costs


# Combine the new costs DataArray with the existing DataArray along the 'variable' dimension
combined_da = xr.concat([da, costs_da], dim='variable')

# Save the combined DataArray to a new netCDF file
combined_da = combined_da.drop_encoding() # this is required to stop 'transmission_cost' and 'storage_cost' from truncating
combined_da.to_netcdf(path_to_save)

end = time.time()
print('Time Taken: ', (end - start)*1000, 'ms')