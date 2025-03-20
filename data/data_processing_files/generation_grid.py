'''
Exactly the same as create_full_generation_data.py, except that the returned files
contain generation data at every MERRA-2 grid square, rather than the average over all grid squares.
Returns the data in the data_at_every_grid_point folder as four np arrays:

coords.npz (2044 tuples)
countries.npz (2044 strings of country names)
solar_generation.npz (2044 lists. Each list contains solar generation at every hour)
wind_generation.npz (2044 lists. Each list contains wind generation at every hour)

NOTE: some of these files are too large to be stored in the repo. 
However, the resulting files from any scripts which depend on them are included in the repo.
'''

import pandas as pd
import numpy as np
import os
import xarray as xr
import time as t


def get_wind_power(data):
    # calculate wind speed
    wind_speed_50m = np.sqrt(data['U50M']**2 + data['V50M']**2)
    wind_speed_10m = np.sqrt(data['U10M']**2 + data['V10M']**2)

    rho_surface = (data['PS'] * M_a) / (R * data['T2M']) #calculate surface air density
    rho_hub_height = rho_surface * np.exp(- (g * M_a * hub_height) / (R * data['T2M']) ) #calculate air density at hub height
    hellman_exponent = np.log(wind_speed_50m / wind_speed_10m) / np.log(5) #calculate the Hellman exponent
    wind_speed_100m = wind_speed_50m * np.power(2, hellman_exponent) #extrapolate the wind speed to 100m using Hellman exponent
    wind_power = 0.5 * wind_turbine_efficiency * rho_hub_height * area * (wind_speed_100m**3) #calculate the wind power output

    # Wind power is set to zero above and below the operating speeds.
    # And it is limited to the rated power (cannot generate more than this).
    wind_power = np.where(wind_speed_100m > 25, 0, wind_power)
    wind_power = np.where(wind_speed_100m < 3, 0, wind_power)
    wind_power = np.where(wind_power > power_rating, power_rating, wind_power)
    return wind_power


def get_solar_power(data):

    # calculate the solar power for all locations in one step (no iteration)
    solar_power = np.zeros(data['SWGDN'].shape)

    data['tilt'] = 0.76 * data['lat'] + 3.1 # this is the optimal tilt for solar panels between 25 and 50 degrees latitude
    data['tilt'], _ = xr.broadcast(data['tilt'], data['lon'])
    data['tilt'], _ = xr.broadcast(data['tilt'], data['time'])

    # calculations to get effective area of solar panel
    data['solar_time'] = (data['lon'] / 15) + (data['time'].dt.hour + data['time'].dt.minute / 60)
    data['solar_time'], _ = xr.broadcast(data['solar_time'], data['lat'])  # broadcasting along the 'lat' dimension

    data['sun_azi'] = 15 * data['solar_time'] # convention is that azimuth is 180 degrees at midday at the greenwich meridian.
    data['day_of_year'] = data['time'].dt.dayofyear
    data['peak_solar_alt'] = 90 - data['lat'] - 23.5 * np.cos(np.radians(360 * (data['day_of_year'] + 10) / 365))
    data['sun_alt'] = - data['peak_solar_alt'] * np.cos(np.pi*data['solar_time']/12) # this is 90 - zenith angle
    data['sun_alt'] = (data['sun_alt'].transpose('time', 'lat', 'lon'))



    # AOI projection is the dot product of the sun vector and the normal vector of the panel (cosine of the angle of incidence)
    data['AOI_projection'] = np.maximum(0, np.sin(np.radians(data['sun_alt'])) * np.cos(np.radians(data['tilt'])) + np.cos(np.radians(data['sun_alt'])) * np.sin(np.radians(data['tilt'])) * np.cos(np.radians(180 - data['sun_azi'])))

    # Aeff currently has dimensions of (lat, time, lon) which is not consistent with everything else. Switch to (time, lat, lon)
    data['AOI_projection'] = (data['AOI_projection'].transpose('time', 'lat', 'lon'))

    # BRL MODEL for diffuse horizontal irradiance
    dims = data['SWGDN'].dims
    coords = data['SWGDN'].coords
    clearness_index = np.where(data['SWTDN'] != 0, data['SWGDN'] / data['SWTDN'], 0)
    data['clearness_index'] = xr.DataArray(clearness_index, coords=coords, dims=dims)
    data['diffuse_irrad'] = data['SWGDN'] / (1 + np.exp(-5.0033 + 8.6025 * data['clearness_index']))


    # Get the direct normal irradiance (irrad along the path of the sun)
    dni = np.where(
        data['sun_alt'] > 3,
        (data['SWGDN'] - data['diffuse_irrad']) / np.sin(np.radians(data['sun_alt'])),
        0
    )
    data['DNI'] = xr.DataArray(dni, coords=coords, dims=dims)
    data['direct_irrad_inplane'] = data['DNI'] * data['AOI_projection'] # AOI_projection is cosine of the angle

    # diffuse irrad in-plane (ASSUME AN ISOTROPIC SKY) - SEE SANDIA pvlib documentation
    data['diffuse_irrad_inplane'] = data['diffuse_irrad'] * (1 + np.cos(np.radians(data['tilt'])))/2

    # ground reflected in-plane irrad  - SEE SANDIA pvlib documentation
    data['ground_reflected_irrad_inplane'] = albedo * data['SWGDN'] * (1 - np.cos(np.radians(data['tilt'])))/2


    data['in_plane_irrad'] = data['direct_irrad_inplane'] + data['diffuse_irrad_inplane'] + data['ground_reflected_irrad_inplane']
    data['normalized_irrad'] = data['in_plane_irrad'] / ref_irrad

    data['T2M in Celsius'] = data['T2M'] - 273.15
    data['T_panel'] = data['T2M in Celsius'] + (data['in_plane_irrad'] * 0.035) - 25

    # Calculate relative efficiency using np.where. This is necessary because often normalized_irrad is zero, throwing a NaN in the relative efficiency calculation.
    eff_rel = np.where(
        data['normalized_irrad'] > 0, 
        np.maximum(0, 1 + k_1 * np.log(data['normalized_irrad']) + 
                    k_2 * np.log(data['normalized_irrad'])**2 + 
                    data['T_panel'] * (k_3 + k_4 * np.log(data['normalized_irrad']) + 
                                        k_5 * np.log(data['normalized_irrad'])**2 + 
                                        k_6 * data['T_panel'])), 
        0  # Return relative efficiency of zero when normalized irradiance is zero or less
    )

    data['eff_rel'] = xr.DataArray(eff_rel, dims=['time', 'lat', 'lon'], coords=data['normalized_irrad'].coords)
    
    solar_power = data['eff_rel'] * 0.21 * data['in_plane_irrad'] # 21% is typical efficiency of a solar panel.

    return solar_power


#tavg1_2d_slv_Nx
#U50M is the eastward wind at 50 meters
#V50M is the northward wind at 50 meters
#U10M is the eastward wind at 10 meters
#V10M is the northward wind at 10 meters
#PS surface pressure in Pa
#T2M 2-meter air temperature

#tavg1_2d_rad_Nx
#SWGDN is the shortwave incident radiation

# use process_time to measure the time taken
start_time = t.process_time()

# SET PARAMETERS
solar_efficiency = 0.21 
wind_turbine_efficiency = 0.4
rotor_diameter = 110
area = np.pi * ((rotor_diameter / 2) ** 2)
hub_height = 100
power_rating = 4.1e6 #onshore only. Occurs at ~12m/s wind speed
cut_in = 3
cut_out = 25

# albedo of the ground
albedo = 0.3

# Molar Mass of Air in kg/mol
M_a = 0.028964

# universal gas constant in NM / (mol K)
R = 8.3144598

# acceleration due to gravity in m/s^2
g = 9.80665

# panel temperature coefficient of irradiance in W m^-2
coeff_irrad = 0.035

# panel temperature coefficients for c-Si panels
ref_T_mod = 25 # Celsius
ref_irrad = 1000 # W/m^2

k_1 = -0.017162
k_2 = -0.040289
k_3 = -0.004681
k_4 = 0.000148
k_5 = 0.000169
k_6 = 0.000005

print("Opening datasets...")
ds_wind = xr.open_dataset('./data/data_processing_files/slv_data/MERRA2_100.tavg1_2d_slv_Nx.19800101.SUB.nc')
ds_solar = xr.open_dataset('./data/data_processing_files/rad_data/MERRA2_100.tavg1_2d_rad_Nx.19800101.SUB.nc')
ds_countries = xr.open_dataset('./data/data_processing_files/country_coords.nc')

coords = np.load('./data/data_processing_files/coords.npz')
coord_list = coords['arr_0']
lon = coord_list[:, 0]
lat = coord_list[:, 1]


print("Datasets opened.")

lon_min, lon_max = -25, 55
lat_min, lat_max = 35, 75

# Subset the data
ds_solar = ds_solar.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
ds_wind = ds_wind.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
ds_countries = ds_countries.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

ds = xr.merge([ds_wind, ds_solar])

time_values = ds_solar['time'].values  # Use the time values from the 3D dataset
ds_countries_3d = ds_countries.expand_dims(time=time_values)
ds = xr.merge([ds, ds_countries_3d])

# reduce the size of the dataset so that any lon/lat values where there are no European countries are removed
print("Removing non-European countries...")
countries_mask = (ds_countries_3d != '')
for variable in ds:
    countries_mask[variable] = countries_mask['country']
ds = ds.where(countries_mask, drop=True)
print("Non-European countries removed.")

# Initialize the list of coordinates and countries
list_of_coords = []
list_of_countries = []
# Iterate over the longitude and latitude values
for lon in ds_countries.lon.values:
    for lat in ds_countries.lat.values:
        country = ds_countries.sel(lon=lon, lat=lat, method='nearest')['country'].values
        if country != '':
            list_of_coords.append((lon, lat))
            list_of_countries.append(country)
print("Coordinates and countries stored.")
print("Time taken: ", t.process_time() - start_time)


#cycle over every file
start_date = '1980-01-01'
#end_date = '1980-01-03'
end_date = '2022-12-31'
#end_date = '1981-01-01'

# Generate a date range
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Filter out the dates on leap years and the 29th February
filtered_dates = [date.strftime('%Y%m%d') for date in date_range if not (date.year % 4 == 0 and date.month == 2 and date.day == 29)]

lon = ds['lon'].values  # Use the time values from the 3D dataset
lat = ds['lat'].values  # Use the time values from the 3D dataset
time = np.arange(24*len(filtered_dates))

# Initialise the full dataset
print("Initialising full dataset...")
large_ds = xr.Dataset({
    'Wind Power': (['time', 'lat', 'lon'], np.empty((len(time), len(lat), len(lon)))),
    'Solar Power': (['time', 'lat', 'lon'], np.empty((len(time), len(lat), len(lon)))),
    'country': (['time', 'lat', 'lon'], np.empty((len(time), len(lat), len(lon)), dtype=object))
}, coords={'time': time, 'lat': lat, 'lon': lon})

print("Beginning data processing...")
# print elapsed time
print("Time taken: ", t.process_time() - start_time)
# Iterate through each date in the range
for i, date in enumerate(filtered_dates):
    if date[-4:] == '0101':
        print(date)    # Format the date convention

    ### NOTE: REVISE THESE FILE PATHS

    matching_solar_files = [file for file in os.listdir('./data/data_processing_files/rad_data/') if date in file]
    matching_wind_files = [file for file in os.listdir('./data/data_processing_files/slv_data') if date in file]
    solar_file = matching_solar_files[0]
    wind_file = matching_wind_files[0]
    solar_file_path = os.path.join('./data/data_processing_files/rad_data', solar_file)
    wind_file_path = os.path.join('./data/data_processing_files/slv_data', wind_file)

    ds_wind = xr.open_dataset(wind_file_path)
    ds_solar = xr.open_dataset(solar_file_path)
    ds_countries = xr.open_dataset('./data/data_processing_files/country_coords.nc')
    

    lon_min, lon_max = -25, 55
    lat_min, lat_max = 35, 75

    # Subset the dataset based on longitude and latitude
    ds_solar = ds_solar.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

    ds = xr.merge([ds_wind, ds_solar])

    time_values = ds_solar['time'].values  # Use the time values from the 3D dataset

    ds_countries_3d = ds_countries.expand_dims(time=time_values)
    ds = xr.merge([ds, ds_countries_3d])
    countries_mask = (ds_countries_3d != '')
    for variable in ds:
        countries_mask[variable] = countries_mask['country']
    ds = ds.where(countries_mask, drop=True)

    # Calculate the solar power
    ds['Solar Power'] = get_solar_power(ds)

    coords = ds['Solar Power'].coords
    dims = ds['Solar Power'].dims
    wind_power = get_wind_power(ds)
    ds['Wind Power'] = xr.DataArray(wind_power, coords=coords, dims=dims)

    ds['country'] = ds['country'].transpose('time', 'lat', 'lon')

    # Add the data to the large dataset
    large_ds['Wind Power'][i*24:(i+1)*24, :, :] = ds['Wind Power'].values
    large_ds['Solar Power'][i*24:(i+1)*24, :, :] = ds['Solar Power'].values
    large_ds['country'][i*24:(i+1)*24, :, :] = ds['country'].values

print("Data processing complete.")
print("Time taken: ", t.process_time() - start_time)

# Initialize NumPy arrays to store the solar and wind power data
num_coords = len(list_of_coords)
num_time_steps = len(large_ds.time)

huge_array_of_solar = np.zeros((num_coords, num_time_steps))
huge_array_of_wind = np.zeros((num_coords, num_time_steps))

print("Time: ", t.process_time())
print("Beginning data storage...")
# Iterate over the coordinates and store the data in the arrays
for i, (lon, lat) in enumerate(list_of_coords):
    pc_complete = (i / num_coords) * 100
    print('Time: ', t.process_time())
    print(f"Progress: {pc_complete:.2f}%")
    huge_array_of_solar[i, :] = large_ds.sel(lon=lon, lat=lat, method='nearest')['Solar Power'].values
    huge_array_of_wind[i, :] = large_ds.sel(lon=lon, lat=lat, method='nearest')['Wind Power'].values

list_of_countries = np.array(list_of_countries)
list_of_coords = np.array(list_of_coords)


# Save the data with compression
np.savez_compressed('.data/data_processing_files/solar_generation.npz', huge_array_of_solar)
np.savez_compressed('.data/data_processing_files/wind_generation.npz', huge_array_of_wind)
np.savez_compressed('.data/data_processing_files/coords.npz', list_of_coords)
np.savez_compressed('.data/data_processing_files/countries.npz', list_of_countries)

print("Data storage complete.")

# END TIMER
end_time = t.process_time()
print("Time taken: ", end_time - start_time)