This repository contains code for a simulation of the European electrical grid, based on assumed grid parameters and hourly demand data, along with weather data from the MERRA-2 dataset.

The code was initially written by Jamie Dunsmore, with some contributions from L.M. Arthur. The code is written in Python, with some functions rewritten in C for performance reasons. The simulation is designed to run on a single machine.

# Sources for the raw data
European electricity demand data is taken from the ENTSO platform: https://transparency.entsoe.eu

UK electricity demand data is taken from the National Grid ESO website: https://www.nationalgrideso.com/data-portal/historic-demand-data/historic_demand_data_2022

Weather data is taken from the NASA MERRA-2 dataset: https://disc.gsfc.nasa.gov/datasets?keywords=MERRA-2&page=1

In order to download MERRA-2 data files, follow the instructions here: https://disc.gsfc.nasa.gov/information/documents?title=Data%20Access

# Running the simulation
The modified code is in the ```/src``` folder. It has been directly adapted from the original code, with some functions rewritten in C to enhance performance. To run the code, you will need to have a C compiler installed on your machine, and have the ctypes library installed in your Python environment.

1. Once the repository is cloned, navigate to the ```/euro_grid``` directory in your terminal. This is important! If you are not in the project directory, the relative paths in the simulation will not work properly, and data may fail to import. Run the following command to compile the C code into a shared object file:

    ```cc -fPIC -shared -o src/eurogridsim.so src/sim.c```

2. To run the simulation, simply execute the ```main.py``` script in the ```/src``` directory. 

# Modifying the simulation
To modify the grid parameters or input data, it should be straightforward to modify the ```main.py``` script. To modify the grid operation procedures (the logic), you will need to modify the ```run_timestep``` function in the ```sim.c``` file. To modify the data output, you will need to modify the ```TimestepResult``` or ```RunResult``` structs in the ```sim.c``` file, as well as the ```RunResult``` class in the ```main.py``` file. 

# Configuration
The simulation is currently configured to take a set of grid parameters, hourly demand data, and weather data as input. The simulation then runs, and produces a set of output data, including grid reliability, carbon emissions, and (eventually) cost. 

## Independent variables
- Weather data
- Demand data
- Renewable generation capacity (overbuild factor)
- Relative weighting of wind and solar generation
- Storage capacity
- Dispatchable gas capacity
- Storage threshold
- Geographic distribution for wind and solar generation

## Dependent variables
- Grid reliability (fraction of time with power)
- Carbon emissions (from dispatchable gas generation)
- Cost

## MERRA-2 data used in the analysis
### From the tavg1_2d_slv_Nx dataset:
- U50M: eastward wind at 50 meters (m/s)
- V50M: northward wind at 50 meters (m/s)
- U10M: eastward wind at 10 meters (m/s)
- V10M: northward wind at 10 meters (m/s)
- PS: surface pressure (Pa)
- T2M: 2-meter air temperature (K)
### From the tavg1_2d_rad_Nx dataset:
- SWGDN: surface incoming shortwave flux (W/m^2)
- SWTDN: toa incoming shortwave flux (W/m^2)

# Note on Large Data Files
Some scripts in this repo read in files with large amounts of data. These include the ```generation.py``` and ```generation_grid.py``` scripts which read in 10s of GBs of MERRA2 data, and the ```create_post_2006_dataset.py``` script which reads in the >1GB ```solar_generation.npz``` and ```wind_generation.npz``` files. These files were too large to include in this repository. However the output files which would be created if the scripts were run with the full input files are included in the repository in all cases. 

Furthermore, a sample month of MERRA-2 data is included in the repository so that the functionality of ```generation.py``` and ```generation_grid.py``` can be understood.