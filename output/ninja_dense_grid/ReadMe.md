Details about the ninja_dense_grid simulation:

backup_capacities = np.linspace(0, 1, 21)

storage_capacities = np.linspace(0, 6000, 31)

overbuild_factors = np.linspace(3, 9, 31)

thresholds = np.linspace(0, 0.5, 11)

prop_winds = np.linspace(0.5, 0.9, 9)


This is the first attempted simulation with the renewables.ninja data rather than the old supply data.
It is the same parameters as dense_grid_1 except...

I plan on tweaking the cost estimates to make storage more expensive than $100 /kWh, so I've capped the upper limit on storage to 6000 rather than 15000.

I will run with the 2022 capacity weighted average first to get a feel for the results. Are they the same as before etc?

Expect it to take 3 hours to run.