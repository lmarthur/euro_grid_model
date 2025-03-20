Details about the dense_grid_1 simulation:

backup_capacities = np.linspace(0, 1, 21)

storage_capacities = np.linspace(0, 15000, 31)

overbuild_factors = np.linspace(3, 9, 31)

thresholds = np.linspace(0, 0.5, 11)

prop_winds = np.linspace(0.5, 0.9, 9)

These ranges were chosen so that they would contain the minimum cost solution for:

grid reliability varying from 99% to 100%

Nat gas varying from 0% to 5% ish

Storage overnight capital costs varying from $100 / kWh to $10 /kWh.
Note that the optimum storage level for $100 / kWh is always below 5000 GWh,
but typically lies between 10000 GWh and 15000 GWh for storage costs of $10 / kWh.

Took about 3 hours to run on J. Dunsmore's laptop.
More dense grid searches are definitely possible if we let the simulation run for longer,
and are also possible if we choose to focus in on storage at a specific storage price, rather than trying
to capture the minimum cost configuration for significant changes in the overnight capital cost of storage.
