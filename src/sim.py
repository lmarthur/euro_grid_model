# Code by Jamie Dunsmore, MIT Nuclear Science and Engineering
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.ticker as ticker



def run_timestep(renewable_supply, storage_start, threshold_frac, backup, total_storage_capacity, demand):
    '''
    Takes in current renewable generation, current storage levels and current demand at each timestep,
    as well as the storage threshold fraction, avaliable backup and total storage capacity. 
    Performs the simulation to determine the storage levels at the end of the timestep
    as well as the total supply (renewables plus dispatched storage plus backup)
    and the amount of backup used at the timestep.
    '''
    threshold_capacity = total_storage_capacity * threshold_frac #converts the fraction to GWh

    storage_end = storage_start #just initialise
    backup_used = 0

    if storage_start > threshold_capacity:
        if renewable_supply < demand:
            #renewables alone cannot meet demand
            if storage_start + renewable_supply >= demand:
                #demand can be met using renewables and storage
                real_supply = demand
                storage_end -= (demand - renewable_supply)
                # storage may drop below the threshold, but I think that's fine

            elif storage_start + renewable_supply + backup >= demand:
                # demand can be met using renewables, gas backup and storage. Deploy storage first.
                storage_end = 0
                real_supply = demand
                backup_used = demand - storage_start - renewable_supply
                # since we started the timestep with storage above the threshold, don't use gas backup to rechrage storage.
                # doesn't matter too much anyway because this is a real edge case. Shouldn't think too much about it.

            else:
                #demand cannot be met. Deploy all natural gas and empty the storage.
                real_supply = renewable_supply + storage_start + backup
                storage_end = 0
                backup_used = backup
        else:
            #renewables alone can meet demand. Use excess generation to top up storage tank if required.
            real_supply = demand
            storage_end += renewable_supply - demand
            storage_end = min(total_storage_capacity, storage_end) #just prevent overfilling of the storage tank


    else:
        if renewable_supply < demand:
            if backup + renewable_supply >= demand:
                #renewables plus gas backup can meet demand. Use excess nat gas to boost storage levels above the threshold.
                real_supply = demand
                storage_end += (backup + renewable_supply - demand)
                if storage_end > threshold_capacity:
                    backup_used = backup - (storage_end - threshold_capacity)
                    storage_end = threshold_capacity
                else:
                    backup_used = backup

            elif storage_start + renewable_supply + backup >= demand:
                #renewables plus gas plus storage can meet demand.
                real_supply = demand
                backup_used = backup
                storage_end -= (demand - renewable_supply - backup)

            else:
                #can't meet demand. Empty storage and use as all the nat gas.
                real_supply = renewable_supply + storage_start + backup
                storage_end = 0
                backup_used = backup
        else:
            #renewables can meet demand. Use excess generation and natural gas to raise storage levels above the threshold.
            real_supply = demand
            if storage_start + renewable_supply - demand > threshold_capacity:
                # renewables alone can fill the storage tank above threshold
                storage_end += (renewable_supply - demand)
                storage_end = min(total_storage_capacity, storage_end) #don't overfill the storage tank
            else:
                # use nat gas to help fill the storage tank above threshold
                storage_end += renewable_supply + backup - demand
                if storage_end > threshold_capacity:
                    # don't use more gas than is necessary to fill the tank to the threshold level
                    backup_used = backup - (storage_end - threshold_capacity)
                    storage_end = threshold_capacity
                else:
                    backup_used = backup

    return storage_end, real_supply, backup_used
