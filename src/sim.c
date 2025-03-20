// This file contains the C code for the run_timestep and run_simulation functions, which are called by the Python code, as well as helper functions and structs. The functions are based on original Python code written by Jamie Dunsmore. 

// Defines a data structure to hold the results of a timestep
typedef struct {
    double storage_end;
    double real_supply;
    double backup_used;
} TimestepResult;

// Defines a data structure to hold the results of a simulation run
typedef struct {
    double lost_hours;
    double backup_used;
} RunResult;

// Returns the minimum of two doubles
double min(double a, double b) {
    return a < b ? a : b;
}

// Runs a timestep of the simulation
TimestepResult run_timestep(double renewable_supply, double storage_start, double threshold, double backup, double total_storage_capacity, double demand) {
    // Create a TimestepResult object to hold the results and initialize it
    TimestepResult result;
    result.storage_end = storage_start;
    result.backup_used = 0;
    double threshold_capacity = total_storage_capacity * threshold;
    if (storage_start > threshold_capacity) {
        // storage above threshold
        if (renewable_supply < demand) {
            // renewables alone can't meet demand
            if (storage_start + renewable_supply >= demand) {
                // renewables + storage can meet demand
                result.real_supply = demand;
                result.storage_end -= (demand - renewable_supply);
            } else if (storage_start + renewable_supply + backup >= demand) {
                // renewables + storage + backup can meet demand
                result.storage_end = 0;
                result.real_supply = demand;
                result.backup_used = demand - storage_start - renewable_supply; 
            } else {
                // renewables + storage + backup can't meet demand
                result.real_supply = renewable_supply + storage_start + backup;
                result.storage_end = 0;
                result.backup_used = backup;
            }
        } else {
            // Renewables alone can meet demand, top up storage
            result.real_supply = demand;
            result.storage_end += renewable_supply - demand;
            result.storage_end = min(total_storage_capacity, result.storage_end);
        }
    } else {
        // storage below threshold
        if (renewable_supply < demand) {
            if (backup + renewable_supply >= demand) { 
                // renewables + backup can meet demand
                result.real_supply = demand;
                result.storage_end += (backup + renewable_supply - demand);
                if (result.storage_end > threshold_capacity) {
                    result.backup_used = backup - (result.storage_end - threshold_capacity);
                    result.storage_end = threshold_capacity;
                } else {
                    result.backup_used = backup;
                }

            } else if (storage_start + renewable_supply + backup >= demand) {
                // renewables + storage + backup can meet demand
                result.real_supply = demand;
                result.backup_used = backup;
                result.storage_end -= (demand - renewable_supply - backup);

            } else {
                // renewables + storage + backup can't meet demand
                result.real_supply = renewable_supply + storage_start + backup;
                result.storage_end = 0;
                result.backup_used = backup;
            }
        } else {
            // renewables alone can meet demand
            result.real_supply = demand;
            if (storage_start + renewable_supply - demand > threshold_capacity) {
                result.storage_end += (renewable_supply - demand);
                result.storage_end = min(total_storage_capacity, result.storage_end);
            } else {
                result.storage_end += (renewable_supply + backup - demand);
                if (result.storage_end > threshold_capacity) {
                    result.backup_used = backup - (result.storage_end - threshold_capacity);
                    result.storage_end = threshold_capacity;
                } else {
                    result.backup_used = backup;
                }
            }
        }
    }
    return result;
}

// function that loops over time steps and stores the RunResult
RunResult run_simulation(double *renewable_supply, double storage_start, double threshold, double backup, double total_storage_capacity, double *demand, int n_timesteps) {
    RunResult result;
    result.lost_hours = 0;
    result.backup_used = 0;
    double backup_used_result = 0;

    for (int i = 0; i < n_timesteps; i++) {
        TimestepResult timestep_result = run_timestep(renewable_supply[i], storage_start, threshold, backup, total_storage_capacity, demand[i]);

        backup_used_result += timestep_result.backup_used;

        if (timestep_result.real_supply < demand[i]) {
            result.lost_hours++;
        }

        storage_start = timestep_result.storage_end;
    }

    result.lost_hours /= (n_timesteps / 8760);
    result.backup_used = backup_used_result / (n_timesteps / 8760);

    return result;
}