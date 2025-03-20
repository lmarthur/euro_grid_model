import unittest

from ctypes import *
so_file = "./src/eurogridsim.so"
eurogridsim = CDLL(so_file)

class TimestepResult(Structure):
    _fields_ = [
        ("storage_end", c_double),
        ("real_supply", c_double),
        ("backup_used", c_double),
    ]

# Set the return type of the simulation function to the struct
eurogridsim.run_timestep.restype = TimestepResult

# Define a lightweight wrapper around the simulation function
def run_timestep(renewable_supply, storage_start, threshold_frac, backup, total_storage_capacity, demand):
    result = eurogridsim.run_timestep(
        c_double(renewable_supply),
        c_double(storage_start),
        c_double(threshold_frac),
        c_double(backup),
        c_double(total_storage_capacity),
        c_double(demand)
    )
    return result.storage_end, result.real_supply, result.backup_used

class TestRunTimestep(unittest.TestCase):

    ##### FIRST TEST CASES WHERE STORAGE IS ABOVE THRESHOLD #####

    # backup should not be used at all here
    def test_storage_above_threshold___renewables_and_storage_meet_demand(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=200,
            threshold_frac=0.99,
            backup=50,
            total_storage_capacity=200,
            demand=100
        )
        self.assertEqual(result, (150, 100, 0))

    # natgas is called in to meet demand. Since we start the timestep above the storage threshold, nat gas can only be used to meet demand and not to fill the storage.
    def test_storage_above_threshold___renewables_and_storage_and_gas_meet_demand(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=200,
            threshold_frac=0.1,
            backup=800,
            total_storage_capacity=1000,
            demand=500
        )
        self.assertEqual(result, (0, 500, 250))

    # use all renewables, nat gas and storage in this case
    def test_storage_above_threshold___cannot_meet_demand(self):
        result = run_timestep(
            renewable_supply=30,
            storage_start=300,
            threshold_frac=0.2,
            backup=50,
            total_storage_capacity=1000,
            demand=500
        )
        self.assertEqual(result, (0, 380, 50))

    # excess renewables should fill up the storage in this case
    def test_storage_above_threshold___renewables_meet_demand(self):
        result = run_timestep(
            renewable_supply=1500,
            storage_start=500,
            threshold_frac=0.1,
            backup=50,
            total_storage_capacity=1000,
            demand=80
        )
        self.assertEqual(result, (1000, 80, 0))

    ##### SECOND TEST CASES WHERE STORAGE IS BELOW THRESHOLD #####
        
    # use gas to recharge the storage, but only to the threshold
    def test_storage_below_threshold___renewables_plus_backup_meet_demand___apply_threshold_cutoff(self):
        result = run_timestep(
            renewable_supply=100,
            storage_start=500,
            threshold_frac=0.9,
            backup=10000,
            total_storage_capacity=1000,
            demand=200
        )
        self.assertEqual(result, (900, 200, 500))

    # use gas to recharge the storage. Don't need to worry about the threshold here
    def test_storage_below_threshold___renewables_plus_backup_meet_demand(self):
        result = run_timestep(
            renewable_supply=100,
            storage_start=100,
            threshold_frac=0.8,
            backup=300,
            total_storage_capacity=1000,
            demand=350
        )
        self.assertEqual(result, (150, 350, 300))

    # use gas first, then fill remaining gap with storage
    def test_storage_below_threshold___renewables_plus_storage_plus_backup_meet_demand(self):
        result = run_timestep(
            renewable_supply=20,
            storage_start=300,
            threshold_frac=0.8,
            backup=100,
            total_storage_capacity=500,
            demand=200
        )
        self.assertEqual(result, (220, 200, 100))

    # use everything
    def test_storage_below_threshold___cant_meet_demand(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=100,
            threshold_frac=0.8,
            backup=200,
            total_storage_capacity=1000,
            demand=500
        )
        self.assertEqual(result, (0, 350, 200))

    # use renewables. Don't overfill storage
    def test_storage_below_threshold___renewables_meet_demand___renewables_take_storage_over_threshold(self):
        result = run_timestep(
            renewable_supply=5000000,
            storage_start=100,
            threshold_frac=0.8,
            backup=200,
            total_storage_capacity=1000,
            demand=500
        )
        self.assertEqual(result, (1000, 500, 0))

    # use renewables. Since we started the timestep below the threshold, we can use gas to recharge the storage up to the threshold.
    def test_storage_below_threshold___renewables_meet_demand___apply_threshold_cutoff(self):
        result = run_timestep(
            renewable_supply=600,
            storage_start=100,
            threshold_frac=0.8,
            backup=3000,
            total_storage_capacity=1000,
            demand=500
        )
        self.assertEqual(result, (800, 500, 600))

    # use renewables. Since we started the timestep below the threshold, we can use gas to recharge the storage.
    def test_storage_below_threshold___renewables_meet_demand(self):
        result = run_timestep(
            renewable_supply=600,
            storage_start=100,
            threshold_frac=0.8,
            backup=200,
            total_storage_capacity=1000,
            demand=500
        )
        self.assertEqual(result, (400, 500, 200))

######################
##### EDGE CASES #####
######################

class TestRunTimestepEdgeCases(unittest.TestCase):

    # Test when threshold is 0
    def test_threshold_zero(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=100,
            threshold_frac=0,
            backup=50,
            total_storage_capacity=200,
            demand=100
        )
        self.assertEqual(result, (50, 100, 0))

    # Test when threshold and storage are 0
    def test_threshold_zero(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=0,
            threshold_frac=0,
            backup=50,
            total_storage_capacity=200,
            demand=100
        )
        self.assertEqual(result, (0, 100, 50))

    # Test when renewable supply equals demand
    def test_renewable_supply_equals_demand(self):
        result = run_timestep(
            renewable_supply=100,
            storage_start=100,
            threshold_frac=0.5,
            backup=50,
            total_storage_capacity=200,
            demand=100
        )
        self.assertEqual(result, (100, 100, 0))

    # Test when renewable supply is zero
    def test_zero_renewable_supply(self):
        result = run_timestep(
            renewable_supply=0,
            storage_start=100,
            threshold_frac=0.5,
            backup=50,
            total_storage_capacity=200,
            demand=100
        )
        self.assertEqual(result, (50, 100, 50))

    # Test when storage is zero
    def test_zero_storage(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=0,
            threshold_frac=0.5,
            backup=50,
            total_storage_capacity=200,
            demand=200
        )
        self.assertEqual(result, (0, 100, 50))

    # Test when backup is zero
    def test_zero_backup(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=100,
            threshold_frac=0.5,
            backup=0,
            total_storage_capacity=200,
            demand=100
        )
        self.assertEqual(result, (50, 100, 0))

    # Test when total storage capacity is zero
    def test_zero_total_storage_capacity(self):
        result = run_timestep(
            renewable_supply=50,
            storage_start=0,
            threshold_frac=0.5,
            backup=50,
            total_storage_capacity=0,
            demand=100
        )
        self.assertEqual(result, (0, 100, 50))

    # Test when demand is zero
    def test_zero_demand(self):
        result = run_timestep(
            renewable_supply=5000,
            storage_start=100,
            threshold_frac=0.5,
            backup=50,
            total_storage_capacity=200,
            demand=0
        )
        self.assertEqual(result, (200, 0, 0))

    # Test when all values are zero
    def test_all_zero(self):
        result = run_timestep(
            renewable_supply=0,
            storage_start=0,
            threshold_frac=0,
            backup=0,
            total_storage_capacity=0,
            demand=0
        )
        self.assertEqual(result, (0, 0, 0))

if __name__ == '__main__':
    unittest.main()