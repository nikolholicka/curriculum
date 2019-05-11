# library used to check working virtual environment
import importlib
# library used to write tests
import pytest
# import data from JN
import ipynb.fs.full.index as notebook
from ipynb.fs.full.index import Driver, Passenger, Ride, meryl, daniel, flatiron_taxi, ride_to_school, ride_home

# format for writing tests
# all functions that are to be run by test suite *must* be prepended with test_

# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"

def test_class_definitions():
    class ExampleClass:
        pass
    assert "Driver" in notebook.__dict__, "Driver class must be defined"
    assert "Passenger" in notebook.__dict__, "Passenger class must be defined"
    assert "Ride" in notebook.__dict__, "Ride class must be defined"
    assert type(Driver) == type(ExampleClass)
    assert type(Passenger) == type(ExampleClass)
    assert type(Ride) == type(ExampleClass)


def test_instances():
    assert "meryl" in notebook.__dict__, "meryl instance must be defined and instantiated from the Passenger class"
    assert "daniel" in notebook.__dict__, "daniel instance must be defined and instantiated from the Passenger class"
    assert "flatiron_taxi" in notebook.__dict__, "flatiron_taxi instance must be defined and instantiated from the Driver class"
    assert "ride_to_school" in notebook.__dict__, "ride_to_school instance must be defined and instantiated from the Ride class"
    assert "ride_home" in notebook.__dict__, "ride_home instance must be defined and instantiated from the Ride class"
    assert type(meryl) == type(Passenger()), "meryl instance must be defined and instantiated from the Passenger class"
    assert type(daniel) == type(Passenger()), "daniel instance must be defined and instantiated from the Passenger class"
    assert type(flatiron_taxi) == type(Driver()), "flatiron_taxi instance must be defined and instantiated from the Driver class"
    assert type(ride_to_school) == type(Ride()), "ride_to_school instance must be defined and instantiated from the Ride class"
    assert type(ride_home) == type(Ride()), "ride_home instance must be defined and instantiated from the Ride class"
