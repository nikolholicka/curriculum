# importing testing framwork
import pytest
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import travel_month, number_of_weeks, travelling_schedule

# format for writing tests
# all functions that are to be run by test suite *must* be prepended with test_

# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"

def test_travel_month():
    assert travel_month is not None, "Remember to reassign the travel_month variable"
    assert travel_month.lower() == "January".lower()

def test_number_of_weeks():
    assert number_of_weeks is not None, "Remember to reassign the num_of_weeks variable"
    assert number_of_weeks is 3 or 5, "Follow the instructions to find the correct number of weeks we will travel"

def test_travelling_schedule():
    assert travelling_schedule is not None, "Remember to reassign the travel_month variable"
    assert travelling_schedule.lower() == "I will be travelling 5 weeks starting in the month of January".lower(), "You must interpolate both the number of weeks we will be travelling as well as the travel_month"
