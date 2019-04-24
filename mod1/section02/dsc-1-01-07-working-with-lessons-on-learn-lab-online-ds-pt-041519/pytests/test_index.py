# importing testing framwork
import pytest
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import number, flatiron_mantra # variable names go here

# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"

def test_number():
    assert number is not None, "Remember to replace None with 42!"
    assert number == 42, "We want to assign the `number` variable to 42"

def test_flatiron_mantra():
    assert flatiron_mantra is not None, "Remember to replace None with the flatiron_mantra, `Learn. Love. Code.`!"
    assert flatiron_mantra == "Learn. Love. Code.", "We want to assign the `flatiron_mantra` variable to the string `Learn. Love. Code.`"
