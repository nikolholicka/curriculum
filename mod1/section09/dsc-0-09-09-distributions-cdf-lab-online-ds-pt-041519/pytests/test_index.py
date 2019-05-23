# importing testing framwork
import pytest
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import # variable names go here

# format for writing tests
# all functions that are to be run by test suite *must* be prepended with test_
def test_name_of_test_here():
    assert True, "AssertionError will *not* raise and this message will not show"
    assert False, "AssertionError will raise and output this message in the trace"

# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"
