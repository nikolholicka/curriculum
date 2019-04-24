# importing testing framwork
import pytests
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import sentence, flatiron_mantra, num_to_string, full_address

# format for writing tests
# all functions that are to be run by test suite *must* be prepended with test_
def test_sentence():
    assert sentence is not None, "The sentence variable must be assigned to a value!"
    assert type(sentence) == type(""), "The sentence must be a string!"
    assert sentence == 'Wow we love coding and strings!', "Remember we want to *captilize* this string in order to lower case all but the first character in this string"

def test_flatiron_mantra():
    assert flatiron_mantra is not None, "The flatiron_mantra variable must be assigned to a value!"
    assert type(flatiron_mantra) == type(""), "The flatiron_mantra must be a string!"
    assert flatiron_mantra == "Learn. Love. Code.", ""

def test_num_to_string():
    assert num_to_string is not None, "The num_to_string variable must be assigned to a value!"
    assert type(num_to_string) == type(""), "The num_to_string must be a string!"
    assert num_to_string == '1234', ""

def test_full_address():
    assert full_address is not None, "The full_address variable must be assigned to a value!"
    assert type(full_address) == type(""), "The full_address must be a string!"
    assert full_address == '1234 Abc street, Hometown USA', ""

# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"
