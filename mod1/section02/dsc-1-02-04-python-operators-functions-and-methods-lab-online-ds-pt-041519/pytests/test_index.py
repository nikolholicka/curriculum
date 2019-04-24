# importing testing framwork
import pytest
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import * # variable names go here

# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"

def test_yell_hello():
    assert type(yell_hello) == type("HELLO, THERE"), "yell_hello must be a string"
    assert yell_hello == "HELLO, THERE", "Remember to put the string in all caps"

def test_whisper_hey():
    assert type(whisper_hey) == type("HELLO, THERE"), "whisper_hey must be a string"
    assert whisper_hey == "psst, hey", "Remember to put the string in all lowercase"

def test_flatiorn_mantra():
    assert type(flatiorn_mantra) == type("HELLO, THERE"), "flatiorn_mantra must be a string"
    assert flatiorn_mantra == "Learn. Love. Code.", "Remember to put the string in title case"

def test_type_string():
    assert type_string == type("HELLO, THERE"), "type_string must be of type string"
    assert type(type_string) == type(dict) # another test to make sure the type str is returned. type() of a type returns type

def test_type_list():
    assert type_list == type([]), "type_list must be a of type list"
    assert type(type_list) == type(dict) # another test to make sure the type list is returned. type() of a type returns type

def test_lenght_of_list():
    assert type(lenght_of_list) == type(10), "lenght_of_list must be a list"
    assert lenght_of_list == 3, "lenght_of_list should contain only 3 elements"

def test_longest_word_in_list():
    assert type(longest_word_in_list) == type(""), "longest_word_in_list must be a string"
    assert longest_word_in_list == "list", "longest_word_in_list should be the longest word in the list"

def test_smallest_number():
    assert type(smallest_number) == type(10), "smallest_number must be a number"
    assert smallest_number == 1, "smallest_number should be the smallest number in the list"

def test_sum_of_numbers():
    assert type(sum_of_numbers) == type(10), "sum_of_numbers must be a number"
    assert sum_of_numbers == 11, "sum_of_numbers should be the sum of all numbers in the list"

def test_boolean_compare():
    assert boolean_compare == False
    assert boolean_compare2 == False

def test_number_compare():
    assert number_compare == True
    assert number_compare2 == True
    assert number_compare3 == False

def test_string_compare():
    assert string_compare == True
    assert string_compare2 == False
    assert string_compare3 == True

def test_list_compare():
    assert list_compare == True
    assert list_compare2 == True
    assert list_compare3 == False
    assert list_compare4 == True
    assert list_compare5 == False

def test_logical_compare():
    assert logical_compare == []
    assert logical_compare2 == True
    assert logical_compare3 == 0
    assert logical_compare4 == 2
    assert logical_compare5 == 2
    assert logical_compare6 == False
    assert logical_compare7 == False

def test_identity_compare():
    assert identity_compare == False
    assert identity_compare2 == True
    assert identity_compare3 == True
    assert identity_compare4 == True
    assert identity_compare5 == False
    assert identity_compare6 == False
