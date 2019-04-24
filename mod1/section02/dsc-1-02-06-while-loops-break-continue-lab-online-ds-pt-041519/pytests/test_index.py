# importing testing framwork
import pytest
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import slices_of_pie, slices_eaten, time_for_breakfast, number_of_cooked_pancakes, line_of_hungry_patrons, fed_patrons, first_dog_person, iteration_count, cat_owners, thirty_something_yr_old, dog_owner_names, dog_names, cat_owner_names, cat_names, list_of_odd_numbers_plus_ten

# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"

def test_find_first_dog_person():
    assert type(slices_of_pie) == type(1)
    assert type(slices_eaten) == type(1)
    assert slices_of_pie == 0
    assert slices_eaten == 6

def test_time_for_breakfast():
    assert type(time_for_breakfast) == type(1)
    assert time_for_breakfast == 1198
    assert number_of_cooked_pancakes == 5

def test_line_of_hungry_patrons():
    assert len(line_of_hungry_patrons) == 15
    assert len(fed_patrons) == 15
    assert line_of_hungry_patrons == [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    assert fed_patrons == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

def test_find_first_dog_person():
    assert first_dog_person == {'name': "Katie", 'age': 30, 'job': "Teacher", 'pet': "Dog", 'pet_name': "Frank"}
    assert iteration_count > 0
    assert iteration_count <= 2

def test_cat_owners():
    assert type(cat_owners) == type([])
    assert len(cat_owners) == 2
    assert cat_owners == [{'name': 'Owen', 'age': 26, 'job': 'Sales person', 'pet': 'Cat', 'pet_name': 'Cosmo'}, {'name': 'Josh', 'age': 22, 'job': 'Student', 'pet': 'Cat', 'pet_name': 'Chat'}]

def test_thirty_something_yr_old():
    assert type(thirty_something_yr_old) == type({})
    assert thirty_something_yr_old == {'name': "Katie", 'age': 30, 'job': "Teacher", 'pet': "Dog", 'pet_name': "Frank"}

def test_dog_owner_names():
    assert type(dog_owner_names) == type([])
    assert dog_owner_names == ['Katie', 'Estelle', 'Gustav']

def test_dog_names():
    assert type(dog_names) == type([])
    assert dog_names == ['Frank', 'Gabby', 'Helen']

def test_cat_owner_names():
    assert type(cat_owner_names) == type([])
    assert cat_owner_names == ['Daniel', 'Owen', 'Josh']

def test_cat_names():
    assert type(cat_names) == type([])
    assert cat_names == ['Gato', 'Cosmo', 'Chat']

def test_list_of_odd_numbers_plus_ten():
    assert type(list_of_odd_numbers_plus_ten) == type([])
    assert len(list_of_odd_numbers_plus_ten) == 35
    assert list_of_odd_numbers_plus_ten == [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79]
