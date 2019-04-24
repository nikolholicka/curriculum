# importing testing framwork
import pytests

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import is_it_hot, day_of_the_week, ends_with, less_than_50

# format for writing tests
# all functions that are to be run by test suite *must* be prepended with test_

def test_less_than_50():
    assert less_than_50 is not None, "Make sure to set up your conditional so that it re-assigns the value for `less_than_50`"
    assert type(less_than_50) == type(100), "less_than_50 should be a number"
    assert less_than_50 == 49, "Remember to assign the correct vzlue to the variable less_than_50"

def test_is_it_hot():
    assert is_it_hot is not None, "Make sure to set up your conditional so that it re-assigns the value for `is_it_hot`"
    assert type(is_it_hot) == type(""), "is_it_hot should be a String"
    assert is_it_hot == "It is so hot out!", "Remember any temperature above 80 degrees should assign the value `It is so hot out!`"

def test_day_of_the_week():
    assert day_of_the_week is not None, "Make sure to set up your conditional so that it re-assigns the value for `day_of_the_week`"
    assert type(day_of_the_week) == type(""), "day_of_the_week should be a String"
    assert day_of_the_week.lower() == "wednesday", "Remember to assign the name for the day of the week that corresponds to the number today_is"

def test_ends_with():
    assert ends_with is not None, "Make sure to set up your conditional so that it re-assigns the value for `ends_with`"
    assert type(ends_with) == type(True), "ends_with should be a boolean"
    assert ends_with == True, "Remember to test whether the substring `on` is the same as the ending of `Python` and assign the correct value to the variable `ends_with`"
