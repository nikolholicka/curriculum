import pytest
from ipynb.fs.full.index import *

def test_instance_types():
    assert type(meryl) == type(Driver())
    assert type(daniel) == type(Driver())
    assert type(niky) == type(Passenger())
    assert type(terrance) == type(Passenger())

def test_variables():
    assert polite_greeting == "Hey, how are you?"
    assert no_time_to_talk == "Punch it! They're on our tail!"

def test_instance_methods():
    assert daniel.greeting() == "Hey, how are you?"
    assert daniel.ask_for_destination() == "Where would you like to go today?"
    assert niky.in_a_hurry() == "Punch it! They're on our tail!"
    assert niky.reply_greeting() == "I am doing well! Thanks for picking me up today!"
