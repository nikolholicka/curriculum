# importing testing framwork
import pytests

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import italy, mexico, kindof_neighbors, countries, unique_countries, num_of_repeats

# format for writing tests
# all functions that are to be run by test suite *must* be prepended with test_
def test_italy():
    assert italy == 'Italy'

def test_mexico():
    assert mexico == 'Mexico'

def test_kindof_neighbors():
    assert kindof_neighbors == ['USA', 'Argentina', 'Mexico', 'USA']

def test_malta_presence():
    assert 'Malta' in countries

def test_new_mexico_presence():
    assert 'New Mexico' not in countries

def test_unique_countries():
    assert sorted(['Argentina', 'Canada', 'Croatia', 'Finland', 'Italy', 'Malta', 'Mexico', 'Morocco', 'South Korea', 'USA']) == sorted(unique_countries)

def test_num_repeats():
    assert num_of_repeats == 3
