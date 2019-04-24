# importing testing framwork
import pytests

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import greenville, greenville_population, greenville_area, city_keys, city_values, cities, salina, los_cabos_pop, city_count, pyeongchang_keys, pyeongchang_values

# format for writing tests
# all functions that are to be run by test suite *must* be prepended with test_

def test_greenville_population():
    assert greenville_population == 84554

def test_greenville_area():
    assert greenville_area == 68

def test_city_keys():
    assert city_keys == ['Area', 'City', 'Country', 'Population']

def test_city_values():
    assert city_values == [68, 'Greenville', 'USA', 84554]

def test_salina():
    assert salina == {'Area': 27, 'City': 'Salina Island', 'Country': 'Italy', 'Population': 4000}

def test_los_cabos_pop():
    assert los_cabos_pop == 287651

def test_city_count():
    assert city_count == 12

def test_change_spelling():
    assert cities[11]['City'] == 'PyeongChang'

def test_pyeongchang_keys():
    assert type(pyeongchang_keys) == type(list())
    assert pyeongchang_keys == ['City', 'Country', 'Population', 'Area']

def test_pyeongchang_values():
    assert type(pyeongchang_values) == type(list())
    assert pyeongchang_values == ['PyeongChang', 'South Korea', 2581000, 3194]
