# importing testing framwork
import pytests
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import cities, city_indices, city_names, names_and_ranks, city_populations, city_areas

def test_cities_list():
    assert type(cities) == type([])
    assert len(cities) == 12

def test_city_indices():
    assert type(city_indices) == type([])
    assert city_indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def test_city_names():
    assert type(city_names) == type([])
    assert len(city_names) == 12
    assert city_names == ['Buenos Aires', 'Toronto', 'Pyeongchang', 'Marakesh', 'Albuquerque', 'Los Cabos', 'Greenville', 'Archipelago Sea', 'Walla Walla Valley', 'Salina Island', 'Solta', 'Iguazu Falls']

def test_names_ranks():
    assert type(names_and_ranks) == type([])
    assert len(names_and_ranks) == 12
    assert names_and_ranks == ['1. Buenos Aires', '2. Toronto', '3. Pyeongchang', '4. Marakesh', '5. Albuquerque', '6. Los Cabos', '7. Greenville', '8. Archipelago Sea', '9. Walla Walla Valley', '10. Salina Island', '11. Solta', '12. Iguazu Falls']

def test_city_populations():
    assert type(city_populations) == type([])
    assert len(city_populations) == 12
    assert city_populations == [2891000, 2800000, 2581000, 928850, 559277, 287651, 84554, 60000, 32237, 4000, 1700, 0]

def test_city_areas():
    assert type(city_areas) == type([])
    assert len(city_areas) == 12
    assert city_areas == [4758, 2731, 3194, 200, 491, 3750, 68, 8300, 33, 27, 59, 672]
