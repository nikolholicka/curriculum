# importing testing framwork
import pytest
# library used to check working virtual environment
import importlib

# importing objects from the jupyter notebook here
from ipynb.fs.full.index import countries, colors, players, captains, home_team_players, forwards, player_with_highest_num, player_names


# tests to ensure correct environment is loaded
def test_conda_environment_activated():
    assert importlib.util.find_spec("obscure"), "It looks like you didn't 'conda activate learn-env' - try that then run the test again!"

def test_countries_list():
    assert countries is not None
    assert type(countries) == type([])
    assert len(countries) == 2
    assert countries == ['France', 'Australia']

def test_colors_list():
    assert colors is not None
    assert type(colors) == type([])
    assert len(colors) == 5
    assert colors == ['blue', 'white', 'red', 'green', 'gold']

def test_players_list():
    assert players is not None
    assert type(players) == type([])
    assert type(players[0]) == type({})
    assert len(players) == 22
    assert players == return_players()

def test_captains_list():
    assert captains is not None
    assert type(captains) == type([])
    assert type(captains[0]) == type({})
    assert len(captains) == 2
    assert captains == [{'name': 'Hugo LLORIS', 'captain': True, 'shirt_number': 1, 'position': 'Goalie'}, {'name': 'Mile JEDINAK', 'captain': True, 'shirt_number': 15, 'position': 'Midfield'}]

def test_home_team_players_list():
    assert home_team_players is not None
    assert type(home_team_players) == type([])
    assert type(home_team_players[0]) == type({})
    assert len(home_team_players) == 11
    assert home_team_players == return_home_team_players()

def test_forwards_list():
    assert forwards is not None
    assert type(forwards) == type([])
    assert type(forwards[0]) == type({})
    assert len(forwards) == 3
    assert forwards == [{'name': 'Mathew LECKIE', 'captain': False, 'shirt_number': 7, 'position': 'Forward'}, {'name': 'Robbie KRUSE', 'captain': False, 'shirt_number': 10, 'position': 'Forward'}, {'name': 'Andrew NABBOUT', 'captain': False, 'shirt_number': 11, 'position': 'Forward'}]

def test_player_with_highest_num_dict():
    assert player_with_highest_num is not None
    assert type(player_with_highest_num) == type({})
    assert player_with_highest_num == {'name': 'Tom ROGIC', 'captain': False, 'shirt_number': 23, 'position': 'Midfield'}

def test_player_names_list():
    assert player_names is not None
    assert type(player_names) == type([])
    assert type(player_names[0]) == type("")
    assert len(player_names) == 22
    assert player_names == ['Hugo Lloris', 'Benjamin Pavard', 'Raphael Varane', 'Samuel Umtiti', 'Paul Pogba', 'Antoine Griezmann', 'Kylian Mbappe', 'Ousmane Dembele', 'Corentin Tolisso', 'Ngolo Kante', 'Lucas Hernandez', 'Mathew Ryan', 'Mark Milligan', 'Mathew Leckie', 'Robbie Kruse', 'Andrew Nabbout', 'Aaron Mooy', 'Mile Jedinak', 'Aziz Behich', 'Joshua Risdon', 'Trent Sainsbury', 'Tom Rogic']

def return_players():
    return [{'name': 'Hugo LLORIS', 'captain': True, 'shirt_number': 1, 'position': 'Goalie'}, {'name': 'Benjamin PAVARD', 'captain': False, 'shirt_number': 2, 'position': 'Defender'}, {'name': 'Raphael VARANE', 'captain': False, 'shirt_number': 4, 'position': 'Defender'}, {'name': 'Samuel UMTITI', 'captain': False, 'shirt_number': 5, 'position': 'Defender'}, {'name': 'Paul POGBA', 'captain': False, 'shirt_number': 6, 'position': 'Midfield'}, {'name': 'Antoine GRIEZMANN', 'captain': False, 'shirt_number': 7, 'position': 'Forward'}, {'name': 'Kylian MBAPPE', 'captain': False, 'shirt_number': 10, 'position': 'Forward'}, {'name': 'Ousmane DEMBELE', 'captain': False, 'shirt_number': 11, 'position': 'Forward'}, {'name': 'Corentin TOLISSO', 'captain': False, 'shirt_number': 12, 'position': 'Midfield'}, {'name': 'Ngolo KANTE', 'captain': False, 'shirt_number': 13, 'position': 'Midfield'}, {'name': 'Lucas HERNANDEZ', 'captain': False, 'shirt_number': 21, 'position': 'Defender'}, {'name': 'Mathew RYAN', 'captain': False, 'shirt_number': 1, 'position': 'Goalie'}, {'name': 'Mark MILLIGAN', 'captain': False, 'shirt_number': 5, 'position': 'Defender'}, {'name': 'Mathew LECKIE', 'captain': False, 'shirt_number': 7, 'position': 'Forward'}, {'name': 'Robbie KRUSE', 'captain': False, 'shirt_number': 10, 'position': 'Forward'}, {'name': 'Andrew NABBOUT', 'captain': False, 'shirt_number': 11, 'position': 'Forward'}, {'name': 'Aaron MOOY', 'captain': False, 'shirt_number': 13, 'position': 'Midfield'}, {'name': 'Mile JEDINAK', 'captain': True, 'shirt_number': 15, 'position': 'Midfield'}, {'name': 'Aziz BEHICH', 'captain': False, 'shirt_number': 16, 'position': 'Defender'}, {'name': 'Joshua RISDON', 'captain': False, 'shirt_number': 19, 'position': 'Defender'}, {'name': 'Trent SAINSBURY', 'captain': False, 'shirt_number': 20, 'position': 'Defender'}, {'name': 'Tom ROGIC', 'captain': False, 'shirt_number': 23, 'position': 'Midfield'}]

def return_home_team_players():
    return [{'name': 'Hugo LLORIS', 'captain': True, 'shirt_number': 1, 'position': 'Goalie'}, {'name': 'Benjamin PAVARD', 'captain': False, 'shirt_number': 2, 'position': 'Defender'}, {'name': 'Raphael VARANE', 'captain': False, 'shirt_number': 4, 'position': 'Defender'}, {'name': 'Samuel UMTITI', 'captain': False, 'shirt_number': 5, 'position': 'Defender'}, {'name': 'Paul POGBA', 'captain': False, 'shirt_number': 6, 'position': 'Midfield'}, {'name': 'Antoine GRIEZMANN', 'captain': False, 'shirt_number': 7, 'position': 'Forward'}, {'name': 'Kylian MBAPPE', 'captain': False, 'shirt_number': 10, 'position': 'Forward'}, {'name': 'Ousmane DEMBELE', 'captain': False, 'shirt_number': 11, 'position': 'Forward'}, {'name': 'Corentin TOLISSO', 'captain': False, 'shirt_number': 12, 'position': 'Midfield'}, {'name': 'Ngolo KANTE', 'captain': False, 'shirt_number': 13, 'position': 'Midfield'}, {'name': 'Lucas HERNANDEZ', 'captain': False, 'shirt_number': 21, 'position': 'Defender'}]
