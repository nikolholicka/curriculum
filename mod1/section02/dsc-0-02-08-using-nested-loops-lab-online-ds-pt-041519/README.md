
# Using Nested Loops - Lab

## Introduction
In this lab, we will practice using nested loops to iterate over nested data structures and create new collections. We'll be using data from a soccer match to practice our nested loops.

## Objectives
You will be able to:
* Combine different types of loops
* Understand, explain and use nested loops to iterate through nested data structures

## Instructions

Use nested loops and the below object, `soccer_match`, to complete the following prompts and get the desired return values.

Each object's structure is identical, but make sure to have a good idea of what that structure looks like before getting started. 


```python
soccer_match = [
  { "home_team": True,
    "away_team": False,
    "country": "France",
    "num_passes": 484,
    "passes_completed": 423,
    "fouls_committed": 16,
    "colors": ["blue", "white", "red"],
    "players": [
      {
        "name": "Hugo LLORIS",
        "captain": True,
        "shirt_number": 1,
        "position": "Goalie"
      },
      {
        "name": "Benjamin PAVARD",
        "captain": False,
        "shirt_number": 2,
        "position": "Defender"
      },
      {
        "name": "Raphael VARANE",
        "captain": False,
        "shirt_number": 4,
        "position": "Defender"
      },
      {
        "name": "Samuel UMTITI",
        "captain": False,
        "shirt_number": 5,
        "position": "Defender"
      },
      {
        "name": "Paul POGBA",
        "captain": False,
        "shirt_number": 6,
        "position": "Midfield"
      },
      {
        "name": "Antoine GRIEZMANN",
        "captain": False,
        "shirt_number": 7,
        "position": "Forward"
      },
      {
        "name": "Kylian MBAPPE",
        "captain": False,
        "shirt_number": 10,
        "position": "Forward"
      },
      {
        "name": "Ousmane DEMBELE",
        "captain": False,
        "shirt_number": 11,
        "position": "Forward"
      },
      {
        "name": "Corentin TOLISSO",
        "captain": False,
        "shirt_number": 12,
        "position": "Midfield"
      },
      {
        "name": "Ngolo KANTE",
        "captain": False,
        "shirt_number": 13,
        "position": "Midfield"
      },
      {
        "name": "Lucas HERNANDEZ",
        "captain": False,
        "shirt_number": 21,
        "position": "Defender"
      }
    ],
  },
  { "home_team": False,
    "away_team": True,
    "country": "Australia",
    "num_passes": 390,
    "passes_completed": 332,
    "fouls_committed": 19,
    "colors": ["green", "gold"],
    "players": [
      {
        "name": "Mathew RYAN",
        "captain": False,
        "shirt_number": 1,
        "position": "Goalie"
      },
      {
        "name": "Mark MILLIGAN",
        "captain": False,
        "shirt_number": 5,
        "position": "Defender"
      },
      {
        "name": "Mathew LECKIE",
        "captain": False,
        "shirt_number": 7,
        "position": "Forward"
      },
      {
        "name": "Robbie KRUSE",
        "captain": False,
        "shirt_number": 10,
        "position": "Forward"
      },
      {
        "name": "Andrew NABBOUT",
        "captain": False,
        "shirt_number": 11,
        "position": "Forward"
      },
      {
        "name": "Aaron MOOY",
        "captain": False,
        "shirt_number": 13,
        "position": "Midfield"
      },
      {
        "name": "Mile JEDINAK",
        "captain": True,
        "shirt_number": 15,
        "position": "Midfield"
      },
      {
        "name": "Aziz BEHICH",
        "captain": False,
        "shirt_number": 16,
        "position": "Defender"
      },
      {
        "name": "Joshua RISDON",
        "captain": False,
        "shirt_number": 19,
        "position": "Defender"
      },
      {
        "name": "Trent SAINSBURY",
        "captain": False,
        "shirt_number": 20,
        "position": "Defender"
      },
      {
        "name": "Tom ROGIC",
        "captain": False,
        "shirt_number": 23,
        "position": "Midfield"
      }
    ]
  }
]
```


```python
# iterate over the soccer_match list to create a new list with the name of the country for each team
countries = None
# code goes here
```


```python
# iterate over the soccer_match list to create a new list with the colors for each team
# this should be only one list containing strings for each of the country's colors
colors = None
# code goes here
```


```python
# iterate over the soccer_match list to create a new list with the players from each team
# this should be only one list containing the dictionaries for each of the country's players
players = None
# code goes here
```


```python
# iterate over the soccer_match list to create a new list with the captains from each team
# this should be only one list containing the dictionaries for each of the country's captains
captains = None
# code goes here
```


```python
# iterate over the soccer_match list to create a new list with the players from ONLY the home team
# this should be only one list containing the dictionaries for each of the home team's players
home_team_players = None
# code goes here
```


```python
# iterate over the soccer_match list to create a new list 
# with the players from ONLY the away team with the position of forward
# this should be only one list containing the dictionaries for each of the selected players
forwards = None
# code goes here
```


```python
# iterate over the soccer_match list and find the player with the highest shirt_number
# assign the player with the highst shirt_number to player_with_highest_num variable
player_with_highest_num = None
# code goes here
```


```python
# as we can see, the players all have their last names in all caps, as if they are being yelled
# we want a list of all the names of the players in this match
# but we want both first and last names to be formatted in title case
player_names = None
# code goes here
```

## Summary

In this lab, we practiced using nested loops to iterate through a nested data structure using data from a soccer match. Nested data structures can be quite complicated and it can become difficult to access more nested data. With nested loops, we are able to dynamically access this nested data and work with it as we would with a flatter data structure. It is important to think about the structure of the data before and while you're working so you know exactly what data you are working with at each level.
