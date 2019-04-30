
# Accessing Data within Pandas - Lab

## Introduction

In this lab, we'll look at a data set which contains information World cup matches. Let's use the pandas commands learned in the previous lecture to learn more about our data!

## Objectives
You will be able to:
* Understand and explain some key Pandas methods
* Access DataFrame data by using the label
* Perform boolean indexing on both Series and DataFrames
* Use simple selectors for series
* Set new Series and DataFrame inputs

## Load the data

Load the file `WorldCupMatches.csv` as a dataframe in Pandas

## Common methods and attributes

Use the correct method to look at the first 7 rows of the data set.

Look at the last 3 rows of the data set.

Get a concise summary of your data using `.info()`

Obtain a tuple representing the number of rows and number of columns

Use the appropriate attribute to get the column names

## Selecting dataframe information

When looking at the dataframe's `.head()`, you might have noticed that the games are structured chronologically in the dataframe.

Use the right selection method to print all the information from the 3rd to the 5th game.

Now, print all the info from game 5-9, but we're only interested to print out the "Home Team Name" and the "Away Team Name", 

Next, we'd like the information on all the games played in Group 3 for the 1950 World Cup.

Let's repeat the command above, but now we only want to print out the attendance column for the Group 3 games

You can combine conditions like this:

`df[(condition1) | (condition2)]`  -> Returns rows where either condition is true

`df[(condition1) & (condition2)]`  -> Returns rows where both conditions are true

Throughout the entire history of the world cup, How many Home games were played by the Netherlands?

How many games were played by the Netherlands in total?

Next, let's try and figure out how many games the USA played in the 2014 world cup. 

Now, let's try to find out how many countries participated in the 1986 world cup.

Hint 1: as a first step, create a new data set that only contain games in that year.

Hint 2: You can use `.unique()` to make sure you don't end up with duplicate country names.

In the world cup history, how matches had more than 5 goals in total?

## Changing values and creating new columns

With the information you currently have in your `df`, create a new column "Half-time Goals".

Run the code below. You'll notice that for Korea, there are records for both North-Korea (Korea DPR) and South-Korea (Korea Republic). 


```python
df.loc[df["Home Team Name"].str.contains('Korea'), "Home Team Name" ]
```

Imagine that for some reason, we simply want Korea listed as one entry, so we want to replace every "Home Team Name" and "Away Team Name" entry that contains "Korea" to simply "Korea". In the same way, we want to change the columns "Home Team Initials" and "Away Team Initials" to NSK (North & South Korea) instead of "KOR" and "PRK". 

Make sure to verify your answer!
