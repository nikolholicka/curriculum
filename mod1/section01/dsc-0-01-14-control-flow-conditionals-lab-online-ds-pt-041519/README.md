
# Control Flow: Conditionals - Lab

## Introduction
Now that we have been introduced to some different data types, how to use them, and now conditionals, let's put our knowledge to the test and create some conditional statements that selectively assign values to variables based on whether they pass the conditions we set.

## Objectives
You will be able to:
* Understand and use conditionals

## Instructions

Let's use our knowledge of variables and conditionals to assign values based on different conditions. Follow the instructions below to properly assign the values.

With the started code in the cell below, use what you know about numbers and conditionals to assign a value to less_than_50 and follow the directions given in the comments below. Use this as a guide to structuring the rest of the problems in this lab.


```python
number_50 = 50
less_than_50 = None
if number_50 > 100:
    # if number_50 is greater than 100, assign the `less_than_50` variable to the number 100
elif number_50 < 50:
    # if number_50 is greater than 50, assign the `less_than_50` variable to the number 50
else:
    # else assign the `less_than_50` variable to 49
```

Below, use conditionals to tell whether it is hot outside or not. If it is hot, assign the string `"It is so hot out!"` to the variable `is_it_hot`. If it is not hot, assign the string `"This is nothing! Bring on the heat."`. For our purposes, anything over `80` degrees is considered hot.


```python
temperature = 85
is_it_hot = None
# conditionals go here
```

Next, let's see what day of the week it is. There are 7 days in the week starting with Sunday at day `1` and ending with Saturday at day `7`. Use conditional statements to assign the day of the week to the variable `day_of_the_week` based on the number below assigned to the variable `today_is`.
For example, if the day is `2`, we would assign `day_of_the_week` the value `"Monday"`.


```python
today_is = 4
day_of_the_week = None
# conditionals go here
```

Finally, let's take a string and see if it ends with a certain substring. If it does, assign the variable `ends_with` to either `True` or `False`. For example, we have the string "School" and we want to know if it ends with the sub-string "cool". In this case it does not, so, we would assign `False` to the variable `ends_with`. 


```python
string = "Python"
sub_string = "on"
ends_with = None
# conditionals go here
```

### Summary

Great! In this lab we saw how to use our knowedge of conditionals to selectively assign values based on a condition. We will start integrating conditionals in many more ways in our code and we will start to see how useful they can become in more complex problems.
