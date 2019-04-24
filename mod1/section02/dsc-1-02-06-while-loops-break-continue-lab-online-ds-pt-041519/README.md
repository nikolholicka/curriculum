
# While Loops, Break and Continue - Lab

## Introduction
In this lab, we will practice using `while` loops, and `break` and `continue` statements in our code. We will use our control flow statements to iterate through collections and filter out or selectively operate on each element. We'll use `while` loops to perform operations until a given condition is no longer true.

## Objectives
You will be able to:
* Use a break and continue statements inside a loop
* Understand, explain and use while loops

## Instructions

### While Loops
Use while loops to perform the below operations and get the expected return values


```python
slices_of_pie = 6
slices_eaten = 0
# use a while loop to eat each slice of pie
# add each slice to the slices_eaten variable
```


```python
time_for_breakfast = 1468 # in seconds
number_of_cooked_pancakes = 0
# use a while loop to make yourself 5 pancakes for breakfast
# each pancake takes 27 seconds to cook on each side
# you must decrease the time_for_breakfast each time you 
# add a pancake to the skillet (frying pan) or flip a pancake (i.e. 2 times per pancake)
# there is only room for one pancake at a time
```

## For Loops

> **Hint:** You may find the [remove method](https://www.programiz.com/python-programming/methods/list/remove) to be useful for the next problem


```python
line_of_hungry_patrons = list(range(0,30))
fed_patrons = []
# use a while loop to to feed the hungry patrons who have an even number
# add the patrons with an even number to the fed_patrons list
# then remove the even numbered patrons from the line_of_hungry_patrons
# each list should contain 15 elements
```

### `break` And `continue` Statements

We have a list of person objects with all kinds of attributes. We'll use loops to find a person that meets a certain requirement that we are looking for or create new lists with a certain subset of elements. Write for loops with conditional statements in conjunction with `break` and `continue` to get the desired output.


```python
people = [
    {'name': "Daniel", 'age': 29, 'job': "Engineer", 'pet': "Cat", 'pet_name': "Gato"}, 
    {'name': "Katie", 'age': 30, 'job': "Teacher", 'pet': "Dog", 'pet_name': "Frank"},
    {'name': "Owen", 'age': 26, 'job': "Sales person", 'pet': "Cat", 'pet_name': "Cosmo"},
    {'name': "Josh", 'age': 22, 'job': "Student", 'pet': "Cat", 'pet_name': "Chat"},
    {'name': "Estelle", 'age': 35, 'job': "French Diplomat", 'pet': "Dog", 'pet_name': "Gabby"},
    {'name': "Gustav", 'age': 24, 'job': "Brewer", 'pet': "Dog", 'pet_name': "Helen"}
]
```


```python
# use the for loop below to find the *first* person in the list of people that has a dog as their pet
# the iteration count shouldn't exceed 2 iterations
first_dog_person = None
iteration_count = 0
for person in people:
    iteration_count += 1
    pass
```


```python
# use a for loop to create a list of the cat owners who are under the age of 28
cat_owners = None
# for loop goes here
```


```python
# use a for loop to find the first person who is above 29 years old in our list of people
# remember to use a break and or continue statement
thirty_something_yr_old = None
# for loop goes here
```


```python
# use a for loop to create a list of person names and another list of pet names for all dog owners
dog_owner_names = None
dog_names = None
# for loop goes here
```


```python
# use a for loop to create a list of odd numbers from the list of numbers from 0 to 100
# each time there is an odd number, add 10 to it and append it to the list_of_odd_numbers_plus_ten
# stop adding numbers to the list when there are 35 numbers
# use break and continue statements in your code
list_of_numbers = list(range(0,100))
list_of_odd_numbers_plus_ten = []
for number in list_of_numbers:
    pass
```

## Summary

In this lab, we practiced using while loops, which continue executing their block of code until the given condition is no longer truthy. This is useful for instances where we do not have a collection or do not need a collection to solve our problem, especially when we would only like to stop the process according to a certain condition. We then practiced using control flow statements, `break` and `continue`, to selectively operate on elements, append them to new lists, or assign them to new variables.
