
# Creating Functions - Lab

## Introduction

As you know, we can use functions to name snippets of our code, thus making our code more expressive. We can also use functions to allow us to reuse our code. In this lab we will practice using functions for both of those purposes.

## Objectives
You will be able to:
* Create and use your own custom functions

## Instructions: 
### Writing our first functions

Imagine we are creating a list of travel destinations -- which can really turn out to be a full time job if we like to travel. We have our list of `travel_destinations` which we assign below. Write a function called `number_of_destinations` that returns the number of destinations we have on our list.


```python
travel_destinations = ['argentina', 'mexico', 'italy', 'finland', 'canada', 'croatia']
# define function here
```

> Below, remove the first `#` to uncomment the following line(s) of code and then press `shift` + `enter` to run the cell


```python
# number_of_destinations() # 6
```




    6



Now write another function called `next_up` that returns our first destination (the destination with the lowest index), in the `list_of_destinations` list.


```python
# define function here
```

> Below, remove the first `#` to uncomment the following line(s) of code and then press `shift` + `enter` to run the cell


```python
list_of_destinations = ['argentina', 'canada', 'croatia']
# next_up() # 'argentina'
```




    'argentina'



Ok, now write a function called `favorite_destination` that returns the string `'madagascar'`.


```python
# define function here
```

> Below, remove the first `#` to uncomment the following line(s) of code and then press `shift` + `enter` to run the cell


```python
# favorite_destination() # 'madagascar'
```




    'madagascar'



Again, let's declare a list called `favorite_destinations`. Write a new function called `add_favorite_destination` that also returns the string `'madagascar'`, but adds the string `'madagascar'` to the end of the list, `favorite_destinations`, as well.


```python
# define function here
```

> Below, remove the first `#` to uncomment the following line(s) of code and then press `shift` + `enter` to run the cell


```python
favorite_destinations = ['argentina', 'mexico', 'italy', 'finland', 'canada', 'croatia']
# add_favorite_destination()
favorite_destinations[-1] # 'madagascar'
```

    ['argentina', 'mexico', 'italy', 'finland', 'canada', 'croatia', 'madagascar']





    'madagascar'



Now let's write another function called `capitalize_countries` which iterates through the list of `capitalized_destinations` and capitalizes the first letter of each word. It should return a list of capitalized destinations.


```python
capitalized_destinations = ['argentina', 'mexico', 'italy', 'finland', 'canada', 'croatia']
# define function here
```

> Below, remove the first `#` to uncomment the following line(s) of code and then press `shift` + `enter` to run the cell


```python
# capitalize_countries() # ['Argentina', 'Mexico', 'Italy', 'Finland', 'Canada', 'Croatia']
```




    ['Argentina', 'Mexico', 'Italy', 'Finland', 'Canada', 'Croatia']



Great! Now if someone adds a country that is lowercased to our list of destinations, we can simply call our function again to capitalize each of the destinations in the list.

## Summary

Great job! In this lab we were able to get practice both creating and returning values from functions. We also practiced accessing variables not local to the function but in the global scope.
