
# Object Initialization - Lab

## Introduction
In this lab, we will practice defining classes with custom `__init__` methods. We will define two classes, Driver and Passenger in the cells below. Follow the instructions in order to get the tests to pass.

## Objectives

You will be able to:

* Define custom `__init__` methods for object initialization
* Create instance variables in the `__init__` method 
* Use default arguments in the `__init__` method

## Initializing Instance Objects

Let's start off by defining Driver and Passenger classes. In the Passenger class, we will want our passengers to have the attributes `first`, `last`, `email`, and `rides_taken` for their first name, last name, email, and number of rides they have taken. Let's provide our `__init__` method with the default argument of `0` for the `rides_taken` attribute since new passengers should not have taken any rides. 

After we've defined our Passenger `__init__` method, check it out by initializing a new passenger with the first name `"Rebecca"`, the last name `"Black"`, and the email `"rebecca.black@gmail.com"`. Don't worry about creating instance methods or using any attribute decorators in this lesson. We can just instantiate instance variables and use those to access the instance's attributes.


```python
# Define Passenger Class Here
```


```python
rebecca_black = None # initialize Rebecca Black here
print(rebecca_black.first) # "Rebecca"
print(rebecca_black.last) # "Black"
print(rebecca_black.email) # "rebecca.black@gmail.com"
print(rebecca_black.rides_taken) # 0
```

![rebecca black in a car](https://media.giphy.com/media/8SS0MSoBHa8la/giphy.gif)

Great work! Rebecca Black is now in the system and ready to request her ride to the party on Friday! Friday! Friday!

In the Driver class, define an `__init__` method that initializes a driver with the attributes `first`, `last`, and `favorite_hobby` for their first name, last name, and favorite hobby. Let's provide a default argument of `"driving"` for `favorite_hobby`, since we would hope they are ***doing what they love***! 

After we have our Driver `__init__` method, let's initialize a driver with the first name `"Dale"`, last name `"Earnhardt"`. Again, don't worry about creating any instance methods or using any property decorators.


```python
# Define Passenger Class Here
```


```python
dale_earnhardt = None # initialize Dale Earnhardt here
print(dale_earnhardt.first) # "Dale"
print(dale_earnhardt.last) # "Earnhardt"
print(dale_earnhardt.favorite_hobby) # "driving"
```

![DaleEarnhardt](https://media.giphy.com/media/3ohzdSGBkwbvuPdO3S/giphy.gif)

Awesome, maybe Dale will be Rebecca's driver to the party on Friday! Friday! Friday! 

## Summary


In this lab, we practiced definining custom `__init__` methods that allowed us to initialize new instances with a set of predetermined attributes and default attributes.
