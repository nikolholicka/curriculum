
# Instance Methods - Lab

## Introduction
In the last lesson, we talked about instance methods -- what are they and how to define them. In this lab, we are going to put our new skills to the test and start writing our own instance methods on our classes.

## Objectives

You will be able to: 

* Define classes and instantiate instances of those classes
* Define and call instance methods

## Defining Classes and Instance Methods

In this section, define two classes, `Driver`, `Passenger`. 

In the `Driver` class, define the instance method `greeting` that returns the string `"Hey, how are you?"`. Then define the method `ask_for_destination`, which returns the string `"Where would you like to go today?"`. 

In the `Passenger` class, define the instance method `reply_greeting` that returns the string `"I am doing well! Thanks for picking me up today!"`. Then define the method `in_a_hurry`, which returns the string `"Punch it! They're on our tail!"`. 

Define these classes and instance methods in the cells below
    
> **Remember:** *as we learned in the previous lesson, we need to define our instance methods with at least one argument (`self`) in order to call them on an instance object. Don't worry, we will learn more about this argument in a later lesson.*


```python
# Define Driver class here
```


```python
# Define Passenger class here 
```

## Instantiate Instances and Practice Using Instance Methods
Great! We've defined our classes and our instance methods. Now, in this section we are going to actually use them!

Start by instantiating two drivers and two passengers. Assign the drivers to the variables `daniel` and `meryl` and assign the passengers to `niky` and `terrance`.


```python
daniel = None # driver
meryl = None # driver
niky = None # passenger
terrance = None # passenger
```

Alright, we have our passengers and drivers! Now we need to put those instance methods to use. Try them out and assign the return values to the variables below. Let's have `daniel` greet his passenger, who is going to be `niky`. Assign the greeting to the variable, `polite_greeting`. Have `niky` respond by calling `in_a_hurry`, and assign the return value to the variable, `no_time_to_talk`.


```python
polite_greeting = None
print(polite_greeting)
```


```python
no_time_to_talk = None
print(no_time_to_talk)
```

## Extra Practice

Now, let's have some extra practice creating classes, instantiating objects, and using instance methods. 

In the cells below, you'll create three different classes that represent animals in a zoo--lions, tigers, and elephants.  Each animal should have a method, `speak`, which returns a string containing the sound they make (feel free have some fun with this--we don't know how to spell the sound an elephant makes any better than you do!). 


```python
# create Lion class
```


```python
# create Tiger class
```


```python
# create Elephant class
```

Now, in the cell below, create an instance of each animal. 


```python
simba = None
tony = None
dumbo = None
```

Now, append each of them into the list `zoo` in the cell below.


```python
zoo = None
```

Now, loop through the `zoo` list and call print out the `.speak()` method for every animal in the zoo. 

## Summary
In this lab, we practiced defining classes and instance methods. We then instantiated instances of our classes and used them to practice calling our instance methods. 
