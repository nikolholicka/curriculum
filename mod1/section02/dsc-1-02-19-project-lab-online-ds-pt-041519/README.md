
# Project - Building a Calculator


## Introduction 

In this project, you'll create a simple calculator which can perform basic arithmetic operations like addition, subtraction, multiplication or division depending upon the user input.

## Objectives

You will be able to:

* Perform operations on various data types 
* Use loops to perform iteration 
* Use conditionals to make programming decisions
* Define and use functions 
* Ingest and parse user input 

## Approach 

* User choose the desired operation. Options 1, 2, 3, 4 are valid options for operations.  
* Two numbers are taken and an `if…elif…else` branching is used to execute a particular section.

* Using functions `add()`, `subtract()`, `multiply()` and `divide()` evaluate respective operations.

* The code should handle exceptions and must return **"invalid input*** when an unexpected character is given in the input (anything other than 1 - 4).

### Example Interface
Here is the interface you are expected to build. Don't worry if it is not 100% exactly as what is shown. Focus more on the getting the logic correct at this stage. 

```
Please select an operation:
1. Add
2. Subtract
3. Multiply
4. Divide

Select operations form 1, 2, 3, 4 : 1
Enter first number : 20
Enter second number : 13
20 + 13 = 33
```

### Creating Arithmatic Functions

We shall create four functions, one for each arithmatic operation which will perform the required operation and resturn the resulting value as shown below:


```python
# Function to add two numbers 
def add(num1, num2):
    #Perform the calculation
    return None
```


```python
# Function to subtract two numbers 
def subtract(num1, num2):
    #Perform the calculation
    return None
```


```python
# Function to multiply two numbers
def multiply(num1, num2):
    #Perform the calculation
    return None
```


```python
# Function to divide two numbers
def divide(num1, num2):
    #Perform the calculation
    return None
```

### Create a Command-line User Interface
We shall now write the main program body to take user input and call the relevent function


```python
# Print user menu 


# Take input from the user for operation , followed by numbers. 


# Based on operation, pass the two numbers to respective function
# Print the output in a nice manner
# Print "Invalid input" if an unexpected character is seen in input


# Expected output    

# Please select operation -
# 1. Add
# 2. Subtract
# 3. Multiply
# 4. Divide

# Select operations form 1, 2, 3, 4 :1
# Enter first number: 2
# Enter second number: 3
# 2 + 3 = 5
```

    Please select operation -
    1. Add
    2. Subtract
    3. Multiply
    4. Divide
    
    Select operations form 1, 2, 3, 4 :1
    Enter first number: 2
    Enter second number: 3
    2 + 3 = 5


## Bring in the While loop

We can see how the logic set by using if-else statements, along with functions can be used to control the flow of the program in an easy way. Let's add more functionality to our calculator as below:

>Lets try to make it a bit more interesting by introducing the behaviour of a real calculator so our users can choose to either continue with calculationa OR exit the system. Users gets this functionality by pressing `y` for yes and `n` for no towards continuation.

### Example Interface

**Notice `continue: y/n` at the bottom of interface.**

```
Please select an operation:
1. Add
2. Subtract
3. Multiply
4. Divide

Select operations form 1, 2, 3, 4 : 1
Enter first number : 20
Enter second number : 13
20 + 13 = 33

Continue: y/n
```

Let's work towards implementing iteration into the equation and enclose above I/O interface inside a while loops.


```python
# Initialize the code with cont (continue) flag set to yes (y)


# Check for user input after each iteration of the code in a while loop


        # Enclose the I/O  code block inside the while loop
        


#Expected output format

# Select operations form 1, 2, 3, 4 :4
# Enter first number: 5
# Enter second number: 4
# Press d for division and m for modulo operatorm
# 5 / 4 = 1
# Continue? y/n:y
# Select operations form 1, 2, 3, 4 :4
# Enter first number: 5
# Enter second number: 4
# Press d for division and m for modulo operatord
# 5 / 4 = 1.25
```

## Level up (Optional)

The while loop shown above allows the iteration through the code until a specific input from user i.e. `n` is noticed. Let's add some more functionality to this code by asking users about the type of division they are interested in, and this could be either normal division (as before) or a modulo operator (shows remainder).

>Change the code in the division function so that if a user selects division operation, the code should ask the user if he/she wants a normal division `/` (int) or `//` (float) , or a module `%` operator which only returns the remainder of a division. The program should return an exception for any other inputs. 


```python
def divide_v2(num1, num2):
    #Perform the calculation
    return None

```

### Summary

In this lab we saw how loops and conditions can be used to control the logic of a program execution based on user input. We started with building a simple calculator and incrementaly added more functionality to it by adding loops for iteration and further conditions allowing different type of calculations. We also practiced User I/O by taking choices from the users and dealing with exceptions (unexpected input). 
