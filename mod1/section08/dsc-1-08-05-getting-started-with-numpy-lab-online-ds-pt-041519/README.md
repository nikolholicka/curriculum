
# Getting Started with NumPy - Lab

## Introduction

Now that we have introduced NumPy, let's put it to practice. In this lab, we are going to be creating arrays, performing operations on them, and returning new array all using the NumPy library. Let's get started!

## Objectives

You will be able to: 

* Understand how to initialize NumPy arrays from nested Python lists, and access elements using square brackets
* Understand the shape attribute on NumPy arrays
* Understand how to create arrays from scratch including np.zeros, np.ones, np.full
* Learn to perform scalar and vector math  

## Import NumPy under the standard alias


```python
#Your code here
```

## Generating Some Mock Data

Create a NumPy Array for each of the following:
    1. Using a range
    2. Using a Python List
    
Below, create a list in Python that has 5 elements (i.e. [0,1,2,3,4]) and assign it to the variable `py_list`. 

Next, do the same, but instead of a list, create a range with 5 elements and assign it to the variable, `py_range`.

Finally, use the list and range to create NumPy arrays and assign the array from list to the variable `array_from_list`, and the array from the range to the variable `array_from_range`.


```python
#Your code here
```

Next, we have a list of heights and weights and we'd like to use them to create a collection of BMIs. However, they are both in inches and pounds (imperial system), respectively. 

Let's use what we know to create NumPy arrays with the metric equivalent values, (height in meters & weight in kg).

> **Remember:** *NumPy can make these calculations a lot easier and with less code than a list!*

> 1.0 inch = 0.0254 meters

> 2.2046 lbs = 1 kilogram


```python
# use the conversion rate for turning height in inches to meters
list_height_inches = [65, 68, 73, 75, 78]

#Your code here
```


```python
# use the conversion rate for turning weight in pounds to kilograms
list_weight_pounds = [150, 140, 220, 205, 265]

#your code here
```

The metric formula for calculating BMI is as follows:

> BMI = weight (kg) ÷ height^2 (m^2)

So, to get BMI we divide weight by the squared value of height. For example, if i weighed 130kg and was 1.9 meters tall, the calculation would look like:

> BMI = 130 / (1.9*1.9)

Use the BMI calculation to create a NumPy array of BMIs


```python
#Your code here
```

## Create an identity vector using np.ones()


```python
#Your code here
```

## Multiply the BMI_array by your identity vector


```python
#Your code here
```

## Level Up: Using NumPy to Parse a File
The pandas library that we've been using is built on top of NumPy; all columns/series in a Pandas DataFrame are built using NumPy arrays. To get a better idea of a how a built in method like pd.read_csv() works, we'll try and recreate that here!


```python
#Open a text file (csv files are just plaintext seperated by commas)
f = open('bp.txt')
n_rows = len(f.readlines())
print('The file has {} lines.'.format(n_rows)) #Print number of lines in the file
f = open('bp.txt') #After using readlines, we must reopen the file
n_cols = (len(f.readline().split('\t'))) #The file has values seperated by tabs; we read the first line and check it's length.

f = open('bp.txt')

#Your code here
#Pseudocode outline below
#1) Create a matrix of zeros that is the same size of the file
#2) Iterate through the file: "for line in f:" Hint: using enumerate will also be required
    #3) Update each row of the matrix with the new stream of data
    #Hint: skip the first row (its just column names, not the data.)
#4) Preview your results; you should now have a NumPy matrix with the data from the file

```

## Summary

In this lab, we practiced creating NumPy arrays from both lists and rages. We then practiced peforming math operations like converting imperial measurements to metric measurements on each element of a NumPy array to create new arrays with new values. Finally, we used both of our new NumPy arrays to operate on eachother and create new arrays containing the BMIs from our arrays containing heights and weights.
