
# Gaussian/Normal Distribution - Lab

## Introduction

In this lab we shall learn how to generate random normal distributions in python. We shall look into visualising a histogram and building a density function using the formula as well as seaborn's built in functions. 

## Objectives
You will be able to:
* Generate random normal distributions in Python with given parameters
* Calculate the density function for normal distributions
* Use seaborn to visualize distributions with histograms and density functions

## A quick refresher! 
Here's the formula for calculating normal distribution density function.
<img src="formula.jpg" width = 300>

#### First generate a normal distribution containing 5000 values with mu=14 and sigma = 2.8


```python
# Generate a random normal variable with given parameters , n=5000

```

#### Calculate a normalized histogram for this distribution in matplotlib - use bin size = 20. 
#### Get the bin positions and count for each bin 

Refer to [official documentation](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) to view input and output options for `plt.hist()`


```python
# Calculate a histogram for above data distribution

```


![png](index_files/index_6_0.png)


#### Calculate the density function (using above formula) with mu, sigma and bin information calculated above .


```python
# Calculate the normal Density function 
density = None
```

#### Plot the histogram and density function


```python
# Plot histogram along with the density function

```


![png](index_files/index_10_0.png)


#### Visualize the distribution using seaborn and plot the KDE


```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a19d4cda0>




![png](index_files/index_12_1.png)


## Summary

In this lab we saw how to generate random normal distributions in python using numpy. We also looked into calculating the density for gaussian distributions using the general formula as well as seaborn's kde. We shall now move on to see how we can analyze such variables for answering analytical questions. 
