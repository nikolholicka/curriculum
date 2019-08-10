
# The Kolmogorov-Smirnov Test - Lab

## Introduction
In the previous lesson, we saw that the Kolmogorov–Smirnov statistic quantifies a distance between the empirical distribution function of the sample and the cumulative distribution function of the reference distribution, or between the empirical distribution functions of two samples. In this lab, we shall see how to perform this test in python. 

## Objectives

You will be able to:
* Perform 1 sample and 2 sample KS tests in Python and Scipy
* Compare the KS test to visual approaches for checking normality assumptions
* Plot the CDF and ECDF to visualize parametric and empirical cumulative distribution functions

## Generate Data

### Let's import necessary libraries and generate some data 


```python
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Create the normal random variables with mean 0, and sd 3
x_10 = stats.norm.rvs(loc=0, scale=3, size=10)
x_50 = stats.norm.rvs(loc=0, scale=3, size=50)
x_100 = stats.norm.rvs(loc=0, scale=3, size=100)
x_1000 = stats.norm.rvs(loc=0, scale=3, size=1000)
```

### Plot Histograms and QQ plots of above datasets and comment on the output 

- How good are these techniques for checking normality assumptions?
- Compare both these techniques and identify their limitations/benefits etc. 



```python
# Plot histograms and QQplots for above datasets

# You code here
```

    x_10



![png](index_files/index_5_1.png)



![png](index_files/index_5_2.png)


    x_50



![png](index_files/index_5_4.png)



![png](index_files/index_5_5.png)


    x_100



![png](index_files/index_5_7.png)



![png](index_files/index_5_8.png)


    x_1000



![png](index_files/index_5_10.png)



![png](index_files/index_5_11.png)



```python
# You comments here 
```

### Creat a function to plot the normal CDF and ECDF for a given dataset
- Create a function ks_plot(data) to generate an empirical CDF from data
- Create a normal CDF using the same mean = 0 and sd = 3 , having the same number of values as data


```python
# You code here 

def ks_plot(data):

    pass
    
# Uncomment below to run the test
# ks_plot(stats.norm.rvs(loc=0, scale=3, size=100)) 
# ks_plot(stats.norm.rvs(loc=5, scale=4, size=100))

```


![png](index_files/index_8_0.png)



![png](index_files/index_8_1.png)


This is awesome. The difference between two cdfs in the second plot show that sample did not come from the distribution which we tried to compare it against. 

### Now you can run all the generated datasets through the function ks_plot and comment on the output.


```python
# Your code here 
```


![png](index_files/index_10_0.png)



![png](index_files/index_10_1.png)



![png](index_files/index_10_2.png)



![png](index_files/index_10_3.png)



```python
# Your comments here 

```

### KS test in SciPy

Lets run the Kolmogorov-Smirnov test, and use some statistics to get a final verdict on normality. It lets us test the hypothesis that the sample is a part of the standard t-distribution. In SciPy, we run this test using the method below:

```python
scipy.stats.kstest(rvs, cdf, args=(), N=20, alternative='two-sided', mode='approx')
```
Details on arguments being passed in can be viewed at this [link to official doc.](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html)


### Run the KS test for normality assumption using the datasets created earlier and comment on the output
- Perform the KS test against a normal distribution with mean = 0 and sd = 3
- If p < .05 we can reject the null hypothesis and conclude our sample distribution is not identical to a normal distribution.


```python
# Perform KS test 

# Your code here 

# KstestResult(statistic=0.1377823669421559, pvalue=0.9913389045954595)
# KstestResult(statistic=0.13970573965633104, pvalue=0.2587483380087914)
# KstestResult(statistic=0.0901015276393986, pvalue=0.37158535281797134)
# KstestResult(statistic=0.030748345486274697, pvalue=0.29574612286614443)
```

    KstestResult(statistic=0.1377823669421559, pvalue=0.9913389045954595)
    KstestResult(statistic=0.13970573965633104, pvalue=0.2587483380087914)
    KstestResult(statistic=0.0901015276393986, pvalue=0.37158535281797134)
    KstestResult(statistic=0.030748345486274697, pvalue=0.29574612286614443)



```python
# Your comments here 

```


### Generate a uniform distribution and plot / calculate the ks test against a uniform as well as a normal distribution


```python
# Try with a uniform distribution
x_uni = np.random.rand(1000)

# KstestResult(statistic=0.023778383763166322, pvalue=0.6239045200710681)
# KstestResult(statistic=0.5000553288071681, pvalue=0.0)
```

    KstestResult(statistic=0.023778383763166322, pvalue=0.6239045200710681)
    KstestResult(statistic=0.5000553288071681, pvalue=0.0)



```python
# Your comments here 

```

## 2 sample KS test
A two sample KS test is available in SciPy using following function
```python 
scipy.stats.ks_2samp(data1, data2)[source]
```

Let's generate some bi-modal data first for this test 


```python
# Generate binomial data
N = 1000
x_1000_bi = np.concatenate((np.random.normal(-1, 1, int(0.1 * N)), np.random.normal(5, 1, int(0.4 * N))))[:, np.newaxis]
plt.hist(x_1000_bi);
```


![png](index_files/index_21_0.png)


### Plot the CDFs for x_1000_bimodal and x_1000 and comment on the output 


```python

# Plot the CDFs
def ks_plot_2sample(data_1, data_2):
    '''
    Data entereted must be the same size.
    '''
    pass

# Uncomment below to run
# ks_plot_2sample(x_1000, x_1000_bi[:,0])

```


![png](index_files/index_23_0.png)



```python
# You comments here 

```

### Run the two sample KS test on x_1000 and x_1000_bi and comment on the results


```python
# Your code here

# Ks_2sampResult(statistic=0.633, pvalue=4.814801487740621e-118)
```


```python
# Your comments here 


```

## Summary

In this lesson, we saw how to check for normality (and other distributions) using one sample and two sample ks-tests. You are encouraged to use this test for all the upcoming algorithms and techniques that require a normality assumption. We saw that we can actually make assumptions for different distributions by providing the correct CDF function into Scipy KS test functions. 
