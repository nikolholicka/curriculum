
# Confidence Intervals with T Distribution - Lab

## Introduction

In the previous lab we saw that if we have the standard deviation for the population, we can use use z-score to calculate our confidence interval using the mean of sample means. 

If, on the other hand, standard deviation of the population is not known (which is usually the case), you have to use the standard deviation of your sample as a stand in when creating confidence intervals. Since the sample standard deviation is often different then that of the population, further potential errors are introduced to our confidence intervals. To account for this error, we use what's known as a t-critical value instead of the z-critical value.

The t-critical value is drawn from what's known as a t-distribution
> A t-distribution  closely resembles the normal distribution but  gets wider and wider as the sample size falls.

<img src="images/new_t-distr-img.png" width="500">

The t-distribution is available in scipy.stats with the nickname "t" so we can get t-critical values with `stats.t.ppf()`.

## Objectives
You will be able to:

* Understand the concept of a confidence interval and be able to construct one for a mean

* Demonstrate how to use the t-distribution for constructing intervals for small sample sizes

* Express a correct interpretation of confiendence intervals

## Let's get started!


```python
# Import the necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
import math
```

Let's investigate point estimates by generating a population of random age data collected a two different locations and then drawing a sample from it to estimate the mean:


```python
np.random.seed(20)
population_ages1 = np.random.normal(20, 4, 10000) 
population_ages2 = np.random.normal(22, 3, 10000) 
population_ages = np.concatenate((population_ages1, population_ages2))

pop_ages = pd.DataFrame(population_ages)
pop_ages.hist(bins=100,range=(5,33),figsize=(9,9))
pop_ages.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>21.008578</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.671277</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.784588</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18.662256</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.163276</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>23.489438</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.140116</td>
    </tr>
  </tbody>
</table>
</div>




![png](index_files/index_6_1.png)


Let's take a new, smaller sample (<30) and calculate how much sample mean differs from population mean.


```python
np.random.seed(23)

sample_size = 25
sample = None # Take a random sample of size 25 from above population
sample_mean = None  # Calculate sample mean 

# Print sample mean and difference of sample and population mean 

# Sample Mean: 19.870788629471857
# Mean Difference: 1.1377888781920937
```

We can see that sample mean differs from population mean by 1.13 years. We can calculate a confidence interval without the population standard deviation, using the t-distribution using `stats.t.ppf(q, df)` function. This function takes in a value for confidence level required (q) with "degree of freedom" (df) .

> degrees of freedom = sample_size -1.


```python
# Cal culate the t-critical value for 95% confidence level for sample taken above. 
t_critical = None   # Get the t-critical value  by using 95% confidence level and degree of freedom
print("t-critical value:")                  # Check the t-critical value
#print(t_critical)     

# t-critical value:
# 2.0638985616280205
```

Calculate the confidence interval of the sample by sigma and calculating margin of error as:
> **sigma = sample_std/√n**

> **Margin of Error = t-critical-value * sigma**

and finally the confidence interval can be calculated as : 

> **Confidence interval = (sample_mean + margin of error, sample_mean - margin of error)**


```python
# Calculate the sample standard deviation
sample_stdev = None    # Get the sample standard deviation

# Calculate sigma using the formula described above to get population standard deviation estimate
sigma =None

# Calculate margin of error using t_critical and sigma
margin_of_error = None

# Calculate the confidence intervals using calculated margin of error 
confidence_interval = None


print("Confidence interval:")
#print(confidence_interval)

# Confidence interval:
# (18.4609156900928, 21.280661568850913)
```

We can verify our calculations by using the Python function stats.t.interval():


```python
stats.t.interval(alpha = 0.95,              # Confidence level
                 df= 24,                    # Degrees of freedom
                 loc = sample_mean,         # Sample mean
                 scale = sigma)             # Standard deviation estimate
```




    (18.4609156900928, 21.280661568850913)



We can see that the calculated confidence interval includes the population mean calculated above.

Lets run the code multiple times to see how often our estimated confidence interval covers the population mean value:

**Write a function using code above that takes in sample data and returns confidence intervals**




```python
# Function to take in sample data and calculate the confidence interval
def conf_interval(sample):
    '''
    Input:  sample 
    Output: Confidence interval
    '''
    n = len(sample)
    x_hat = sample.mean()
    # Calculate the z-critical value using stats.norm.ppf()
    # Note that we use stats.t.ppf with q = 0.975 to get the desired t-critical value 
    # instead of q = 0.95 because the distribution has two tails.

    t = None  #  t-critical value for 95% confidence
    
    sigma = None # Sample standard deviation

    # Calculate the margin of error using formula given above
    moe = None

    # Calculate the confidence interval by applying margin of error to sample mean 
    # (mean - margin of error, mean+ margin of error)
    conf = None
    
    return conf
```

**Call the function 25 times taking different samples at each iteration and calculating sample mean and confidence intervals**


```python
# set random seed for reproducability
np.random.seed(12)

# Select the sample size 
sample_size = 25

# Initialize lists to store interval and mean values
intervals = []
sample_means = []

# Run a for loop for sampling 25 times and calculate + store confidence interval and sample mean values in lists initialised above

for sample in range(25):

    # Take a random sample of chosen size 
    
    
    # Calculate sample mean and confidence_interval
   
  
    # Calculate and append sample means and conf intervals for each iteration


```

**Plot the confidence intervals along with sample means and population mean**


```python
# Plot the confidence intervals with sample and population means
# Draw the mean and confidence interval for each sample
# Draw the population mean 
```


```python

```




    <matplotlib.collections.LineCollection at 0x1a19dd8860>




![png](index_files/index_21_1.png)


Just like the last lab, all but one of the 95% confidence intervals overlap the red line marking the true mean. This is to be expected: since a 95% confidence interval captures the true mean 95% of the time, we'd expect our interval to miss the true mean 5% of the time.

## Summary

In this lab we learnt how to use confidence intervals when population standard deviation is not known, and the sample size is small (<30) . We also saw how to construct them from random samples. The lesson differentiates between the use cases for z-score and t-distribution. We also saw how t value can be used to define the confidence interval based on confidence level. 
