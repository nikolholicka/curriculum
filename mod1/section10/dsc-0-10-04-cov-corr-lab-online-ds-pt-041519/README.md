
# Covariance and Correlation - Lab

In this lab, we shall working towards calculating covariance and correlation for a given dataset in python. We shall use the formulas shown in previous lesson and verify our results with python libraries.

## Objectives

You will be able to 
* Calculate and and interpret correlation and covariance for given variables
* Build density and scatter plots to visually identify the level of dependence between variables
* Perform covariance and correlation using python and numpy 

### Dataset

Included dataset (heightWeight.csv) includes 20 heights (inches) and weights(pounds). Yes, it is a particularly small dataset and will help us focus more on seeing covariance and correlation in action. At this point, you should be able to calculate the average height and average weight. You can also explain the medians, variances and standard deviations for this dataset.

But all of those measurements are only concerned with a **single variable**. What if we want to see: 

How height interacts with weight ? 

Does weight increase as height increases ?

Are Weight and Height not related at all ?

Note while there are plenty of fat short people and overly skinny tall people, but when you look at the population at large, taller people will tend to weigh more than shorter people. This generalization of information is very common as it shows you a bigger picture that you can build your intuitions upon.

Let's first load this dataset into pandas. Read the file "heightWeight.csv" and for header, length of the records and basic stats. 


```python
# Load the dataset into pandas and perform basic inspection


# 20
#    height  Weight
# 0      68     165
# 1      71     201
# 2      61     140
# 3      69     170
# 4      71     192
#           height      Weight
# count  20.000000   20.000000
# mean   66.850000  165.800000
# std     5.112163   28.971129
# min    58.000000  115.000000
# 25%    63.250000  143.750000
# 50%    68.500000  170.000000
# 75%    71.000000  192.750000
# max    74.000000  210.000000
```

### Calculate covariance 

Here's the covariance formula once again. 

![](cov2.png)

We would use (n-1) due to the fact that we are working with samples of a bigger population here. 

#### Mean normalization 

But before we do this, we have to ensure the that both variables are **Mean Normalized** (as shown in the numerator above). i.e. both variables have mean values = 0 . This allows us to calculate how much they vary while disregarding their distance from each other. A bit like standardization that we saw before, but here we are not standardizing the spread (standard deviation), as that is what needs to be studied. So the formula to mean normalize a data set is : 

> **xi - X(mean)**

Pretty simple, take each element of the variable and subtract the mean value from it. This will create a new "mean-normalized" dataset. Let's write a function that takes in a vector, calculates the mean of vector and subtracts the calculated mean value from each element to calculate xi - X(mean). 

Hint: use `np.mean()` to calculate the mean for above formula 


```python
import numpy as np

# Write a function to take in an iterable, calculate the mean and subtract the mean value
# from each element , creating and returning a new list. 

def mean_normalize(var):

    pass

mean_normalize([1,2,3,4,5]), mean_normalize([11,22,33,44,55])

# ([-2.0, -1.0, 0.0, 1.0, 2.0], [-22.0, -11.0, 0.0, 11.0, 22.0])
```




    (None, None)



Great so you see, our function maintains the variance of list elements and moves their mean to zero. As a quick test, we can visualize what exactly happens to the data with mean normalization. Plot the height variable distribution before and after the normalization process. 


```python
# Visualize the height data distribution before and after mean normalization 

```


```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b9ee668>




![png](index_files/index_7_1.png)


So there you go, not much changes in the shape of the data. Try repeating above with weight. 

#### The dot product

So now that we have our new normalized datasets. According to the numerator in the formula,we have to take the **DOT PRODUCT** of these two vector values. 
> A dot product is a linear algebraic operation that takes two equal-length sequences of numbers and returns a single number which can be used as a measure of similarity between these sequences (also known as vectors).

[Here is a great article explaining this in detail](https://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/). We will cover more of this in our section dedicated to linear algebra. 

For two vectors a and b, a dot product is calculated by multiplying each element of one vector to its counterpart in the second , and then adding them up together.  
```
 a[0] * b[0] + a[1] * b[1] + a[2] * b[2] ...

```

So lets write a function that will take two iterables and return their dot product. 


```python
# Write a function to calculate the dot product of two iterables 

def dot_product(x,y):
    
    
    pass

a = [1,2,3]
b = [4,5,6]

dot_product(a,b)

#  32  calculated as (1*4 + 2*5 + 3*6)
```

So we have the numerator of the formula sorted out. Let's finally write a function `covariance()` that will take height and weight lists we created earlier and return the covariance value using the functions we created earlier. 


```python
# Calculate covariance using functions above

def covariance(var1, var2):

    pass

# Uncomment below to check your function

# covariance(data['height'], data['Weight'])

# 144.75789473684208
```

Let's verify our results with pandas built in `dataFrame.cov()` method.


```python
# uncomment to run
# data.cov()
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
      <th>height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>height</th>
      <td>26.134211</td>
      <td>144.757895</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>144.757895</td>
      <td>839.326316</td>
    </tr>
  </tbody>
</table>
</div>



Okay so covariance (as well as correlation) are usually shown in matrix form. the covariance between height and weight is exactly what we calculated. the matrix also shows the covariance of a variable with itself. So this gives us magnitude which is a bit hard to interpret. How about we visualize height and weight on a scatter plot ! 


```python
# Plot a scatter graph between height and weight to visually inspect the relationship 
```


```python

```




    <matplotlib.collections.PathCollection at 0x1a1ba379e8>




![png](index_files/index_16_1.png)


So we can see there is quite a bit of positive relationship between the two, but a covariance value is a bit hard to interpret. So let's try calculating correlation. 

### Calculate Correlation

Once again, heres the formula to calculate the correlation. 
![](cor.png)

lots of mean normalizations going on here. It shouldn't be too hard now to implement this using our functions above.


```python
# Calculate Correlation between two variables using formula above
import math
def correlation(var1,var2):
    
    pass

# correlation(data['height'], data['Weight'])

# 0.98
```

Wow, 0.98, thats very close to one. So that means height and weight are like TOTALLY dependent on each other. Well, only for this particular sample. And there is a takeaway in this. sample size plays a major rule in determining the nature of a variable and its relationship with other variables. the set of 20 records we have seem to correlate highly, but this might be different for a different set of samples. We shall talk about how to further test such a finding to either reject it , or confirm it as a FACT. 

As a last check , let's use pandas `dataframe.corr()` method to see how that works. 


```python
# uncomment to run
# data.corr()
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
      <th>height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>height</th>
      <td>1.0000</td>
      <td>0.9774</td>
    </tr>
    <tr>
      <th>Weight</th>
      <td>0.9774</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



Another matrix similar to above. And we see that a correlation of a variable to itself will always be = 1. The correlation between height and weight can be rounded off to our results. That is great. Now we know how this works. 

## Summary 

In this lab we saw how to calculate the covariance and correlation between variables. We also looked at mean normalization and dot products which will be revisited later in the course. FInally we saw how to calculate these measures using pandas built in methods. 
