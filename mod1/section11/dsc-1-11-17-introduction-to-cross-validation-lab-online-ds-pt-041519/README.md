
# Introduction to Cross-Validation - Lab

## Introduction

In this lab, you'll be able to practice your cross-validation skills!


## Objectives

You will be able to:

- Compare the results with normal holdout validation
- Apply 5-fold cross validation for regression

## Let's get started

This time, let's only include the variables that were previously selected using recursive feature elimination. We included the code to preprocess below.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_boston

boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
b = boston_features["B"]
logdis = np.log(boston_features["DIS"])
loglstat = np.log(boston_features["LSTAT"])

# minmax scaling
boston_features["B"] = (b-min(b))/(max(b)-min(b))
boston_features["DIS"] = (logdis-min(logdis))/(max(logdis)-min(logdis))

#standardization
boston_features["LSTAT"] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
```


```python
X = None
y = None
```

## Train test split

Perform a train-test-split with a test set of 0.20.


```python
from sklearn.model_selection import train_test_split
```

Fit the model and apply the model to the make test set predictions

Calculate the residuals and the mean squared error

## Cross-Validation: let's build it from scratch!

### Create a cross-validation function

Write a function k-folds that splits a dataset into k evenly sized pieces.
If the full dataset is not divisible by k, make the first few folds one larger then later ones.

We want the folds to be a list of subsets of data!


```python
def kfolds(data, k):
    # Force data as pandas dataframe
    # add 1 to fold size to account for leftovers           
    return None
```

### Apply it to the Boston Housing Data


```python
# Make sure to concatenate the data again
```

### Perform a linear regression for each fold, and calculate the training and test error

Perform linear regression on each and calculate the training and test error.


```python
test_errs = []
train_errs = []
k=5

for n in range(k):
    # Split in train and test for the fold
    train = None
    test = None
    # Fit a linear regression model
    
    #Evaluate Train and Test Errors

# print(train_errs)
# print(test_errs)
```

## Cross-Validation using Scikit-Learn

This was a bit of work! Now, let's perform 5-fold cross-validation to get the mean squared error through scikit-learn. Let's have a look at the five individual MSEs and explain what's going on.

Next, calculate the mean of the MSE over the 5 cross-validations and compare and contrast with the result from the train-test-split case.

##  Summary 

Congratulations! You now practiced your knowledge on k-fold crossvalidation!
