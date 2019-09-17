
# Ridge and Lasso Regression - Lab

## Introduction

In this lab, you'll practice your knowledge on Ridge and Lasso regression!

## Objectives

You will be able to:

- Use Lasso and ridge regression in Python
- Compare Lasso and Ridge with standard regression

## Housing Prices Data

Let's look at yet another house pricing data set.


```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Housing_Prices/train.csv')
```

Look at df.info


```python
# Your code here
```

First, make a selection of the data by removing some of the data with `dtype = object`, this way our first model only contains **continuous features**

Make sure to remove the SalesPrice column from the predictors (which you store in `X`), then replace missing inputs by the median per feature.

Store the target in `y`.


```python
# Load necessary packages


# remove "object"-type features and SalesPrice from `X`


# Impute null values


# Create y

```

Look at the information of `X` again

## Let's use this data to perform a first naive linear regression model

Compute the R squared and the MSE for both train and test set.


```python
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Split in train and test

# Fit the model and print R2 and MSE for train and test

```

## Normalize your data

We haven't normalized our data, let's create a new model that uses `preprocessing.scale` to scale our predictors!


```python
from sklearn import preprocessing

# Scale the data and perform train test split


```

Perform the same linear regression on this data and print out R-squared and MSE.


```python
# Your code here
```

## Include dummy variables

Your model hasn't included dummy variables so far: let's use the "object" variables again and create dummies


```python
# Create X_cat which contains only the categorical variables
```


```python
# Make dummies

```

Merge `x_cat` together with our scaled `X` so you have one big predictor dataframe.


```python
# Your code here
```

Perform the same linear regression on this data and print out R-squared and MSE.


```python
# Your code here
```

Notice the severe overfitting above; our training R squared is quite high, but the testing R squared is negative! Our predictions are far off. Similarly, the scale of the Testing MSE is orders of magnitude higher than that of the training.

## Perform Ridge and Lasso regression

Use all the data (normalized features and dummy categorical variables) and perform Lasso and Ridge regression for both! Each time, look at R-squared and MSE.

## Lasso

With default parameter (alpha = 1)


```python
# Your code here
```

With a higher regularization parameter (alpha = 10)


```python
# Your code here
```

## Ridge

With default parameter (alpha = 1)


```python
# Your code here
```

With default parameter (alpha = 10)


```python
# Your code here
```

## Look at the metrics, what are your main conclusions?   

Conclusions here

## Compare number of parameter estimates that are (very close to) 0 for Ridge and Lasso

Compare with the total length of the parameter space and draw conclusions!


```python
# number of Ridge params almost zero
```


```python
# number of Lasso params almost zero
```

Lasso was very effective to essentially perform variable selection and remove about 25% of the variables from your model!


```python
# your code here
```

## Summary

Great! You now know how to perform Lasso and Ridge regression.
