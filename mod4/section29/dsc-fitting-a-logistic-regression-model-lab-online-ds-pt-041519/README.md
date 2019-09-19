
# Fitting a Logistic Regression Model - Lab

## Introduction
You were previously given a broad overview of logistic regression. This included two separate packages for creating logistic regression models. In this lab, you'll be investigating fitting logistic regressions with statsmodels.



## Objectives

You will be able to:
* Implement logistic regression with statsmodels
* Interpret the statistical results associated with regression model parameters


## Review

The stats model example we covered had four essential parts:
* Importing the data
* Defining X and y
* Fitting the model
* Analyzing model results

The corresponding code to these four steps was:

```
import pandas as pd
import statsmodels.api as sm

#Step 1: Importing the data
salaries = pd.read_csv("salaries_final.csv", index_col = 0)

#Step 2: Defining X and y
x_feats = ["Race", "Sex", "Age"]
X = pd.get_dummies(salaries[x_feats], drop_first=True, dtype=float)
y = pd.get_dummies(salaries["Target"], dtype=float)

#Step 3: Fitting the model
X = sm.add_constant(X)
logit_model = sm.Logit(y.iloc[:,1], X)
result = logit_model.fit()

#Step 4: Analyzing model results
result.summary()
```

Most of this should be fairly familiar to you; importing data with Pandas, initializing a regression object, and calling the fit method of that object. However, step 2 warrants a slightly more in depth explanation.

Recall that we fit the salary data using `Race`, `Sex`, and `Age`. Since `Race` and `Sex` are categorical, we converted them to dummy variables using the `get_dummies()` method. The ```get_dummies()``` method will only convert `object` and `category` data types to dummy variables so it is safe to pass `Age`. Note that we also passed two additional arguments, ```drop_first=True``` and ```dtype=float```. The ```drop_first=True``` argument removes the first level for each categorical variable and the ```dtype=float``` argument converts the data type of all of the dummy variables to float. The data must be float in order to obtain accurate statistical results from statsmodel. Finally, note that y itself returns a pandas DataFrame with two columns as y itself was originally a categorical variable. With that, it's time to try and define a logistic regression model on your own!

## Your Turn - Step 1: Import the Data

Import the data stored in the file **titanic.csv**.


```python
# Your code here
```

## Step 2: Define X and Y

For your first foray into logistic regression, you are going to attempt to build a model that classifies whether an individual survived the Titanic shipwreck or not (yes it's a bit morbid). Follow the programming patterns described above to define X and y.


```python
# Your code here
```

## Step 3: Fit the model

Now with everything in place, initialize a regression object and fit your model!

### Warning: If you receive an error of the form "LinAlgError: Singular matrix"

Stats models was unable to fit the model due to some Linear Algebra problems. Specifically, the matrix was not invertible due to not being full rank. In layman's terms, there was a lot of redundant, superfluous data. Try removing some features from the model and running it again.


```python
# Your code here
```

## Step 4: Analyzing results

Generate the summary table for your model. Then, comment on the p-values associated with the various features you chose.


```python
# Your code here
```

## Your analysis here

## Level - up

Create a new model, this time only using those features you determined were influential based on your analysis in step 4.


```python
# Your code here
```

## Summary 

Well done! In this lab, you practiced using stats models to build a logistic regression model. You then reviewed interpreting the results, building upon your previous stats knowledge, similar to linear regression. Continue on to take a look at building logistic regression models in Sci-kit learn!
