
# Regression with CART Trees - Lab

## Introduction

In this lab, we'll make use of what we learned in the previous lesson to build a model for the ["Petrol Consumption Dataset"](https://www.kaggle.com/harinir/petrol-consumption) from Kaggle. This model will be used to predict gasoline consumption for a bunch of examples, based on drivers' features.

## Objectives
You will be able to:
- Conduct a regression experiment using CART trees
- Evaluate the model fit and study the impact of hyper parameters on the final tree
- Understand training, prediction, evaluation and visualizations required to run regression experiments using trees

## Import necessary libraries 


```python
# Import libraries 
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
```

## Read the dataset `petrol_consumption.csv` and view its head and dimensions


```python
# Read the dataset and view head and dimensions

# Code here
```

    (48, 5)





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
      <th>Petrol_tax</th>
      <th>Average_income</th>
      <th>Paved_Highways</th>
      <th>Population_Driver_licence(%)</th>
      <th>Petrol_Consumption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.0</td>
      <td>3571</td>
      <td>1976</td>
      <td>0.525</td>
      <td>541</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>4092</td>
      <td>1250</td>
      <td>0.572</td>
      <td>524</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>3865</td>
      <td>1586</td>
      <td>0.580</td>
      <td>561</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.5</td>
      <td>4870</td>
      <td>2351</td>
      <td>0.529</td>
      <td>414</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>4399</td>
      <td>431</td>
      <td>0.544</td>
      <td>410</td>
    </tr>
  </tbody>
</table>
</div>



## Check the basic statistics for the dataset and inspect the target variable `Petrol_Consumption`


```python
# Describe the dataset

# Code here
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
      <th>Petrol_tax</th>
      <th>Average_income</th>
      <th>Paved_Highways</th>
      <th>Population_Driver_licence(%)</th>
      <th>Petrol_Consumption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>48.000000</td>
      <td>48.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.668333</td>
      <td>4241.833333</td>
      <td>5565.416667</td>
      <td>0.570333</td>
      <td>576.770833</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.950770</td>
      <td>573.623768</td>
      <td>3491.507166</td>
      <td>0.055470</td>
      <td>111.885816</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.000000</td>
      <td>3063.000000</td>
      <td>431.000000</td>
      <td>0.451000</td>
      <td>344.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.000000</td>
      <td>3739.000000</td>
      <td>3110.250000</td>
      <td>0.529750</td>
      <td>509.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.500000</td>
      <td>4298.000000</td>
      <td>4735.500000</td>
      <td>0.564500</td>
      <td>568.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.125000</td>
      <td>4578.750000</td>
      <td>7156.000000</td>
      <td>0.595250</td>
      <td>632.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>5342.000000</td>
      <td>17782.000000</td>
      <td>0.724000</td>
      <td>968.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Create features, labels and train/test datasets with a 80/20 split

As with the classification task, we will divide our data into attributes/features and labels and consequently into training and test sets.


```python
# Create datasets for training and test


# Code here
```

## Create an instance of CART regressor and fit the data to the model 

As mentioned earlier, for a regression task we'll use a different `sklearn` class than we did for the classification task. The class we'll be using here is the `DecisionTreeRegressor` class, as opposed to the `DecisionTreeClassifier` from before.


```python
# Train a regression tree model with training data 


# Code here 
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, min_samples_leaf=1,
                          min_samples_split=2, min_weight_fraction_leaf=0.0,
                          presort=False, random_state=0, splitter='best')



 ## Using test set, make predictions and calculate the MAE, MSE and RMSE
 
Just as with Decision Trees for classification, there are several commonly used metrics for evaluating the performance of our model. The most common metrics are:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

If these look familiar, it's likely because you have already seen them before--they are common evaluation metrics for any sort of regression model, and as we can see, regressions performed with Decision Tree models are no exception!
 
Since these are common evaluation metrics, sklearn has functions for each of them that we can use to make our job easier. You'll find these functions inside the `metrics` module. In the cell below, calculate each of the three evaluation metrics listed above!


```python
# Predict and evaluate the predictions


# Code here
```

    Mean Absolute Error: 50.8
    Mean Squared Error: 4535.4
    Root Mean Squared Error: 67.34537846058926


## Level Up - Optional 

- In order to understand and interpret a tree structure, we need some domain knowledge in which the data was generated. That can help us inspect each leaf and investigate/prune the tree based on qualitative analysis. 

- Look at the hyper parameters used in the regression tree, check their values ranges in official doc and try running some optimization by growing a number of trees in a loop. 

- Use a dataset that you are familiar with and run tree regression to see if you can interpret the results.

- Check for outliers, try normalization and see the impact on the output 

## Summary 

In this lesson, we developed a tree regressor architecture to train the regressor and predict values for unseen data. We saw that with a vanilla approach, the results were not so great, and this requires further pre-tuning of the model (what we described as hyper parameter optimization OR pruning in the case of trees. 
