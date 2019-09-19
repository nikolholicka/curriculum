
# Logistic Regression in scikit-learn - Lab

## Introduction 

In this lab, you are going to fit a logistic regression model to a dataset concerning heart disease. Whether or not a patient has heart disease is indicated in the final column labeled 'target'. 1 is for positive for heart disease while 0 indicates no heart disease.

## Objectives
You will be able to:

* Implement logistic regression in scikit-learn
* Form conclusions about the performance of a model


## Let's get started!


```python
#Starter Code
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
```


```python
#Starter Code
df = pd.read_csv('heart.csv')
df.head()
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
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trestbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalach</th>
      <th>exang</th>
      <th>oldpeak</th>
      <th>slope</th>
      <th>ca</th>
      <th>thal</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Define appropriate X and y
Recall the dataset is whether or not a patient has heart disease and is indicated in the final column labelled 'target'. With that, define appropriate X and y in order to model whether or not a patient has heart disease.


```python
#Your code here 
X = 
y = 
```

## Normalize the Data
Normalize the data prior to fitting the model.


```python
#Your code here
```

## Train Test Split
Split the data into train and test sets.


```python
#Your code here
```

## Fit a model
Fit an initial model to the training set. In scikit-learn you do this by first creating an instance of the regression class. From there, then use the fit method from your class instance to fit a model to the training data.


```python
logreg = LogisticRegression(fit_intercept = False, C = 1e12) #Starter code
# Your code here
```

## Predict
Generate predictions for the train and test sets. Use the **predict** method from the logreg object.


```python
# Your code here
```

## Initial Evaluation
How many times was the classifier correct for the training set?


```python
# Your code here
```

## How many times was the classifier correct for the test set?


```python
# Your code here
```

## Analysis
Describe how well you think this initial model is performing based on the train and test performance. Within your description, make note of how you evaluated performance as compared to your previous work with regression.


```python
# Your analysis here
```

## Summary

In this lab, you practiced a standard data science pipeline: importing data, splitting into train and test sets, and fitting a logistic regression model. In the upcoming labs and lessons, you'll continue to investigate how to analyze and tune these models for various scenarios.
