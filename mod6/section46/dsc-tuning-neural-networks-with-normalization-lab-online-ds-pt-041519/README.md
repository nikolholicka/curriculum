
# Tuning Neural Networks with Normalization - Lab

## Introduction

For this lab on initialization and optimization, you'll build a neural network to perform a regression task.

It is worth noting that getting regression to work with neural networks can be difficult because the output is unbounded ($\hat y$ can technically range from $-\infty$ to $+\infty$, and the models are especially prone to exploding gradients. This issue makes a regression exercise the perfect learning case for tinkering with normalization and optimization strategies to ensure proper convergence!

## Objectives
You will be able to:
* Build a neural network using Keras
* Normalize your data to assist algorithm convergence
* Implement and observe the impact of various initialization techniques


```python
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import initializers
from keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from keras import optimizers
from sklearn.model_selection import train_test_split
```

    /Users/matthew.mitchell/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


## Loading the data

The data we'll be working with is data related to Facebook posts published during the year of 2014 on the Facebook page of a renowned cosmetics brand.  It includes 7 features known prior to post publication, and 12 features for evaluating the post impact. What we want to do is make a predictor for the number of "likes" for a post, taking into account the 7 features prior to posting.

First, let's import the data set, `dataset_Facebook.csv`, and delete any rows with missing data. Afterwards, briefly preview the data.


```python
#Your code here; load the dataset and drop rows with missing values. Then preview the data.
```

## Defining the Problem

Define X and Y and perform a train-validation-test split.

X will be:
* Page total likes
* Post Month
* Post Weekday
* Post Hour
* Paid
along with dummy variables for:
* Type
* Category

Y will be the `like` column.


```python
#Your code here; define the problem.
```

## Building a Baseline Model

Next, build a naive baseline model to compare performance against is a helpful reference point. From there, you can then observe the impact of various tunning procedures which will iteratively improve your model.


```python
#Simply run this code block, later you'll modify this model to tune the performance
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```

### Evaluating the Baseline

Evaluate the baseline model for the training and validation sets.


```python
#Your code here; evaluate the model with MSE
```


```python
#Your code here; inspect the loss function through the history object
```

> Notice this extremely problematic behavior: all the values for training and validation loss are "nan". This indicates that the algorithm did not converge. The first solution to this is to normalize the input. From there, if convergence is not achieved, normalizing the output may also be required.

## Normalize the Input Data

Normalize the input features by subtracting each feature mean and dividing by the standard deviation in order to transform each into a standard normal distribution. Then recreate the train-validate-test sets with the transformed input data.


```python
## standardize/categorize
```

## Refit the Model and Reevaluate

Great! Now refit the model and once again assess it's performance on the training and validation sets.


```python
#Your code here; refit a model as shown above
```


```python
#Rexamine the loss function
```

> Note that you still haven't achieved convergence! From here, it's time to normalize the output data.

## Normalizing the output

Normalize Y as you did X by subtracting the mean and dividing by the standard deviation. Then, resplit the data into training and validation sets as we demonstrated above, and retrain a new model using your normalized X and Y data.


```python
#Your code here: redefine Y after normalizing the data.
```


```python
#Your code here; create training and validation sets as before. Use random seed 123.
```


```python
#Your code here; rebuild a simple model using a relu layer followed by a linear layer. (See our code snippet above!)
```

Again, reevaluate the updated model.


```python
#Your code here; MSE
```


```python
#Your code here; loss function
```

Great! Now that you have a converged model, you can also experiment with alternative optimizers and initialization strategies to see if you can find a better global minimum. (After all, the current models may have converged to a local minimum.)

## Using Weight Initializers

Below, take a look at the code provided to see how to modify the neural network to use alternative initialization and optimization strategies. At the end, you'll then be asked to select the model which you believe is the strongest.

##  He Initialization


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, kernel_initializer= "he_normal",
                activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val),verbose=0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    1.0392949820359312
    0.8658544030836142


## Lecun Initialization


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, 
                kernel_initializer= "lecun_normal", activation='tanh'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    1.0307351124941144
    0.9292005788570431


Not much of a difference, but a useful note to consider when tuning your network. Next, let's investigate the impact of various optimization algorithms.

## RMSprop


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "rmsprop" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    1.020200641136699
    0.9421919606123765


## Adam


```python
np.random.seed(123)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= "Adam" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    1.0219766410555322
    0.9477664629838952


## Learning Rate Decay with Momentum



```python
np.random.seed(123)
sgd = optimizers.SGD(lr=0.03, decay=0.0001, momentum=0.9)
model = Sequential()
model.add(layers.Dense(8, input_dim=10, activation='relu'))
model.add(layers.Dense(1, activation = 'linear'))

model.compile(optimizer= sgd ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train)
print(MSE_val)
```

    0.8667952792265361
    1.1040802536956849


## Selecting a Final Model

Now, select the model with the best performance based on the training and validation sets. Evaluate this top model using the test set!


```python
#Your code here
```

## Summary  

In this lab, you worked to ensure your model converged properly. Additionally, you also investigated the impact of varying initialization and optimization routines.
