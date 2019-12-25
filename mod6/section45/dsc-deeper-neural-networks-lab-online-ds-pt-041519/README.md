
# Deeper Neural Networks - Lab

## Introduction

In this lesson, we'll dig deeper into the work horse of deep learning, **_Multi-Layer Perceptrons_**! We'll build and train a couple different MLPs with Keras and explore the tradeoffs that come with adding extra hidden layers. We'll also try switching out some of the activation functions we learned about in the previous lesson to see how they affect training and performance. 

## Getting Started

We'll begin by importing everything we need for this lab. Run the cell below 
to import everything we'll need for this lab. 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.datasets import boston_housing, mnist
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
```

For this lab, we'll be working with the [Boston Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). Although we're importing this dataset directly from sklearn, the kaggle link above contains a detailed explanation of the dataset, in case you're interested. We recommend taking a minute to familiarize yourself with the dataset before digging in. 

In the cell below:

* Call `load_breast_cancer()` to store the dataset object. 
* Get the `.data`, `.target`, and `.feature_names` and store them in the appropriate variables below.


```python
bc_dataset = None
data = None
target = None
col_names = None
```

Now, let's create a dataframe so that we can see the data and explore it a bit more easily with the column names attached. 

In the cell below, create a pandas dataframe and pass in the `data`. Also pass in the `col_names` to the `columns` parameter when creating the dataframe. Then, print the head of the dataframe. 


```python
df = None
# df.head()
```

## Getting the Data Ready for Deep Learning

In order to pass this data into a neural network, we'll need to make sure that the data:

* Is purely numerical
* contains no null values
* Is normalized 

Let's begin by calling the dataframe's `.info()` method to check the datatype of each feature. 


```python

```

From the output above, we can see that the entire dataset is already in numerical format. We can also see from the counts that each feature has the same number of entries as the number of rows in the dataframe--that means that no feature contains any null values. Great!

Now, let's check to see if our data needs to be normalized. Instead of doing statistical tests here, let's just take a quick look at the head of the dataframe again. Do this in the cell below. 


```python

```

As we can clearly see from comparing features like `mean radius` and `mean area`, columns have different scales, which means that we need to normalize our dataset. To do this, we'll make use of sklearn's `StandardScaler()` object. 

In the cell below, use create a StandardScaler object and use it to create a normalized version of our dataset. 


```python
scaler = None
scaled_data = None
```

## Binarizing Our Labels

If you took a look at the data dictionary on Kaggle, then you probably noticed the target for this dataset is to predict if the sample is "M" (Malignant) or "B" (Benign). This means that this is a **_Binary Classification_** task, so we'll need to binarize our labels. 

In the cell below, make use of sklearn's `LabelBinarizer()` object to create a binarized version of our labels. 


```python
binarizer = None
labels = None
```

## Building our MLP

Now, we'll build a small **_Multi-Layer Perceptron_** using Keras in the cell below. Our first model will act as a baseline, and then we'll make it bigger to see what happens to model performance. 

In the cell below:

* Create our keras model by instantiating a `Sequential()` object. 
* Use the model's `.add()` method to add a `Dense` layer with 10 neurons and a `'tanh'` activation function. Also set the `input_shape` attribute to `(30,)`, since we have 30 features. 
* Since this is a binary classification task, the output layer should be a `Dense` layer with a single neuron, and the activation set to `'sigmoid'`.


```python
model_1 = None

```

### Compiling the Model

Now that we've created the model, we still have to compile it. 

In the cell below, compile the model. Set the following hyperparameters:

* `loss='binary_crossentropy'`
* `optimizer='sgd'`
* `metrics=['accuracy']`


```python

```

### Fitting the Model

Now, let's fit the model. In addition to our scaled data and our labels, set the following hyperparameters:

* `epochs=25`
* `batch_size=1`
* `validation_split=0.2`


```python
results_1 = None
```

Let's quickly plot our validation and accuracy curves and see if we notice anything. Note that when you call a Keras model's `.fit()` method, it returns a Keras callback containing information on the training process of the model. If you examine the callback's `.history` attribute, you'll find a dictionary containing both the training and validation loss, as well as any metrics we specified when compiling the model (in this case, just accuracy). 

In the cell below, let's quickly create a function for visualizing the loss and accuracy metrics. Since we'll want to do this anytime we train an MLP, its worth wrapping this code in a function so that we can easily reuse it. 


```python
def visualize_training_results(results):
    history = results.history
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure()
    plt.plot(history['val_acc'])
    plt.plot(history['acc'])
    plt.legend(['val_acc', 'acc'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
```


```python
visualize_training_results(results_1)
```

## Detecting Overfitting

You'll probably notice that the model did pretty well! It's always recommended to visualize your training and validation metrics against each other after training a model. By plotting them like this, we can easily detect that the model is starting to overfit. We can tell that this is happening by seeing the model's training performance steadily improve long after the validation performance plateaus. We can see that in the plots above as the training loss continues to decrease and the training accuracy continues to increase, and the distance between the two lines gets greater as the epochs gets higher. 

## Iterating on the Model

By adding another hidden layer, we can a given the model the ability to capture high layers of abstraction in th data. However, increasing the depth of the model also increases the amount of data the model needs to converge to answer, because with a more complex model comes the "Curse of Dimensionality", thanks to all the extra trainable parameters that come from adding more size to our network. 

If there is complexity in the data that our smaller model was not big enough to catch, then a larger model may improve performance. However, if our dataset isn't big enough for the new, larger model, then we may see performance decrease as then model "thrashes" about a bit, failing to converge. Let's try and see what happens. 

In the cell below, recreate the model that you created above, with one exception. In the model below, add a second `Dense` layer with `'tanh'` activation functions and `5` neurons after the first. The network's output layer should still be a `Dense` layer with a single neuron and a sigmoid activation function, since this is still a binary classification task. 

Create, compile, and fit the model in the cells below, and then visualize the results to compare the history. 


```python
model_2 = None

```


```python

```


```python
results_2 = None
```


```python
visualize_training_results(results_2)
```

## What Happened?

Although the final validation score for both models is the same, this model is clearly worse because it hasn't converged yet. We can tell because of the greater variance in the movement of the `val_loss` and `val_acc` lines. This suggests that we can remedy this in 1 of 2 ways:

* Decrease the size of the network, OR
* Increase the size of our training data. 

## Visualizing Why we Normalize Our Data

As a final exercise, let's create a 3rd model that is the same as the first model we created for this exercise in every way. The only difference is that we will train it on our raw dataset, not the normalized version. This way, we can see how much of a difference normalizing our input data makes.

Create, compile, and fit a model in the cell below. The only change in parameters will be using `data` instead of `scaled_data` during the `.fit()` step. 


```python
model_3 = None

```


```python

```


```python
results_3 = None
```


```python
visualize_training_results(results_3)
```

Wow! Our results were much worse--over 20% poorer performance when working with non-normalized input data!  


## Summary

In this lab, we got some practice creating **_Multi-Layer Perceptrons_**, and explored how things like the number of layers in a model and data normalization affect our overall training results!
