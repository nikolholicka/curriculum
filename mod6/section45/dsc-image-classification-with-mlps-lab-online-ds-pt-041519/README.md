
# Deep Networks: Building an Image Classifier - Lab

## Introduction

For the final lab in this section, we'll build a more advanced **_Multi-Layer Perceptron_** to solve image classification for a classic dataset, MNIST!  This dataset consists of thousands of labeled images of handwritten digits, and it has a special place in the history of Deep Learning. 

## Packages

First, let's import all the packages that you 'll need for this lab.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
```

##  The data 

Before we get into building the model, let's load our data and take a look at a sample image and label. 

The MNIST dataset is often used for benchmarking model performance in the world of AI/Deep Learning research. Because it's commonly used, Keras actually includes a helper function to load the data and labels from MNIST--it even loads the data in a format already split into training and testing sets!

Run the cell below to load the MNIST dataset. Note that if this is the first time you've worked with MNIST through Keras, this will take a few minutes while Keras downloads the data. 


```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

Great!  

Now, let's quickly take a look at an image from the MNIST dataset--we can visualize it using matplotlib. Run the cell below to visualize the first image and its corresponding label. 


```python
sample_image = X_train[0]
sample_label =y_train[0]
display(plt.imshow(sample_image))
print("Label: {}".format(sample_label))
```

Great! That was easy. Now, we'll see that preprocessing image data has a few extra steps in order to get it into a shape where an MLP can work with it. 

## Preprocessing Images For Use With MLPs

By definition, images are matrices--they are a spreadsheet of pixel values between 0 and 255. We can see this easily enough by just looking at a raw image:


```python
sample_image
```

This is a problem in its current format, because MLPs take their input as vectors, not matrices or tensors. If all of the images were different sizes, then we would have a more significant problem on our hands, because we'd have challenges getting each image reshaped into a vector the exact same size as our input layer. However, this isn't a problem with MNIST, because all images are black white 28x28 pixel images. This means that we can just concatenate each row (or column) into a single 784-dimensional vector! Since each image will be concatenated in the exact same way, positional information is still preserved (e.g. the pixel value for the second pixel in the second row of an image will always be element number 29 in the vector). 

Let's get started. In the cell below, print the `.shape` of both `X_train` and `X_test`

We can interpret these numbers as saying "X_train consists of 60,000 images that are 28x28". We'll need to reshape them from `(28, 28)`,a 28x28 matrix, to `(784,)`, a 784-element vector. However, we need to make sure that the first number in our reshape call for both `X_train` and `X_test` still correspond to the number of observations we have in each. 

In the cell below:

* Use the `.reshape()` method to reshape X_train. The first parameter should be `60000`, and the second parameter should be `784`.
* Similarly, reshape `X_test` to `10000` and `784`. 
* Also, chain both `.reshape()` calls with an `.astype("float32")`, so that we can our data from type `uint8` to `float32`. 


```python
X_train = None
X_test = None
```

Now, let's check the shape of our training and testing data again to see if it worked. 

Great! Now, we just need to normalize our data!

## Normalizing Image Data

Anytime we need to normalize image data, there's a quick hack we can use to do so easily. Since all pixel values will always be between 0 and 255, we can just scale our data by dividing every element by 255! Run the cell below to do so now. 


```python
X_train /= 255.
X_test /= 255.
```

Great! We've now finished preprocessing our image data. However, we still need to deal with our labels. 

## Preprocessing our Labels

Let's take a quick look at the first 10 labels in our training data:

As we can see, the labels for each digit image in the training set are stored as the corresponding integer value--if the image is of a 5, then the corresponding label will be `5`. This means that this is a **_Multiclass Classification_** problem, which means that we need to **_One-Hot Encode_** our labels before we can use them for training. 

Luckily, Keras provides a really easy utility function to handle this for us. 

In the cell below: 

* Use the function `to_categorical()` to one-hot encode our labels. This function can be found inside `keras.utils`. Pass in the following parameters:
    * The object we want to one-hot encode, which will be `y_train` or `y_test`
    * The number of classes contained in the labels, `10`.


```python
y_train = None
y_test = None
```

Great. Now, let's examine the label for the first data point, which we saw was `5` before. 

Perfect! As we can see, the index corresponding to the number `5` is set to `1`, which everything else is set to `0`. That was easy!  Now, let's get to the fun part--building our model!

## Building Our Model

For the remainder of this lab, we won't hold your hand as much--flex your newfound keras muscles and build an MLP with the following specifications:

* A `Dense` hidden layer with `64` neurons, and a `'tanh'` activation function. Also, since this is the first hidden layer, be sure to also pass in `input_shape=(784,)` in order to create a correctly-sized input layer!
* Since this is a multiclass classification problem, our output layer will need to be a `Dense` layer where the number of neurons is the same as the number of classes in the labels. Also, be sure to set the activation function to `'softmax'`.

## Data Exploration and Normalization

Be sure to carefully review the three code blocks below. Here, we demonstrate some common data checks you are apt to perform after importing, followed by standard data normalization to set all values to a range between 0 and 1.


```python
model_1  = None

```

Now, compile your model with the following parameters:

* `loss='categorical_crossentropy'`
* `optimizer='sgd'`
* `metrics = ['accuracy']`

Let's quickly inspect the shape of our model before training it and see how many training parameters we have. In the cell below, call the model's `.summary()` method. 

50,890 trainable parameters! Note that while this may seem large, deep neural networks in production may have hundreds or thousands of layers and many millions of trainable parameters!

Let's get on to training. In the cell below, fit the model. Use the following parameters:

* Our training data and labels
* `epochs=5`
* `batch_size=64`
* `validation_data=(X_test, y_test)`


```python
results_1 = None
```

## Visualizing Our Loss and Accuracy Curves

Now, let's inspect the model's performance and see if we detect any overfitting or other issues. In the cell below, create two plots:

* The `loss` and `val_loss` over the training epochs
* The `acc` and `val_acc` over the training epochs

**_HINT:_** Consider copying over the visualization function from the previous lab in order to save time!


```python
def visualize_training_results(results):
    pass
```

Pretty good! Note that since our validation scores are currently higher than our training scores, its extremely unlikely that our model is overfitting the training data. This is a good sign--that means that we can probably trust the results that our model is ~91.7% accurate at classifying handwritten digits!

## Building a Bigger Model

Now, let's add another hidden layer and see how this changes things. In the cells below, create a second model. This model should have the following architecture:

* Input layer and first hidden layer same as `model_1`
* Another `Dense` hidden layer, this time with `32` neurons and a `'tanh'` activation function
* An output layer same as `model_1`. 

Build this model in the cell below.


```python
model_2 = None

```

Let's quickly inspect the `.summary()` of the model again, to see how many new trainable parameters this extra hidden layer has introduced.

This model isn't much bigger, but the layout means that the 2080 parameters in the new hidden layer will be focused on higher layers of abstraction than the first hidden layer. Let's see how it compares after training. 

In the cells below, compile and fit the model using the same parameters as we did for `model_1`.


```python
results_2 = None
```

Now, visualize the plots again. 

Slightly better validation accuracy, with no evidence of overfitting--great! If you run the model for more epochs, you'll see the model continue to improve performance, until the validation metrics plateau and the model begins to overfit the training data. 

## A Bit of Tuning

As a final exercise, let's see what happens to the model's performance if we switch activation functions from `'tanh'` to `'relu'`. In the cell below, recreate  `model_2`, but replace all `'tanh'` activations with `'relu'`. Then, compile, train, and plot the results using the same parameters as the other two. 


```python
model_3 = None

```


```python
results_3 = None
```

Performance improved even further! ReLU is one of the most commonly used activation functions around right now--it's especially useful in computer vision problems like image classification, as we've just seen. 

## Summary

In this lab, you once again practiced and reviewed the process of building a neural network. This time, we built a more complex network with additional layers which improved the performance on our data set with MNIST images! 

