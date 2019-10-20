
# K-Nearest Neighbors - Lab

## Introduction

In this lesson, you'll build a simple version of a **_K-Nearest Neigbors Classifier_** from scratch, and train it to make predictions on a dataset!

## Objectives

You will be able to:

* Implement a basic KNN algorithm from scratch

## Getting Started

You'll begin this lab by creating a classifier. To keep things simple, you'll be using a helper function from the scipy library to calculate euclidean distance&mdash;specifically, the `euclidean()` function from the `scipy.spatial.distance` module. Import this function in the cell below.


```python
from scipy.spatial.distance import euclidean as euc
import numpy as np
np.random.seed(0)
```

Great! Now, you need to define a `KNN` class. Since you don't need to do anything at initialization, you don't need to modify the `__init__` method at all.

In the cell below:

* Create an class called `KNN`.
* This class should contain two empty methods--`fit`, and `predict`. (Set the body of both of these methods to `pass`)

## Completing the `fit` Method

Recall that when "fitting" a KNN classifier, all you're really doing is storing the points and their corresponding labels. There's no actual "fitting" involved here, since all you can do is store the data so that you can use it to calculate the nearest neighbors when the `predict` method is called.

The inputs for this function should be:

* `self`, since this will be an instance method inside the `KNN` class.
* `X_train`--A 2-dimensional array. Each of the internal arrays represents a _vector_ for a given point in space. 
* `y_train`--the corresponding labels for each vector in `X_train`. The label at `y_train[0]` is the label that corresponds to the vector at `X_train[0]`, and so on. 

In the cell below, complete the `fit` method.


```python
def fit(self, X_train, y_train):
    pass
    
# This line updates the knn.fit method to point to the function we've just written
KNN.fit = fit
```

### Helper Functions

Next, write two helper functions to make things easier when completing the `predict` function. The first helper function should return an array containing the distance between a point we pass in and every point inside of `X_train`. 

In the cell below, complete the `_get_distances()` function. This function should:

* Take in two arguments: `self` and `x`
* Create an empty array to hold all the distances we're going to calculate
* Enumerate through every item in `self.X_train`. For each item: 
    * Use the `euc()` function we imported to get the distance between x and the current point from `X_train`.
    * Create a tuple containing the index and the distance (in that order!) and append it to our `distances` array.
* Return the `distances` array when a distance has been generated for each item in `self.X_train`.


```python
def _get_distances(self, x):
    pass

# This line attaches the function we just created as a method to our KNN class.
KNN._get_distances = _get_distances
```

Great! The second big challenge in a `predict` method is getting the indices of the k-nearest points. To keep our coming `predict` method nice and clean, abstract this functionality into a helper method called `_get_k_nearest`.  

In the cell below, complete the `_get_k_nearest` function.  This function should:

* Take in 3 arguments:
    * `self`
    * `dists`, an array of tuples containing (index, distance), which will be output from the `_get_distances` method. 
    * `k`, the number of nearest neighbors we want to return.
* Sort our `dists` array by distances values, which are the second element in each tuple
* Return the first `k` tuples from then (now sorted) `dists` array.

**_Hint:_** To easily sort on the second item in the tuples contained within the `dists` array, use the `sorted` function and pass in lambda for the `key=` parameter. To sort on the second element of each tuple, you can just use `key=lambda x: x[1]`!


```python
def _get_k_nearest(self, dists, k):
    pass

# This line attaches the function we just created as a method to our KNN class.
KNN._get_k_nearest = _get_k_nearest
```

Now, you have helper functions to help you get the distances, and then get the k-nearest neighbors based on those distances. The final helper function you'll create will get the labels that correspond to each of the k-nearest point, and return the class that occurs the most. 

Complete the `_get_label_prediction()` function in the cell below. This function should:

* Create a list containing the labels from `self.y_train` for each index in `k_nearest` (remember, each item in `k_nearest` is a tuple, and the index is stored as the first item in each tuple)
* Get the total counts for each label (use `np.bincount()` and pass in the label array created in the previous step)
* Get the index of the label with the highest overall count in counts (use `np.argmax()` for this, and pass in the counts created in the previous step).


```python
def _get_label_prediction(self, k_nearest):
    pass

# This line attaches the function we just created as a method to our KNN class.
KNN._get_label_prediction = _get_label_prediction
```

Great! Now, you need to complete a `predict` method.

## Completing the `predict` Method

This method does all the heavy lifting for KNN, so this will be a bit more complex than the `fit` method. Here's a rough outline of how the method should work:

1. The function takes in an array of vectors that we want predictions for.
1. For each vector that we want to make a prediction for: 
    1a. The classifier calculates the distance between that vector and every other vector in the training set. 
    1b. The classifier identifies the K nearest vectors to the vector you want a prediction for.
    1c. The classifier determines which label the majority of the K nearest neighbors share, and appends this prediction to an array we will output. The index of the prediction in this array should be the same as the index of the point that it corresponds to (e.g. `pred[0]` is the prediction for `X_test[0]`).
2. Once predictions have been generated for every vector in question, return the array of predictions. 

This tells us a few things about what our `predict` function will need to be able to do:

* In addition to `self`, our `predict` function should take in two arguments: 
    * `X_test`, the points we want to classify
    * `k`, which specifies the number of neighbors we should use to make the classification.  We'll set `k=3` as a default, but allow the user to update it if they choose.
* Your method will need to iterate through every item in `X_test`. For each item:
    * Calculate the distance to all points in `X_train` by using our `_get_distances()` helper method we created.
    * Find the k-nearest points in `X_train` by using the `_get_k_nearest()` helper method we created
    * Use the index values contained within the tuples returned by `_get_k_nearest()` to get the corresponding labels for each of the nearest points. 
    * Determine which class is most represented in these labels and treat that as the prediction for this point. Append the prediction to `preds`.
* Once a prediction has been generated for every item in `X_test`, return `preds`

Follow these instructions to complete the `predict()` method in the cell below!


```python
def predict(self, X_test, k=3):
    pass
        
KNN.predict = predict
```

Great! Now, try out your new KNN classifier on a sample dataset to see how well it works!

## Testing Our KNN Classifier

In order to test the performance of yur model, import the **_Iris Dataset_**. Specifically, use the `load_iris` function, which can be found inside of the `sklearn.datasets` module. Then call this function, and use the object it returns. Also import `train_test_split` from `sklearn.model_selection`, as well as `accuracy_score` from `sklearn.metrics`.  Note that there are **_3 classes_** in the Iris Dataset, making this a multi-categorical classification problem. This means that you can't use evaluation metrics that are meant for binary classification problems. For this, just stick to accuracy for now. 

Run the cell below to import everything you'll need from sklearn to test our model. 


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
data = iris.data
target = iris.target
```

Now, you'll need to use `train_test_split` to split our training data into training and testing sets. Pass in the `data`, the `target`, and set a `test_size` of `0.25`.


```python
X_train, X_test, y_train, y_test = None
```

Now, instantiate a knn object, and `fit` it to the data in `X_train` and the labels in `y_train`.


```python
knn = None
```

Now, create some predictions on the test data.  In the cell below, use the `.predict()` method to generate predictions for the data stored in `X_test`.


```python
preds = None
```

And now, for the moment of truth! Test the accuracy of your predictions. In the cell below, complete the call to `accuracy_score` by passing in `y_test` and our `preds`!


```python
print("Testing Accuracy: {}".format(accuracy_score(None, None)))
# Expected Output: Testing Accuracy: 0.9736842105263158
```

Over 97% accuracy! Not bad for a handwritten machine learning classifier!

## Summary

That was great! Next, you'll dive a little deeper into evaluating performance of a KNN algorithm!
