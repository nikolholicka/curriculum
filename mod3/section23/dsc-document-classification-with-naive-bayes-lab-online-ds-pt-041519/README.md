
# Document Classification with Naive Bayes - Lab

## Introduction

In this lecture, you'll practice implementing the Naive Bayes algorithm on your own.

## Objectives

You will be able to:  

* Implement document classification using Naive Bayes
* Understand the need for the Laplacian smoothing correction
* Explain how to code a bag of words representation

## Import the Dataset

To start, import the dataset stored in the text file `SMSSpamCollection`.


```python
#Your code here
```

## Account for Class Imbalance

To help your algorithm perform more accurately, subset the dataset so that the two classes are of equal size. To do this, keep all of the instances of the minority class (spam) and subset examples of the majority class (ham) to an equal number of examples.


```python
#Your code here
```

## Train - Test Split

Now implement a train test split on your dataset.


```python
from sklearn.model_selection import train_test_split
```

## Create the Word Frequency Dictionary for Each Class

Create a word frequency dictionary for each class.


```python
#Your code here
```

## Count the Total Corpus Words
Calculate V, the total number of words in the corpus.


```python
#Your code here
```

## Create a Bag of Words Function

Before implementing the entire Naive Bayes algorithm, create a helper function `bag_it()` to create a bag of words representation from a document's text.


```python
#Your code here
```

## Implementing Naive Bayes

Now, implement a master function to build a naive Bayes classifier. Be sure to use the logarithmic probabilities to avoid underflow.


```python
#Your code here
```

## Test Out Your Classifier

Finally, test out your classifier and measure its accuracy. Don't be perturbed if your results are sub-par; industry use cases would require substantial additional preprocessing before implementing the algorithm in practice.


```python
#Your Code here
```

## Level-Up

Rework your code into an appropriate class structure so that you could easily implement the algorithm on any given dataset.

## Summary

Well done! In this lab, you practiced implementing Naive Bayes for document classification!
