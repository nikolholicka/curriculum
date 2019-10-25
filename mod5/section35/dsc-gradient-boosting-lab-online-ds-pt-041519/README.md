
# Gradient Boosting - Lab

## Introduction

In this lab, we'll learn how to use both Adaboost and Gradient Boosting Classifiers from scikit-learn!

## Objectives

You will be able to:

* Compare and contrast Adaboost and Gradient Boosting
* Use Adaboost to make predictions on a dataset
* Use Gradient Boosting to make predictions on a dataset

## Getting Started

In this lab, we'll learn how to use Boosting algorithms to make classifications on the [Pima Indians Dataset](http://ftp.ics.uci.edu/pub/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names). You will find the data stored within the file `pima-indians-diabetes.csv`. Our goal is to use boosting algorithms to classify each person as having or not having diabetes. Let's get started!

We'll begin by importing everything we need for this lab. In the cell below:

* Import `numpy`, `pandas`, and `matplotlib.pyplot`, and set the standard alias for each. Also set matplotlib visualizations to display inline. 
* Set a random seed of `0` by using `np.random.seed(0)`
* Import `train_test_split` and `cross_val_score` from `sklearn.model_selection`
* Import `AdaboostClassifier` and `GradientBoostingClassifier` from `sklearn.ensemble`
* Import `accuracy_score`, `f1_score`, `confusion_matrix`, and `classification_report` from `sklearn.metrics`


```python
# Your code here
```

Now, use pandas to read in the data stored in `pima-indians-diabetes.csv` and store it in a DataFrame. Display the head to inspect the data we've imported and ensure everything loaded correctly. 


```python
df = None
```

## Cleaning, Exploration, and Preprocessing

The target we're trying to predict is the `'Outcome'` column. A `1` denotes a patient with diabetes. 

By now, you're quite familiar with exploring and preprocessing a dataset, so we won't hold your hand for this step. 

In the following cells:

* Store our target column in a separate variable and remove it from the dataset
* Check for null values and deal with them as you see fit (if any exist)
* Check the distribution of our target
* Scale the dataset
* Split the dataset into training and testing sets, with a `test_size` of `0.25`


```python
target = None
```


```python
# Your code here
```


```python
# Your code here
```


```python
X_train, X_test, y_train, y_test = None
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-11-f223888184cb> in <module>()
    ----> 1 X_train, X_test, y_train, y_test = None
    

    TypeError: 'NoneType' object is not iterable


## Training the Models

Now that we've cleaned and preprocessed our dataset, we're ready to fit some models!

In the cell below:

* Create an `AdaBoostClassifier`
* Create a `GradientBoostingClassifer`


```python
adaboost_clf = None
gbt_clf = None
```

Now, train each of the classifiers using the training data.


```python
# Your code here - Adaboost classifier
```


```python
# Your code here - Gradient Boosting classifier
```

Now, let's create some predictions using each model so that we can calculate the training and testing accuracy for each.


```python
adaboost_train_preds = None
adaboost_test_preds = None
gbt_clf_train_preds = None
gbt_clf_test_preds = None
```

Now, complete the following function and use it to calculate the training and testing accuracy and f1-score for each model. 


```python
def display_acc_and_f1_score(true, preds, model_name):
    acc = None
    f1 = None
    print("Model: {}".format(None))
    print("Accuracy: {}".format(None))
    print("F1-Score: {}".format(None))
    
print("Training Metrics")
display_acc_and_f1_score(y_train, adaboost_train_preds, model_name='AdaBoost')
print("")
display_acc_and_f1_score(y_train, gbt_clf_train_preds, model_name='Gradient Boosted Trees')
print("")
print("Testing Metrics")
display_acc_and_f1_score(y_test, adaboost_test_preds, model_name='AdaBoost')
print("")
display_acc_and_f1_score(y_test, gbt_clf_test_preds, model_name='Gradient Boosted Trees')
```

    Training Metrics
    Model: None
    Accuracy: None
    F1-Score: None
    
    Model: None
    Accuracy: None
    F1-Score: None
    
    Testing Metrics
    Model: None
    Accuracy: None
    F1-Score: None
    
    Model: None
    Accuracy: None
    F1-Score: None


Let's go one step further and create a confusion matrix and classification report for each. Do so in the cell below.


```python
adaboost_confusion_matrix = None
adaboost_confusion_matrix
```


```python
gbt_confusion_matrix = None
gbt_confusion_matrix
```


```python
adaboost_classification_report = None
print(adaboost_classification_report)
```

    None



```python
gbt_classification_report = None
print(gbt_classification_report)
```

    None


**_Question:_** How did the models perform? Interpret the evaluation metrics above to answer this question.

Write your answer below this line:
_______________________________________________________________________________________________________________________________

 
 
As a final performance check, let's calculate the `cross_val_score` for each model! Do so now in the cells below. 

Recall that to compute the cross validation score, we need to pass in:

* A classifier
* All training Data
* All labels
* The number of folds we want in our cross validation score. 

Since we're computing cross validation score, we'll want to pass in the entire (scaled) dataset, as well as all of the labels. We don't need to give it data that has been split into training and testing sets because it will handle this step during the cross validation. 

In the cells below, compute the mean cross validation score for each model. For the data, use our `scaled_df` variable. The corresponding labels are in the variable `target`. Also set `cv=5`.


```python
print('Mean Adaboost Cross-Val Score (k=5):')
print(None)
# Expected Output: 0.7631270690094218
```

    Mean Adaboost Cross-Val Score (k=5):
    None



```python
print('Mean GBT Cross-Val Score (k=5):')
print(None)
# Expected Output: 0.7591715474068416
```

    Mean GBT Cross-Val Score (k=5):
    None


These models didn't do poorly, but we could probably do a bit better by tuning some of the important parameters such as the **_Learning Rate_**. 

## Summary

In this lab, we learned how to use scikit-learn's implementations of popular boosting algorithms such as AdaBoost and Gradient Boosted Trees to make classification predictions on a real-world dataset!
