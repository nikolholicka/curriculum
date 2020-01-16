
# NetworkX Introduction - Lab

## Introduction

In this lab, you'll practice some of the introductory skills for NetworkX introduced in the previous lesson.
To do this, you'll create a graph to visualize users and businesses from yelp reviews.
## Objectives

You will be able to:
* Create basic network graphs using NetworkX
* Add nodes to network graphs with NetworkX
* Add edges to network graphs with NetworkX
* Visualize network graphs with NetworkX

## Import the Data

To start, import the data stored in the file 'Yelp_reviews.csv'


```python
#Your code here
```

## Creating a Graph

Now, create an initial graph!


```python
#Your code here
```

## Adding Nodes

Create a node for each user and each business in the dataset. Networks with multiple node types like this are called **bimodal networks**.

Optionally, go further by creating a list of colors for when you visualize the graph. If you do this, append the color "green" to your color list every time you add a user node and append the color "blue" to your color list every time you add a business node.


```python
#Your code here
```

## Adding Edges

Next, iterate through the dataset and create an edge between users and the businesses they have reviewed.


```python
#Your code here
```

## Visualizing the Graph

Finally, create a visualization of your network. If you chose to color your nodes, pass the list of colors through the optional `node_color` parameter.


```python
#Your code here
```

## Summary

Nice work! In this lab you created an initial network to visualize a bimodal network of businesses and yelp reviewers!
