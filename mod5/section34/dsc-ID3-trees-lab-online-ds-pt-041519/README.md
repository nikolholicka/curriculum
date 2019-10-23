
# ID3 Classification Trees: Perfect Split with Information Gain - Lab

## Introduction

In this lab, we will simulate the example from the previous lesson in python. We will write functions to calculate entropy and IG which will be used for calculating these uncertainty measures and deciding upon creating a split using information gain while growing a ID3 classification tree. We shall attempt to write general function that can be used for other (larger) problems as well. So let's get on with it.

## Objectives
You will be able to:
- Write functions for calculating Entropy and Information gain measures
- Identify the attribute for best split at master and each subsequent node


## Problem

We shall use the same problem about deciding weather to go and play tennis on a given day, given the weather conditions. Here is the data from previous lesson:

|  outlook | temp | humidity | windy | play |
|:--------:|:----:|:--------:|:-----:|:----:|
| overcast | cool |   high   |   Y   |  yes |
| overcast | mild |  normal  |   N   |  yes |
|   sunny  | cool |  normal  |   N   |  yes |
| overcast |  hot |   high   |   Y   |  no  |
|   sunny  |  hot |  normal  |   Y   |  yes |
|   rain   | mild |   high   |   N   |  no  |
|   rain   | cool |  normal  |   N   |  no  |
|   sunny  | mild |   high   |   N   |  yes |
|   sunny  | cool |  normal  |   Y   |  yes |
|   sunny  | mild |  normal  |   Y   |  yes |
| overcast | cool |   high   |   N   |  yes |
|   rain   | cool |   high   |   Y   |  no  |
|   sunny  |  hot |  normal  |   Y   |  no  |
|   sunny  | mild |   high   |   N   |  yes |

## Write a function `entropy(pi)` to calculate total entropy in a given discrete probability distribution `pi`

- The function should input a probability distribution `pi` as an array of class distributions. This should take the form of two integers to represent how many items are in each class.  For example: `[4,4]` indicates that there is four item in each class. `[10,0]` indicates that there are 10 items in one class and 0 in the other.
- Calculate and return entropy according to the formula: $$Entropy(p) = -\sum (P_i . log_2(P_i))$$


```python
from math import log
def entropy(pi):
    '''
    return the Entropy of a probability distribution:
    entropy(p) = - SUM (Pi * log(Pi) )
    '''

    pass


# Test the function 

print(entropy([1,1])) # Maximum Entropy e.g. a coin toss
print (entropy([0,6])) # No entropy, ignore the -ve with zero , its there due to log function
print (entropy([2,10])) # A random mix of classes

# 1.0
# 0.0
# 0.6500224216483541
```

    None
    None
    None


## Write a function `IG(D,a)` to calculate the information gain 

- The function should input `D` as a class distribution array for target class, and `a` the class distribution of the attribute to be tested
- Using the `entropy()` function above, calculate the information gain as:

$$gain(D,A) = Entropy(D) - \sum(\frac{|D_i|}{|D|}.Entropy(D_i))$$

where `Di` represents distribution of each class in `a`.



```python
def IG(D, a):
    '''
    return the information gain:
    gain(D, A) = entropy(D)− SUM( |Di| / |D| * entropy(Di) )
    '''

    pass


# Uncomment to run the test

# set of example of the dataset - distribution of classes
test_dist = [6, 6] # Yes, No
# attribute, number of members (feature)
test_attr = [ [4,0], [2,4], [0,2] ] # class1, class2, class3 of attr1 according to YES/NO classes in test_dist

print(IG(test_dist, test_attr))

# 0.5408520829727552
```

    None


## First Iteration - Decide Best Split for master node

- Create The class distribution `play` as a list showing frequencies of both classes from the dataset
- Similarly create variables for four categorical feature attributes showing the class distribution for each class with respect to the target classes (yes and no)
- Pass the play distribution with each attribute to calculate the information gain


```python


# Your code here




# Information Gain:

# Outlook: 0.41265581953400066
# Temperature: 0.09212146003297261
# Humidity: 0.0161116063701896
# Wind:, 0.0161116063701896
```

    Information Gain:
    
    Outlook: 0.2467498197744391
    Temperature: 0.029222565658954647
    Humidity: 0.15183550136234136
    Wind:, 0.04812703040826927


We see here that the outlook attribute gives us the highest value for information gain, hence we choose this for creating a split at root node. So far we have our root node looking as below:
<img src='images/tree_fs.png'  width ="500"  >


## Second Iteration

Since the first iteration determines what split we should make for the root node of our tree, it's pretty simple. Now, we move down to the second level, and start finding the optimal split for each of the nodes on this level. The first branch (edge) of three above that leads to the "Sunny" outcome. Check for temperature, humidity and wind attributes to see which one provides the highest information gain.

For the steps as above. Remember, we have 6 positive and 1 negative examples in the "sunny" branch.


```python


# Your code here 




# Information Gain:

# Temperature: 0.7974288158134881
# Humidity: 0.6824544962108586
# Wind:, 0.7084922088251644
```

    Information Gain:
    
    Temperature: 0.7974288158134881
    Humidity: 0.9402859586706309
    Wind:, 0.5117145300992023


So here we see that temperature gives us the the highest information gain, so we'll use it to split our tree as shown below:

<img src='images/humid_fs.png'  width ="650"  >


Let's continue. 

## Third Iteration

We'll now calculate splits for the 'temperature' node we just created for days where the weather is sunny. Temperature has three possible values: [Hot, Mild, Cool]. This means that for each of the possible temperatures, we'll need to calculate if splitting on windy or humidity gives us the greatest possible information gain.

Why are we doing this next instead of the rest of the splits on level 2? Because Decision Trees are a Greedy Algorithm, meaning that the next choice is always the one that will give it the greatest information gain. In this case, evaluating the temperature on sunny days gives us the most information gain, so that's where we'll go next.

## All the Other Iterations

What happens once we get down to a 'pure' split? Obviously, we stop splitting. Once that happens, we go back to the highest remaining uncalculated node, and calculate the best possible split for that one. We then continue on with that branch, until we have exhausted all possible splits or we run into a split that gives us 'pure' leaves where all 'play=Yes' is on one side of the split, and all 'play=No' is on the other.

## Summary 

Now, you've seen:

- How to calculate entropy
- How to calculate information gain
- How to figure out the optimal split
- How to figure out what the next split you should calculate should be ('greedy' approach)
- This lab should have helped you familiarize yourself with how Decision Trees work 'under the hood', and demystified - How the algorithm actually 'learns' from data. Great job
