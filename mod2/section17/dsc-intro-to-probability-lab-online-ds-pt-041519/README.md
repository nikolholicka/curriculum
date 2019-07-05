
# Introduction to Probability - Lab

## Introduction

Now that you know what sets are, we can go on and work with two sets that are of key importance when talking about probability: the event space and the sample space. These two concepts are foundational for calculating probabilities when assuming each event in the event space *has a same probability of happening*. Typical examples are rolling a dice (if the dice is "fair", the chance of throwing each number between 1 and 6 is 1/6) and flipping a coin (1/2 heads vs tails). You'll get a better sense of how all of this works in this lab.

## Objectives

You will be able to:

- Learn about experiments, outcomes and event space
- Understand probability through relative frequency
- Learn about the probability axioms
- Learn about the addition law of probability
- Learn that where each outcome is equally likely, the probability is equal to number of outcomes in the event space divided by number of outcomes in the sample space


##  Sample space, event space and the law of relative frequency

#### a. Let's throw a dice once

First, create a set `roll_dice` that holds the sample space of rolling a 6-sided dice once.


```python
roll_dice = None
```

Now, let's assume that the event space is defined by "throwing a number higher than 4". This means that we consider the outcome "successful" if a 5 or a 6 is thrown. Create a set that holds these values.


```python
event = None
```

Now use the formula $P(E) = \dfrac{|E|}{|S|}$ (this formula is called "Laplace's formula" and strongly related to the law of relative frequency) to calculate the probability.


```python
prob_5_6 = None
prob_5_6  # 0.3333333333333333
```

Using this formula, it should be clear that the answer is 1/3 or 0.3333....  

#### b. Now, let's simulate rolling dice to see how the law of relative frequency works.

As mentioned in the lecture, the law of relative frequency can be used to prove certain probabilities. But how does this work exactly? You're about to find out!

$$P(E) = \lim_{n\rightarrow\infty} \dfrac{S{(n)}}{n}$$

As you can see in the formula, the law states that when repeating an experiment $n$ times, where $n$ is very big, and you divide the number of "good" outcomes by the sample space (here we call it event E), you get to the probability of the event E. It should be clear that we get a more accurate number for P(E) when $n$ grows.

Let's see how this works. First, let's randomly generate values between 1 and 6. You can use `numpy` (imported as `np`) to generate random integers between 1 and 6 by setting the correct arguments. The `np.random` module is a very useful tool for this. We helped you with the code here, but you'll get more practice and a thorough explanation later on!


```python
import numpy as np
np.random.randint(1,7) # you will get a random value between 1 and 6. See how it changes when you rerun
```

Now, let's repeat this experiment 10 times, then 1000 times, then 1 million times, then 100 million times. 
You can do this by specifying the argument `size` within the numpy function used above. Store the values in the pre-defined variables below.


```python
np.random.seed(12345) # to make sure there is no randomness

dice_10 = np.random.randint(1,7,size= None)
dice_1k = np.random.randint(1,7,size= None)
dice_1m = np.random.randint(1,7,size=None)
dice_100m = np.random.randint(1,7,size=None)
```

next, let's count the number of "events". Remember that an event here is defined as throwing a 5 or a 6. Store them in the values below.


```python
event_10 = None
event_1k = None
event_1m = None
event_100m = None
```

Next, you'll divide the number of events for each $n$ by the respective values for $n$. What do you see?


```python
prob_10 = None
prob_1k = None
prob_1m = None
prob_100m = None
prob_10, prob_1k, prob_1m, prob_100m  # 0.5 0.331 0.333657 0.33329752
```

You see that the probability converges to 0.3333333... for higher values of $n$. 

##  The Probability Axioms

You're working at the United Nations, and want to get a better sense of the world population. 

You come across some numbers and find the list of probabilities of being an inhabitant for each of the seven continents (rounded up to 3 digits):

- P(Africa) = 0.161
- P(Antarctica) = 0.000
- P(Asia) = 0.598
- P(Europe) = 0.10
- P(North-America) = 0.079
- P(Australia) = 0.005
- P(South-America) = 0.057

store these values using the variable names below:


```python
P_afr = None
P_ant = None
P_as = None
P_eur = None
P_na = None
P_aus = None
P_sa = None
```

Now create the sample space set names `continents`. Store the sample space in a numpy array.


```python
continents = np.array([P_afr, P_ant, P_as, P_eur, P_na, P_aus, P_sa])
print(continents)
```

We want to make sure that the three probability axioms are fulfilled, because they assure us that $(\Omega,E,P)$ is a **probability space**:

- if we have a sample space $S$ (or $\Omega$)
- if we have an event space $E$ and a probability measure $P$, 
- **and** the three probability axioms are fulfilled, 

The third axiom is fairly ad hoc, and you will basically have to deduce from the context whether individual events are independent. It is fairly straightforward, however, that people can not be inhabitants of two continents at the same time, so for now, we will assume that we're good for axiom three.

However, we can use the numpy array `continents` to verify if axiom 1 and 2 are fulfilled. Create a function "axioms" that returns the message "We're good!" if both axiom 1 and 2 are fulfilled, and "Not quite!" if that's not the case.


```python
def check_axioms(sample_space):
    None
```

Now test your newly created function out on `continents`


```python
check_axioms(continents)
```

You want to make sure your test returns `"Not quite!"` for the following numpy arrays. Go ahead and test away!


```python
test_1 = np.array([0.05, 0.2, 0.3, 1.01])
test_2 = np.array([0.05, 0.5, 0.6, -0.15])
test_3 = np.array([0.043,0.05,.02,0.3,0.2])
```


```python
None
None
None
```

Great! We tested it and seems like our set `continents` is a true probability space.

## Some more practice on the sample and event spaces

In this exercise, we'll look at possible outcomes when throwing a dice twice. For your convenience, we created the NumPy array below.

Next, we'll compute a couple or probabilities associated with doing this.


```python
import numpy as np
sample_dice = np.array([(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
              (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
              (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
              (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
              (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
              (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)])
```

## Some more practice on the sample and event spaces

In this exercise, we'll look at possible outcomes when throwing a dice twice. For your convenience, we created the NumPy array below.

Next, we'll compute a couple or probabilities associated with doing this.


```python
import numpy as np
sample_dice = np.array([(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
              (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
              (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
              (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
              (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
              (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)])
```

Look at the shape of the array to reassure we haven't made any mistakes.


```python
None # should be equal to (36,2)
```

Use Python to obtain the following probabilities:

#### a. What is the probability of throwing a 5 at least once?

First, use sample_dice to get "True" values for each time a 5 occurs. This means that, 


```python
set_5 = None # Your output should be shape (36, 2) but with booleans instead of numbers
```

Next, make sure that you get a value `True` for each pair where at least one 5 was thrown.


```python
true_5 = None # Your output should be shape (36,) and have booleans. 
              # "True" if at least one of the previous pairs was true. Tip: use np.any()
print(true_5)
```

Applying the `sum()` function you can get to the total number of items in the event space. Divide this by the total number in the sample space.


```python
prob_5 = None
print(prob_5)
```

#### b. What is the probability of throwing a 5 or 6 at least once?


```python
set_5 = None
set_6 = None
```


```python
set_5_6 = None
```


```python
set_any_5_6 = None
print(set_any_5_6) 
```


```python
prob_5_6 = None
print(prob_5_6)
```

#### c. What is the probability of the outcome having a sum of exactly 8?


```python
sum_dice = None
sum_8 = None
```


```python
prob_sum_8 = None
print(prob_sum_8)
```

## Now let's try creating your own event space!

A teaching assistant is holding office hours so students can make appointments. She has 6 appointments scheduled today, 3 by male students, and 3 by female students. 

Create a NumPy array of possible outcomes in the same way as we did in the "throwing a dice twice" exercise. It will be quite a bit of typing, as your resulting NumPy array will have a shape (20,6)!


```python
sample_mf= None
```


```python
None # get the shape of sample_mf
```


```python
sample_length= None
print(sample_length)
```

#### 1. Calculate the probability that at least 2 out of the first 3 appointments are with female students

First, select the first 3 appointment slots and check for "F".


```python
first_3_F = None
None
```


```python
num_F = None
print(num_F)
```


```python
F_2plus = None
print(F_2plus)
```


```python
prob_F_2plus = None
print(prob_F_2plus)
```

#### 2. Calculate the probability that after 4 appointment slots, all the female students have had an appointment


```python
None
```

You noticed that coming up with the sample space was probably the most time-consuming part of the exercise, and it would really become unfeasible to write this down for say, 10 or, even worse, 20 appointments in a row. You'll learn about methods that make this easy in the next lecture!

## The Addition Law of Probability
At a supermarket, we randomly select customers, and make notes of whether a certain customer owns a Visa card (event A) or an Amex credit card (event B). Some customers own both cards.
You can assume that:

- P(A) = 0.5
- P(B) = 0.4
- both A and B = 0.25.

1) compute the probability that a selected customer has at least one credit card.

2) compute the probability that a selected customer doesn't own any of the mentioned credit cards.

3) compute the probability that a customer *only* owns VISA card.

(You can use python here, but you don't have to)

## Summary

In this lab, you got to practice your knowledge on the foundations of probability through working on problems regarding the law of relative frequency, the probability axioms and the addition law of probability.
