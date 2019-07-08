
# Conditional Probability - Lab

## Introduction

In order to be ready for real world applications of probability, it is important to understand what happens when probabilities are not independent. Very often, the probability of a certain event depends on other events happening! Let's see how this all works in this lab.

## Objectives

You will be able to:

- Understand and explain the conditional probability - $P(A \cap B) = P(A \mid B) P(B)$
- Use the multiplication rule to find the probability of the intersection of two events
- Apply the techniques learned in the lesson to simple problems

## Exercise 1
A coin is tossed and a single 6-sided dice is rolled. Find the probability of landing on the head side of the coin and rolling a 3 on the dice.


```python
# Your solution
```

## Exercise 2


After conducting a survey, one of the outcomes was that 8 out of 10 of the surey subjects liked chocolate chip cookies. If three survey subjects are chosen at random **with replacement**, what is the probability that all three like chocolate chip cookies?


```python
# Your Solution
```

## Exercise 3
70% of your friends like chocolate flavored ice cream , and 35% like chocolate AND like strawberry flavors.

What percent of those who like chocolate also like strawberry?


```python
# Your solution 
```

50% of your friends who like chocolate also like strawberry

## Exercise 4
What is the probability of drawing 2 consecutive aces from a deck of cards. 


```python
# Your solution
```

## Exercise 5
In a manufacturing factory that produces a certain product, there are 100 units of the product, 5 of which are defective. We pick three units from the 100 units at random. 

What is the probability that none of them are defective?
Hint: Use the chain rule here!


```python
# Your solution
```

## Exercise 6

Let's consider the example where 2 dice are thrown. Given that **at least one** of the dice has come up on a number higher than 4, what is the probability that the sum is 8?

Let $i,j$ be the numbers shown on the dice. The events $A$ and $B$ are described below:

* **Event $A$ is when either $i$ or $j$ is 5 or 6** (keep an eye on either - or)
* **Event $B$ is when $i + j = 8$**


* What is the size of sample space $\Omega$ ?
* What is $P(A \cap B)$ ?
* What is $P(A)$ ?
* Use above to calculate $P(B \mid A)$


```python
# Your solution
```

## Exercise 7

Let's consider a credit card example. At a supermarket, customers are selected randomly, the store owner recorded whether costumers owned a Visa card (event A) or an Amex credit card (event B). Some customers own both cards.
You can assume that:

- $P(A)$ = 0.5
- $P(B)$ = 0.4
- both $A$ and $B$ = 0.25.


With the knowledge we have about conditional probabilities, compute and interpret the following probabilities:

- $P(B \mid A)$
- $P(B' \mid A)$
- $P(A \mid B)$
- $P(A' \mid B)$



```python
# Your solution
```

## Summary 

In this lab you practiced conditional probability and its theorem with some simple problems. The key takeaway from this lab is to be able to identify random events as dependent or independent and calculating the probability of their occurrence using appropriate methods. Next you'll learn about some more conditional probability axioms, building on the knowledge we have so far. 
