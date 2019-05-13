
# Object Oriented Shopping Cart - Lab

## Introduction
In this lab we will be mimicing the functionality of a shopping cart with our knowledge of object oriented Python. Our shopping cart will be able to add items of different quantities and prices to our cart, calculate discounts, keep track of what items have been added, and void transactions.

## Objectives

You will be able to:

* Build a class with instance methods
* Call instance methods inside of other instance methods
* Use instance methods to track information pertinent to an instance of a class

## Instructions

In this lab, you'll practice your object orientated programming skills by modifying the shopping_cart.py file.

To start, we'll set this notebook to autoreload packages so that when you update shopping_cart.py, you can experiment with the effects here. Remember that while the package will be reloaded, you will need to reinitialize your class instance. Here's an example to get you started:


```python
%load_ext autoreload
%autoreload 2
```


```python
#Import our custom package
from shopping_cart import ShoppingCart
```


```python
#Initialize an instance of our shopping cart class
shopping_cart = ShoppingCart()
```

## Add an Initialization Behavior to the ShoppingCart Class

Update your shopping_cart.py file to include an __init__ method. This should define three default attributes: 'total', which should be set to 0, 'employee_discount', set to None and 'items', set to a blank list. The line of code below should work and produce the previewed output once you do this.


```python
shopping_cart = ShoppingCart() #Add a line to reinitialize an instance of the class
print(shopping_cart.total)
print(shopping_cart.employee_discount)
print(shopping_cart.items)
```

    0
    None
    []


## Add and `add_item()` method.

Define an instance method called `add_item` that will add an item to our cart. It should take in the name of an item, its price and an optional quantity. The method should increase the shopping cart's total by the appropriate amount and return the new total for the shopping cart.

> **hint:** think about how you would like to keep this information in your list of items. Can we imagine wanting to ever check the price of an individual item after we've added it to our cart? What data type do we know of that can associate the item name with it's price?


```python
shopping_cart.add_item("rainbow sandals", 45.99) # 45.99
```




    45.99




```python
shopping_cart.add_item("agyle socks", 10.50) # 56.49
```




    56.49




```python
shopping_cart.add_item("jeans", 50.00, 3) # 206.49
```




    206.49



## Add Summary Methods `mean_item_price()` and `median_item_price()` 

Define two more instance methods: `mean_item_price` and `median_item_price`, which should return the average price per item and the median price of the items in your cart, respectively. 

> **Remember:** the mean is the average price per item and to find the median we must do three things:
* First put all numbers in our list in ascending order (smallest to greatest)
* Then check to see if there is an odd number of elements in our list. If so, the middle number is the median
* Finally, if there is an even number of elements in the list, the median will be the average or mean of the two center elements (e.g. given the list `[1,2,3,4]` the elements `2` and `3` are the two center elements and the median would be (2 + 3)/2 or `2.5`).


```python
shopping_cart.mean_item_price() # 41.29
```




    41.298




```python
shopping_cart.median_item_price() # 50.00
```




    50.0



## Add an `apply_discount` method

Now, let's define an instance method called `apply_discount` that applies a discount if one is provided and returns the discounted total. For example, if we initialize a new shopping cart with a discount of 20% then our total should be discounted in the amount of 20%. So, if our total were `$100`, after the discount we only would owe `$80`.

If our shopping cart does not have an employee discount, then it should return a string saying: `"Sorry, there is no discount to apply to your cart :("`


```python
discount_shopping_cart = ShoppingCart(20)
print(discount_shopping_cart.add_item("rainbow sandals", 45.00)) # 45.00
print(discount_shopping_cart.add_item("agyle socks", 15.00)) # 60.00
print(discount_shopping_cart.apply_discount()) # 48.00
print(discount_shopping_cart.add_item("macbook air", 1000.00)) # 1060.00
print(discount_shopping_cart.apply_discount()) # 848.00
print(shopping_cart.apply_discount()) # Sorry, there is no discount to apply to your cart :(
```

    45.0
    60.0
    48.0
    1060.0
    848.0
    Sorry, there is no discount to apply to your cart :(


## Add a `void_last_item()` method

Finally, we are missing one piece of functionality. What if we just accidentally added something to our cart or decided that this item is too expensive for our budget? Let's define a method called `void_last_item` that removes the last item from our shopping cart and updates its total.  If there are no items in the shopping cart, `void_last_item` should return `"There are no items in your cart!"`.


```python
shopping_cart.void_last_item()
shopping_cart.total # 156.49
```




    156.49



## Summary
In this lab, we practiced using instance methods to mimic the functionality of a shopping cart as well as defined methods that give us the mean and median prices of all the items in our cart. 
