
# Exploring and Transforming JSON Schemas

# Introduction

In this lesson, you'll formalize how to explore a JSON file whose structure and schema is unknown to you. This often happens in practice when you are handed a file or stumble upon one with little documentation.

## Objectives
You will be able to:
* Explore unknown JSON schemas
* Access and manipulate data inside a JSON file
* Convert JSON to alternative data formats

## Loading the JSON file

Load the data from the file disease_data.json.


```python
#Your code here 
```

## Explore the first and second levels of the schema hierarchy


```python
#Your code here
```

## Convert to a DataFrame

Create a DataFrame from the JSON file. Be sure to retrive the column names for the dataframe. (Search within the 'meta' key of the master dictionary.) The DataFrame should include all 42 columns.


```python
#Your code here
```

## Level-Up
## Create a bar graph of states with the highest asthma rates for adults age 18+

## Level-Up!
## Create a function (or class) that returns an outline of the schema structure like this: 
<img src="images/outline.jpg" width="350">

Rules:
* Your outline should follow the numbering outline above (I, A, 1, a, i).
* Your outline should be properly indented! (Four spaces or one tab per indentation level.)
* Your function goes to at least a depth of 5 (Level-up: create a parameter so that the user can specify this)
* If an entry is a dictionary, list its keys as the subheadings
* After listing a key name (where applicable) include a space, a dash and the data type of the entry
* If an entry is a dict or list put in parentheses how many items are in the entry
* lists will not have key names for their entries (they're just indexed)
* For subheadings of a list, state their datatypes. 
* If a dictionary or list is more then 5 items long, only show the first 5 (we want to limit our previews); make an arbitrary order choice for dictionaries. (Level-up: Parallel to above; allow user to specify number of items to preview for large subheading collections.)


```python
# Your code here; you will probably want to define subfunctions.
def print_obj_outline(json_obj):
    return outline
```


```python
outline = print_obj_outline(data)
```


```python
print(outline) #Your function should produce the following output for this json object (and work for all json files!)
```

    I. root - <class 'dict'> (2 items)
        A. meta <class 'dict'> (1 items)
            1. view <class 'dict'> (40 items)
                a. id <class 'str'> 
                b. name <class 'str'> 
                c. attribution <class 'str'> 
                d. attributionLink <class 'str'> 
                e. averageRating <class 'int'> 
        B. data <class 'list'> (60266 items)
            1. <class 'list'> (42 items)
                a. <class 'int'> 
                b. <class 'str'> 
                c. <class 'int'> 
                d. <class 'int'> 
                e. <class 'str'> 
            2. <class 'list'> (42 items)
                a. <class 'int'> 
                b. <class 'str'> 
                c. <class 'int'> 
                d. <class 'int'> 
                e. <class 'str'> 
            3. <class 'list'> (42 items)
                a. <class 'int'> 
                b. <class 'str'> 
                c. <class 'int'> 
                d. <class 'int'> 
                e. <class 'str'> 
            4. <class 'list'> (42 items)
                a. <class 'int'> 
                b. <class 'str'> 
                c. <class 'int'> 
                d. <class 'int'> 
                e. <class 'str'> 
            5. <class 'list'> (42 items)
                a. <class 'int'> 
                b. <class 'str'> 
                c. <class 'int'> 
                d. <class 'int'> 
                e. <class 'str'> 


## Summary

Well done! In this lab you got some extended practice exploring the structure of JSON files and writing a recursive generalized function for outlining a JSON file's schema! 
