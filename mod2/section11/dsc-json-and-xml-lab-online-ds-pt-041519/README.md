
# JSON and XML - Lab

## Introduction

In this lab, you'll practice navigating JSON and XML data structures.

## Objectives
You will be able to:
* Effectively use the JSON module to load and parse JSON documents
* Read and access data stored in JSON and XML
* Compare  and contrast the JSON and XML as data interchange types


## XML


```python
import xml.etree.ElementTree as ET
```

### Create an XML tree and retrieve the root tag.


```python
#Your code here
```

### How many direct descendents does the root tag have?


```python
#Answer: 1
```

### How many different types of tags are there within the entire XML file?


```python
# Your code here
```

### Create a DataFrame listing the number of each type of tag. 
Sort the DataFrame in descending order by the tag count. The first entry should demonstrate there are 286 row tags in the XML file.   
(Your DataFrame will be a single column, so could also be thought of as a Series.)


```python
import pandas as pd
```


```python
#Your code here
```

## JSON

### Open the same dataset from json


```python
#Your code here
```

### What is the root data type of the json file?


```python
### Your code here
```

### Navigate to the 'data' key of your loaded json object. What data type is this?


```python
#Your code here
```

### Preview the first entry from the value returned by the 'data' key above.


```python
#Your code here
```

### Preview the Entry under meta -> view -> columns (the keys of three successively nested dictionaries)

### Create a DataFrame from your json data
The previous two questions previewed one entry from the data object within the json file, as well as the column details associated with that data from the meta entry within the json file. Both should have 19 entries. Create a DataFrame of the data. Be sure to use the information from the meta entry to add appropriate column names to your DataFrame.


```python
#Your code here
```

### What's wrong with the first row of the DataFrame?


```python
#Your code here
```

#Your answer here

## Summary

Congratulations! You've started exploring some more complicated data structures used for the web and got to practice data munging and exploring!
