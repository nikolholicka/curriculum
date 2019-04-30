
# Understanding Pandas Series and DataFrames - Lab

# Introduction

In this lab, let's get some hands on practice working with data cleanup using Pandas.

## Objectives
You will be able to:

* Manipulate columns in DataFrames (df.rename, df.drop)
* Manipulate the index in DataFrames (df.reindex, df.drop, df.rename)
* Manipulate column datatypes


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
df = pd.read_csv('turnstile_180901.txt')
print(len(df))
df.head()
```

    197625





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C/A</th>
      <th>UNIT</th>
      <th>SCP</th>
      <th>STATION</th>
      <th>LINENAME</th>
      <th>DIVISION</th>
      <th>DATE</th>
      <th>TIME</th>
      <th>DESC</th>
      <th>ENTRIES</th>
      <th>EXITS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A002</td>
      <td>R051</td>
      <td>02-00-00</td>
      <td>59 ST</td>
      <td>NQR456W</td>
      <td>BMT</td>
      <td>08/25/2018</td>
      <td>00:00:00</td>
      <td>REGULAR</td>
      <td>6736067</td>
      <td>2283184</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A002</td>
      <td>R051</td>
      <td>02-00-00</td>
      <td>59 ST</td>
      <td>NQR456W</td>
      <td>BMT</td>
      <td>08/25/2018</td>
      <td>04:00:00</td>
      <td>REGULAR</td>
      <td>6736087</td>
      <td>2283188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A002</td>
      <td>R051</td>
      <td>02-00-00</td>
      <td>59 ST</td>
      <td>NQR456W</td>
      <td>BMT</td>
      <td>08/25/2018</td>
      <td>08:00:00</td>
      <td>REGULAR</td>
      <td>6736105</td>
      <td>2283229</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A002</td>
      <td>R051</td>
      <td>02-00-00</td>
      <td>59 ST</td>
      <td>NQR456W</td>
      <td>BMT</td>
      <td>08/25/2018</td>
      <td>12:00:00</td>
      <td>REGULAR</td>
      <td>6736180</td>
      <td>2283314</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A002</td>
      <td>R051</td>
      <td>02-00-00</td>
      <td>59 ST</td>
      <td>NQR456W</td>
      <td>BMT</td>
      <td>08/25/2018</td>
      <td>16:00:00</td>
      <td>REGULAR</td>
      <td>6736349</td>
      <td>2283384</td>
    </tr>
  </tbody>
</table>
</div>



# Practice

## Objectives
You will be able to:
* Understand and explain what Pandas Series and DataFrames are and how they differ from dictionaries and lists
* Create Series & DataFrames from dictionaries and lists
* Manipulate columns in DataFrames (df.rename, df.drop)
* Manipulate the index in DataFrames (df.reindex, df.drop, df.rename)
* Manipulate column datatypes

# Rename all the columns to lower case


```python
#Your code here
```

# Change the Index to be the Line Names


```python
#Your code here
```

# Painstakingly change the index back


```python
# Your code here
```

## Create another column 'Num_Lines' that is a count of how many lines pass through a station. Then sort your dataframe by this column in descending order.


```python
# Your code here
```

## Write a function to clean a column name.


```python
def clean(col_name):
    cleaned = #Your code here; whatever you want to do to col_name. Hint: think back to str methods.
    return cleaned
```


```python
#This is a list comprehension. It applies your clean function to every item in the list.
#We then reassign that to df.columns
#You shouldn't have to change anything here.
#Your function above should work appropriately here.
df.columns = [clean(col) for col in df.columns] 
```


```python
#Checking the output, we can see the results.
df.columns
```

## Compare subway traffic by day of the week. Display this as a graph.


```python
#Your code here
```

## Is there more subway traffic on a weekend or a weekday?    Be specific in comparing magnitudes.


```python
#Your code here
```

# Drop a couple of columns


```python
# Your code here
```
