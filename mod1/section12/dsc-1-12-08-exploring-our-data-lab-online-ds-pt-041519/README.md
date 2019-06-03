
# Exploring Our Data - Lab

## Introduction 

In this lab we shall perform a exploratory data analysis task, using statistical and visual EDA skills we have seen so far. We shall continue using the Walmart sales database that we have acquired and cleaned in the previous labs. 

## Objectives
You will be able to:

* Check the distribution of various columns
* Examine the descriptive statistics of our data set
* Create visualizations to help us better understand our data set

## Data Exploration

In the previous lab, we performed some data cleansing and scrubbing activities to create data subset, deal with null values and categorical variables etc. In this lab, we shall perform basic data exploration to help us better understand the distributions of our variables. We shall consider regression assumptions seen earlier to help us during the modeling process. 

*The dataset for this lab has been taken from our data scrubbing lab, just before we encoded our categorical variables as one hot. This is to keep the number of columns same as original dataset to allow more convenience during exploration.* 

### Load the dataset 'walmart_dataset.csv' as pandas dataframe and check its contents 


```python
# You code here 
```




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
      <th>Store</th>
      <th>Dept</th>
      <th>Weekly_Sales</th>
      <th>IsHoliday</th>
      <th>Type</th>
      <th>Size</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
      <th>binned_markdown_1</th>
      <th>binned_markdown_2</th>
      <th>binned_markdown_3</th>
      <th>binned_markdown_4</th>
      <th>binned_markdown_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>24924.50</td>
      <td>False</td>
      <td>A</td>
      <td>0.283436</td>
      <td>-1.301205</td>
      <td>-1.56024</td>
      <td>0.40349</td>
      <td>0.913194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>50605.27</td>
      <td>False</td>
      <td>A</td>
      <td>0.283436</td>
      <td>-1.301205</td>
      <td>-1.56024</td>
      <td>0.40349</td>
      <td>0.913194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>13740.12</td>
      <td>False</td>
      <td>A</td>
      <td>0.283436</td>
      <td>-1.301205</td>
      <td>-1.56024</td>
      <td>0.40349</td>
      <td>0.913194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>39954.04</td>
      <td>False</td>
      <td>A</td>
      <td>0.283436</td>
      <td>-1.301205</td>
      <td>-1.56024</td>
      <td>0.40349</td>
      <td>0.913194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>32229.38</td>
      <td>False</td>
      <td>A</td>
      <td>0.283436</td>
      <td>-1.301205</td>
      <td>-1.56024</td>
      <td>0.40349</td>
      <td>0.913194</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Describe the dataset using 5 point statistics and record your observations


```python
# your code here 
```




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
      <th>Store</th>
      <th>Dept</th>
      <th>Weekly_Sales</th>
      <th>Size</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>CPI</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>97839.000000</td>
      <td>97839.000000</td>
      <td>97839.000000</td>
      <td>9.783900e+04</td>
      <td>9.783900e+04</td>
      <td>9.783900e+04</td>
      <td>9.783900e+04</td>
      <td>9.783900e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.474545</td>
      <td>43.318861</td>
      <td>17223.235591</td>
      <td>-8.044340e-14</td>
      <td>2.339480e-13</td>
      <td>4.784098e-13</td>
      <td>-9.181116e-15</td>
      <td>1.795967e-12</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.892364</td>
      <td>29.673645</td>
      <td>25288.572553</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-1098.000000</td>
      <td>-1.611999e+00</td>
      <td>-3.843452e+00</td>
      <td>-1.691961e+00</td>
      <td>-1.958762e+00</td>
      <td>-2.776898e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>19.000000</td>
      <td>2336.485000</td>
      <td>-1.028620e+00</td>
      <td>-7.087592e-01</td>
      <td>-1.053793e+00</td>
      <td>-1.266966e-01</td>
      <td>-6.503157e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.000000</td>
      <td>36.000000</td>
      <td>7658.280000</td>
      <td>2.834360e-01</td>
      <td>1.340726e-01</td>
      <td>1.180741e-01</td>
      <td>4.995210e-01</td>
      <td>-4.621274e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.000000</td>
      <td>71.000000</td>
      <td>20851.275000</td>
      <td>1.113495e+00</td>
      <td>8.680410e-01</td>
      <td>8.243739e-01</td>
      <td>6.346144e-01</td>
      <td>7.089160e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>99.000000</td>
      <td>693099.360000</td>
      <td>1.171380e+00</td>
      <td>1.738375e+00</td>
      <td>2.745691e+00</td>
      <td>8.517705e-01</td>
      <td>2.361469e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Your observations here 
```

### Use pandas histogram plotting to plot histograms for all the variables in the dataset


```python
# Your code here 
```


![png](index_files/index_10_0.png)



```python
# Your observations here 
```

### Build normalized histograms with kde plots to explore the distributions further. 
### Use only the continuous variables in the dataset to plot these visualizations. 


```python
# Your code here 
```



![png](index_files/index_13_2.png)



![png](index_files/index_13_3.png)



![png](index_files/index_13_4.png)



![png](index_files/index_13_5.png)



![png](index_files/index_13_6.png)



![png](index_files/index_13_7.png)



```python
# State your observations here 
```

### Build joint plots to check for the linearity assumption between predictors and target variable

Let's use a slightly more advanced plotting technique in seaborn that uses scatter plots, distributions, kde and simple regression line - all in a single go. Its called a `jointplot`. [Here is the official doc. for this method](https://seaborn.pydata.org/generated/seaborn.jointplot.html). 

Here is how you would use it:

> **`sns.jointplot(x= <column>, y= <column>, data=<dataset>, kind='reg')`**

A joint plot will allow us to visually inspect linearity as well as normality assumptions as a single step. 


```python
# Your code here 
```


![png](index_files/index_16_0.png)



![png](index_files/index_16_1.png)



![png](index_files/index_16_2.png)



![png](index_files/index_16_3.png)



![png](index_files/index_16_4.png)



![png](index_files/index_16_5.png)



![png](index_files/index_16_6.png)



```python
# Provide your observations here 
```

### So Now what ?

Okie so our key assumptions at this stage don't hold so strong. But that does not mean that should give up and call it a poor dataset. There are lot of pre-processing techniques we can still apply to further clean the data and make it more suitable for modeling. 

![](https://i.stack.imgur.com/yZQgZ.gif)

For building our initial model, we shall use this dataset for a multiple regression experiment and after inspecting the combined effect of all the predictors on the target, we may want to further pre-process the data and take it in for another analytical ride. 

The key takeaway here is that we will hardly come across with a real world dataset that meets all our expectations. Another reason to move ahead with this dataset is to ehelp us realize the importance of pre-processing for an improved model building. and we must always remember: 

> Model development is an iterative process. It hardly ever gets done in the first attempt. 

So looking at above, we shall look at some guidelines on model building and validation in upcoming lessons, before we move on to our regression experiment. 

## Further reading 

Have a look at following resources on how to deal with complex datasets that don't meet our initial expectations. 

[What to Do When Bad Data Thwarts Machine Learning Success](https://towardsdatascience.com/what-to-do-when-bad-data-thwarts-machine-learning-success-fb82249aae8b)

[Practical advice for analysis of large, complex data sets ](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)

[Data Cleaning Challenge: Scale and Normalize Data](https://www.kaggle.com/rtatman/data-cleaning-challenge-scale-and-normalize-data)

## Summary 

In this lesson we performed some basic EDA on the walmart dataset to check for regression assumptions. Initially our assumptions dont hold very strong but we decided to move ahead with building our first model using this dataset and plan further pre-processing in following iterations. 
