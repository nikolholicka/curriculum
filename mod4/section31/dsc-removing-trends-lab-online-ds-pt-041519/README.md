
# Removing Trends - Lab

## Introduction

In this lab, you'll practice your detrending skills!

## Objectives

You will be able to:
* Learn how to remove trends and seasonality
* Use a log transformation to minimize non-stationarity
* Use rolling means to reduce non-stationarity
* Use differencing to reduce non-stationarity

## Detrending the Airpassenger data


```python
# Import necessary libraries

# Import passengers.csv and set it as a time-series object. Plot the TS

```

## Create a stationarity check

At this stage, we can use the code from previous labs to create a function `stationarity_check(ts)` that takes in a time series object and performs stationarity checks including rolling statistics and the Dickey Fuller test. 

We want the output of the function to:
- Plot the original time series along with the rolling mean and rolling standard deviation in one plot
- Output the results of the Dickey-Fuller test


```python
# Create a function to check for the stationarity of a given timeseries using rolling stats and DF test
# Collect and package the code from previous lab

```

Use your newly created function on the airpassenger data set.


```python
# Code here
```

## Perform a log() and sqrt() transform


```python
# Log transform timeseries and compare with original to check the effect

```

moving forward, let's keep working with the log transformed data before subtracting rolling mean, differencing, etc.

## Subtracting the rolling mean

Create a rolling mean using your log transformed time series, with a time window of 7. Plot the log-transformed time series and the rolling mean together.


```python
# your code here
```

Now, subtract the rolling mean from the time series, look at the 10 first elements of the result and plot the result.


```python
# Subtract the moving average from the original data and check head for Nans
```


```python
# Drop the NaN values from timeseries calculated above

```


```python
# Plot the result

```

Finally, use your function `check_stationarity` to see if this series is considered stationary!


```python
# Your code here
```

### Based on the visuals and on the Dickey-Fuller test, what do you conclude?



```python
# Your conclusion here
```

## Subtracting the weighted rolling mean

Repeat all the above for the *weighter* rolling mean. Start from the log-transformed data again. Compare the Dickey-Fuller Test results. What do you conclude?


```python
# Use Pandas ewma() to calculate Weighted Moving Average of ts_log

# Plot the original data with exp weighted average

```


```python
# Subtract the moving average from the original data and plot
```


```python
# do a stationarity check
```

### Based on the visuals and on the Dickey-Fuller test, what do you conclude?



```python
# Your conclusion here
```

## Differencing

Using exponentially weighted moving averages, we seem to have removed the upward trend, but not the seasonality issue. Now use differencing to remove seasonality. Make sure you use the right amount of `periods`. Start from the log-transformed, exponentially weighted rolling mean-subtracted series.

After you differenced the series, run the `stationarity check` again.


```python
# difference your data and look at the head
```


```python
# plot your differenced time series
```


```python
# drop nas
```


```python
# perform the stationarity check
```

### Your conclusion


```python
# Your conclusion here
```

## Summary 

In this lab, you learned how to make time series stationary through using log transforms, rolling means and differencing.
