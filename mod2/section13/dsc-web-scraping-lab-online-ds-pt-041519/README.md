
# Web Scraping - Lab

## Introduction

Now that you've seen a more extensive example of developing a web scraping script, it's time to further practice and formalize that knowledge by writing functions to parse specific pieces of information from the web page and then synthesizing these into a larger loop that will iterate over successive web pages in order to build a complete dataset.

## Objectives

You will be able to:

* Write functions to parse specific information from a web page
* Iterate over successive web pages in order to create a dataset

## Lab Overview

This lab will build upon the previous lesson. In the end, you'll look to write a script that will iterate over all of the pages for the demo site and extract the title, price, star rating and availability of each book listed. Building up to that, you'll formalize the concepts from the lesson by writing functions that will extract a list of each of these features for each web page. You'll then combine these functions into the full script which will look something like this:  

```python
df = pd.DataFrame()
for i in range(2,51):
    url = "http://books.toscrape.com/catalogue/page-{}.html".format(i)
    soup = BeautifulSoup(html_page.content, 'html.parser')
    new_titles = retrieve_titles(soup)
    new_star_ratings = retrieve_ratings(soup)
    new_prices = retrieve_prices(soup)
    new_avails = retrieve_avails(soup)
    ...
 ```

## Retrieving Titles

To start, write a function that extracts the titles of the books on a given page. The input for the function should be the `soup` for the HTML of the page.


```python
def retrieve_titles(soup):
    #Your code here
```

## Retrieve Ratings

Next, write a similar function to retrieve the star ratings on a given page. Again, the function should take in the `soup` from the given html page and return a list of the star-ratings for the books. These star ratings should be formatted as integers.


```python
def retrieve_ratings(soup):
    #Your code here
```

## Retrieve Prices

Now write a function to retrieve the prices on a given page. The function should take in the `soup` from the given page and return a list of prices formatted as floats.


```python
def retrieve_prices(soup):
    #Your code here
```

## Retrieve Availability

Write a function to retrieve whether each book is available or not. The function should take in the `soup` from a given html page and return a list of the availability for each book.


```python
def retrieve_availabilities(soup):
    #Your code here
```

## Create a Script to Retrieve All the Books From All 50 Pages

Finally, write a script to retrieve all of the information from all 50 pages of the books.toscrape.com website. 


```python
#Your code here
```

## Level-Up: Write a new version of the script you just wrote. 

If you used url hacking to generate each successive page url, instead write a function that retrieves the link from the `"next"` button at the bottom of the page. Conversely, if you already used this approach above, use URL-hacking (arguably the easier of the two methods in this case).


```python
#Your code here
```

## Summary

Well done! You just completed your first full web scraping project! You're ready to start harnessing the power of the web!
