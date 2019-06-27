
# MongoDB - Lab

## Introduction

In this lesson, we'll get some hands-on experience with MongoDB!

## Objectives
You will be able to:
-  Use MongoDB, a popular NoSQL database
-  Create, Read, Update, and Delete Information in MongoDB

## Getting Started

To begin this lab, make sure that you start up the mongoDB server in your terminal first!


## Connecting to the MongoDB Database

In the cell below, import the appropriate library and connect to the mongoDB server. Create a new database called `'lab_db'`.

## Creating a Collection

Now, create a collection called `'lab_collection'` inside `'lab_db'`.


```python
mycollection = None
```

## Adding Some Data

Now, we're going to add some example records into our database. In the cells below, create dictionary representations of the following customer records:


|     Name     |            Email           |  Mailing_Address  | Balance |                         Notes                         |
|:------------:|:--------------------------:|:-----------------:|:-------:|:-----------------------------------------------------:|
|  John Smith  |    j.smith@thesmiths.com   | 123 mulberry lane |   0.0   |    Called technical support, issue not yet resolved   |
|  Jane Smith  |  jane_smith@thesmiths.com  |         Null          |  25.00  |                   Null                                    |
|  Adam Enbar  | adam@theflatironschool.com |    11 Broadway    |  14.99  |           Set up on recurring billing cycle           |
| Avi Flombaum |  avi@theflatironschool.com |    11 Broadway    |   0.0   |                   Null                                    |
|   Steven S.  |     steven.s@gmail.net     |         Null          |  -20.23 | Refunded for overpayment due to price match guarantee |


Your first challenge is to take all of the data in the table above and create the corresponding documents and add then to our mongo database. Note that fields that contain 'Null' should not be included in the document (recall that since mongo is schema-less, each document can be different). 

Create the documents from the table listed above, and then use `insert_many()` to insert them into the collection.


```python
customer_1 = None
customer_2 = None
customer_3 = None
customer_4 = None
customer_5 = None

all_records = None

insertion_results = None
```

Now, access the appropriate attribute from the results object returned from the insertion to see the unique IDs for each record inserted, so that we can confirm each were inserted correctly. 

## Querying and Filtering

In the cell below, return the name and email address for every customer record. Then, print every item from the query to show that it worked correctly. 


```python
query_1 = None

```

Great! Now, let's write a query that gets an individual record based on a stored key-value pair a document contains. 

In the cell below, write a query to get the record for John Smith by using his name. Then, print the results of the query to demonstrate that it worked correctly.  


```python
query_2 = None

```

Great! Now, write a query to get the names, email addresses, and balances for customers that have a balance greater than 0. Use a modifier to do this. 

**_HINT_**: In the query below, you'll be passing in two separate dictionaries. The first one defines the logic of the query, while the second tells which fields we want returned. 


```python
query_3 = None

```

## Updating a Record

Now, let's update some records. In the cell below. set the mailing address for Steven S. to `'367 55th St., apt 2A'`.


```python
record_to_update_1 = None
update_1 = None

```

Now, write a query to check that the update worked for this document in the cell below. 


```python
query_4 =None

```

Now, let's assume that we want to add birthdays for each customer record. Consider the following table:

|     Name     |  Birthday  |
|:------------:|:----------:|
|  John Smith  | 02/20/1986 |
|  Jane Smith  | 07/07/1983 |
|  Adam Enbar  | 12/02/1982 |
| Avi Flombaum | 04/17/1983 |
|   Steven S.  | 08/30/1991 |

We want to add birthdays for each person, but we want to do so in a way where we don't have to do the same repetitive task over and over again. This seems like a good opportunity to write a function to handle some of the logic for us!

In the cell below:

* Store the names in the `names_list` variable as a list.
* Store the birthdays in the `birthdays_list` variable as a list.
* Write a function that takes in the two different lists, and updates each record by adding in the appropriate key-value pair containing that user's birthday.

**_Hint:_** There are several ways that you could write this, depending on whether you want to use the `update_one()` method or the `update_many()` method. See if you can figure out both approaches!


```python
names_list = None
birthdays_list = None

def update_birthdays(names, birthdays):
    pass
        
update_birthdays(names_list, birthdays_list)
```

Now, write a query to check your work and see that the birthdays were added correctly.

Great! It looks like the birthdays have been successfully added to every record correctly!

## Summary

In this lesson, we got some hands-on practice working with MongoDB!
