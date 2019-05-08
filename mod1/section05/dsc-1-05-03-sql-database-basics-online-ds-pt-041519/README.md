
# SQL Database Basics 

## Introduction

We'll cover how to create and delete database tables in SQLite as well as how to add columns to an existing table.  

## Objectives

You will be able to:

* Understand the structure of a table in a relational database

## Database Structure

Relational Databases like SQLite store data in a structure we refer to as a table. You can think of a table in a database a lot like you would a spreadsheet. We define specific columns in our table, and then we store any number of what we refer to as 'records' as rows in our database. A record is just information referring to one specific entity. For instance, if you had a table called "People" you could imagine a structure like this:

| name | age | email |
|------|-----|-------|
| Bob | 29 |  bob@flatironschool.com |
| Avi  |  28   | avi@flatironschool.com  |
| Adam |  28   | adam@flatironschool.com |

Each column has a name, and each row contains the corresponding information about a person.

### Note on Column Names

When we name columns in our database, there are a couple of conventions we will follow. The first is that we will always use lowercase letters when referring to columns in our database. SQLite isn't case sensitive about its commands or column names, but it is generally best practice for us to stick to lowercase for our column names.

The second convention we want to follow is more important. That is, when we have multiple words in a column name, we link them together using underscores rather than spaces. We call this convention "snake_case". So, for instance, if we wanted to be more specific with our email column above, we can name it email_address. If we wanted to split up name to first and last we might have columns called first_name and last_name.


## Database Tables

In the following sections, we'll cover how to create, alter, and delete database tables. This reading is accompanied by a code along exercise that you can do in your terminal or here in your jupyter notebook. You don't need to fork this repository, and there are no tests to pass. Follow along with the reading and code along instructions. 

### Following Along In A Jupyter Notebook

To execute SQL statements inside a jupter notebook, we need to write a bit of Python code as well as explicitly import the sqlite3 module. Below, we are going to show how to import, connect to an instance of sqlite3 and create a database file.

First we import the sqlite3 module
```python
import sqlite3
```

Next we create our connection to the sqlite database file `pet_database.db` by using the method `.connect()` and the file name we would like for our database.
```python
connection = sqlite3.connect('pet_database.db')
```

Then we create the *cursor* which we will use to execute SQL statements
```python
cursor = connection.cursor()
```

Finally, when we want to execute our SQL statements we reference our SQL cursor object and call the method `.execute()` with our SQL statement as the argument
```python
sql_return = cursor.execute('''SQL statement GOES here;''')
```
To see a list of the information we retrieved from our SQL statement, we can take our `sql_return` variable and call the method `.fetchall()`, which will return a list of records (if we are executing a `SELECT` statement.


```python
# import sqlite3 and set up a connection and cursor object here to the 'pet_database.db'
```

> **Note:** If you look at your file directory after running the above cell, you will see that there is now a pet_database.db file. Sqlite3 created this for us!

### Create Table

When we create a new database, it comes like a sort of blank slate.

Once you create a database (which you can do with the `sqlite3 database_name.db` command), we create a table using the following statement: 

```sql
CREATE TABLE table_name;
```

But before we're able to store any actual data in a table, we'll need to define the columns in the table as well as the specific type of data each column will store. 

Let's give it a shot. For the purposes of this code along, you'll be typing these commands into your Jupyter Notebook using the SQL `cursor` object we just created. 

### Code Along I: Creating a Table

* Using the new database, connection, and cursor objects we just created above, follow along with the commands below to create your first table:

```python
import sqlite3
connection = sqlite3.connect('pet_database.db')
cursor = connection.cursor()
```

* Now with the cursor object, let's create our table:

```python
cursor.execute('''CREATE TABLE cats;''')
``` 
> **Note:** When we tell Python to execute a sql statement like we do above, it must be wrapped in a string. We use three quotes in a row on either side of the text by convention so that if we have multi-lined SQL statements, it will be sytactically correct. If we only use one quote, we will get an error since Python won't know that the next line of our SQL is apart of the same string.


```python
# create table here
```

You should see the following error:

```sql
---------------------------------------------------------------------------
OperationalError                          Traceback (most recent call last)
<ipython-input-8-e79ad94ed0d5> in <module>()
----> 1 cursor.execute('''CREATE TABLE cats;''')

OperationalError: near ";": syntax error
```

SQLite expects us to include at least some definition of the structure of this table as well. In other words, when we create database tables, we need to specify some column names, along with the type of data we are planning to store in each column. More on data types later. 

Let's try that table statement again:

```python
cursor.execute('''
                CREATE TABLE cats (
                id INTEGER PRIMARY KEY,
                name TEXT, 
                age INTEGER
                );'''
               )
```

Let's break down the above code: 

1. Use the `CREATE TABLE` command to create a new table called "cats"
2. Include a list of column names along with the type of data they will be storing. `TEXT` means we'll be storing plain old text, `INTEGER` means we'll store a number. Note that the use of capitalization is arbitrary, but it is a convention to help separate the SQL commands from the names we make up for our tables and columns. 
3. Every table we create, regardless of the other column names and data types, should be defined with an id INTEGER PRIMARY KEY column, including data type and primary key designation. Our SQLite database tables *must be indexed by a number*. We want each row in our table to have a number, which we'll call "id", just like in an Excel spreadsheet. Numbering our table rows makes our data that much easier to access, update, and organize. SQLite comes with a data type designation called "Primary Key". Primary keys are unique and auto-incrementing, meaning they start at 1 and each new row automatically gets assigned the next numeric value. 

Okay, let's check and make sure that we successfully created that table. To do this we'll be using SQL commands. To list all the tables in the database we'll use the following `SELECT` command. Don't worry too much about this command right now, but know that the `sqlite_master` is name for the sqlite database instance connected to our `cursor` object. Type the command into the next cell and hit enter, you should get:

```python
cursor.execute('''SELECT name FROM sqlite_master WHERE type='table';''').fetchall()

[('cats',)]
```

We can look at the structure, or "schema", of our database (i.e. the tables and their columns + column data types) with the same command but changing `name` to `*` (which really just means get everything).

```python
cursor.execute('''SELECT * FROM sqlite_master WHERE type='table';''').fetchall()

[('table',
  'cats',
  'cats',
  2,
  'CREATE TABLE cats (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)')]
```

Let's move on to altering our table. 

### Alter Table

Let's say that, after creating a database and creating a table to live inside that database, we decide we want to add or remove a column. We can do so with the `ALTER TABLE` statement. 

### Code Along II: Adding, Removing and Renaming Columns

* Let's say we want to add a new column, `breed`, to our `cats` table. 

```python
cursor.execute('''ALTER TABLE cats ADD COLUMN breed TEXT;''')
```


```python
# alter table here
```

Let's check out our schema now: 


```python
# check schema here
```

```python
cursor.execute('''SELECT * FROM sqlite_master WHERE type='table';''').fetchall()

[('table',
  'cats',
  'cats',
  2,
  'CREATE TABLE cats (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, breed TEXT)')]
```

Notice that the `ALTER` statement isn't here, but instead SQLite has updated our original CREATE statement. The schema reflects the current structure of the database, which is reflected as the CREATE statement necessary to create that structure.

* Unfortunately, altering a column name and/or deleting a column can be tricky in SQLite3. There are workarounds, however. We're not going to get into that right now, but you can explore the documentation on this topic [here](https://www.sqlite.org/lang_altertable.html).

Fortunately, SQLite still supports most of what we'll need to use it for one way or another. For now, if you need to change a column name, it's best to simply delete the table and re-create it. 


### Drop Table

Lastly, we'll discuss how to delete a table from a database with the `DROP TABLE` statement.


### Code Along III: Deleting a Table 

Deleting a table is very simple: 

```python
cursor.execute('''DROP TABLE cats;''')
```


```python
# drop table here
```

## Summary

That's it for now! In this lesson, you learned how to create and delete database tables in SQLite as well as how to add columns to an existing table.
