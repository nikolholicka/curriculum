
# HTTP Request/Response Cycle - Codealong

##  Introduction
When developing a Web application as we saw in previous lesson, the request/response cycle is a useful guide to see how all the components of the app fit together. The request/response cycle traces how a user's request flows through the app. Understanding the request/response cycle is helpful to figure out which files to edit when developing an app (and where to look when things aren't working). This lesson will show how this setup works using python. 

## Objectives

You will be able to: 

* Understand and explain the HTTP Request/Response cycle
* Make http requests in Python using the ‘requests’ library

## The `requests` Library in Python

Dealing with HTTP requests could be a challenging task  any programming language. Python with two built-in modules, `urllib` and `urllib2` to handle these requests but these could be very confusing  and the documentation is not clear. This requires the programmer to write a lot of code to make even a simple HTTP request.

To make these things simpler, one easy-to-use third-party library, known as` Requests`, is available and most developers prefer to use it instead or urllib/urllib2. It is an Apache2 licensed HTTP library powered by urllib3 and httplib. Requests is add-on library that allows you to send HTTP requests using Python. With this library, you can access content like web page headers, form data, files, and parameters via simple Python commands. It also allows you to access the response data in a simple way.

![](images/logo.png)

Below is how you would install and import the requests library before making any requests. 
```python
# Uncomment and install requests if you don't have it already
# !pip install requests

# Import requests to working environment
import requests
```


```python
# Code here
```

## The `.get()` Method

Now we have requests library ready in our working environment, we can start making some requests using the `.get()` method as shown below:
```python
### Making a request
resp = requests.get('https://www.google.com')
```


```python
# Code here 
```

GET is by far the most used HTTP method. We can use GET request to retrieve data from any destination. 

## Status Codes
The request we make may not be always successful. The best way is to check the status code which gets returned with the response. Here is how you would do this. 
```python
# Check the returned status code
resp.status_code == requests.codes.ok
```


```python
# Code here 
```

So this is a good check to see if our request was successful. Depending on the status of the web server, the access rights of the clients and availability of requested information. A web server may return a number of status codes within the response. Wikipedia has an exhaustive details on all these codes. [Check them out here](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes).

## Response Contents
Once we know that our request was successful and we have a valid response, we can check the returned information using `.text` property of the response object. 
```python
print (resp.text)
```


```python
# Code here 
```

So this returns a lot of information which by default is not really human understandable due to data encoding, HTML tags and other styling information that only a web browser can truly translate. In later lessons we'll learn how we can use **_Regular Expressions_**  to clean this information and extract the required bits and pieces for analysis. 

## Response Headers
The response of an HTTP request can contain many headers that holds different bits of information. We can use `.header` property of the response object to access the header information as shown below:

```python
# Read the header of the response - convert to dictionary for displaying k:v pairs neatly
dict(resp.headers)
```


```python
# Code here 
```

The content of the headers is our required element. You can see the key-value pairs holding various pieces of  information about the resource and request. Let's try to parse some of these values using the requests library:

```python
print(resp.headers['Date'])  # Date the response was sent
print(resp.headers['server'])   # Server type (google web service - GWS)
```


```python
# Code here 
```

## Try `httpbin`
`httpbin.org` is a popular website to test different HTTP operation and practice with request-response cycles. Let's use httpbin/get to analyze the response to a GET request. First of all, let's find out the response header and inspect how it looks. 

```python
r = requests.get('http://httpbin.org/get')

response = r.json()  
print(r.json())  
print(response['args'])  
print(response['headers'])  
print(response['headers']['Accept'])  
print(response['headers']['Accept-Encoding'])  
print(response['headers']['Host'])  
print(response['headers']['User-Agent'])  
print(response['origin'])  
print(response['url'])  
```


```python
# Code here 
```

Let's use `requests` object structure to parse the values of headers as we did above. 

```python
print(r.headers['Access-Control-Allow-Credentials'])  
print(r.headers['Access-Control-Allow-Origin'])  
print(r.headers['CONNECTION'])  
print(r.headers['content-length'])  
print(r.headers['Content-Type'])  
print(r.headers['Date'])  
print(r.headers['server'])  
```


```python
# Code here 
```

## Passing Parameters in GET
In some cases, you'll need to pass parameters along with your GET requests. These extra parameters usually take the the form of query strings added to the requested URL. To do this, we need to pass these values in the `params` parameter. Let's try to access information from `httpbin` with some user information. 

Note: The user information is not getting authenticated at `httpbin` so any name/password will work fine. This is merely for practice. 

```python
credentials = {'user_name': 'FlatironSchool', 'password': 'learnlovecode'}  
r = requests.get('http://httpbin.org/get', params=credentials)

print(r.url)  
print(r.text)  
```


```python
# Code here 
```

## HTTP POST method 

Sometimes we need to send one or more files simultaneously to the server. For example, if a user is submitting a form and the form includes different fields for uploading files, like user profile picture, user resume, etc. Requests can handle multiple files on a single request. This can be achieved by putting the files to a list of tuples in the form (`field_name, file_info)`.


```python
import requests

url = 'http://httpbin.org/post'  
file_list = [  
    ('image', ('fi.png', open('images/fi.png', 'rb'), 'image/png')),
    ('image', ('fi2.jpeg', open('images/fi2.jpeg', 'rb'), 'image/png'))
]

r = requests.post(url, files=file_list)  
print(r.text)  
```


```python
# Code here  
```

This was a brief introduction to how you would send requests and get responses from a web server, while totally avoiding the web browser interface. Later we'll see how we can pick up the required data elements from the contents of the web page for analytical purpose. 

## Summary
In this lesson, we provided an introduction to the `requests` library in python. We saw how to use the get method to send requests to web servers, check server status, look at the header elements of a web page and how to send extra parameters like user information. 
