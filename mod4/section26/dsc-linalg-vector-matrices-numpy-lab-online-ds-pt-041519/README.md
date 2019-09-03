
# Vectors and Matrices in Numpy - Lab

## Introduction

In this lab, you'll solve some simple matrix creation and manipulation exercises based on what you've learned so far in this section. The key takeaway here is to be able to understand how to use indexing with matrices and vectors while applying some basic operations.

## Objectives
You will be able to:
* Define vectors and matrices in NumPy
* Check the shape of vectors and matrices
* Access and manipulate individual scalar components of a matrix. 

## 1. Define two arrays $A$  with shape $ (4 \times 2)$ and $B$ with shape $(2 \times 3)$ 
So    $A =    
  \left[ {\begin{array}{cc}
   1402 & 191 \\
   1371 &  821\\
   949 &  1437 \\
   147 & 1448 \\
  \end{array} }\right]
$
and
$
B =    
  \left[ {\begin{array}{ccc}
   1 & 2 & 3 \\
   4 & 5 & 6\\
  \end{array} }\right]
$


```python
# Code Here
```

    A=
     [[1402  191]
     [1371  821]
     [ 949 1437]
     [ 147 1448]]
    B=
     [[1 2 3]
     [4 5 6]]


## 2. Print the dimensions of $A$ and $B$ 


```python
# Code Here
```

    Shape of A: (4, 2)
    Shape of B: (2, 3)


## 3. Print some of the elements from $A$: 
* first row and first column
* first row and second column
* third row and second column
* fourth row and first column


```python
# Code Here
```

    1402
    191
    1437
    147


## 4. Write a routine to populate a matrix with random data
* Create an $(3 \times 3)$ Numpy array with all zeros (use `np.zeros()`)
* Access each location $(i,j)$ of this matrix and fill in random values between the range 1 and 10. 


```python
# Code Here (due to random data , your output might be different)
```

    before random data:
     [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    
    after random data:
     [[2. 7. 5.]
     [7. 9. 3.]
     [6. 5. 9.]]


## 5. Turn the above routine into a function
* Create two $(4 \times 4)$ zero valued matrices and fill with random data using the function
* Add the matrices together in numpy 
* Show the results


```python
# Code Here (due to random data , your output might be different)
```

    Final output
    
     [[12.  4. 13.  8.]
     [13.  5.  9.  9.]
     [11. 11. 11. 17.]
     [16. 14.  8. 10.]]


## Summary 

In this lab, we saw how to create and manipulate vectors and matrices in numpy. We shall now move forward to learning more complex operations including dot products and inverses. 
