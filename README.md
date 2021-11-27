# This page not ready yet!

# Linear-SGD
Linear Regression SGD Optimization Implementation

## Overview
The **SGD** algorithm used as **machine learning** method for weights optimization for given statistical model. The method based on multiple iterations, when in each iteration the model learns from the error of the prediction in order get better weight values. The code [**'sgd.py'**](https://github.com/EtzionR/Linear-SGD/sgd.py) used as from-scarch implementation of the SGD-Algorithm for a linear model.

The familiar linear model is based on matrix X and weights B, that for their product we get the Y values (in addition to bulit-in errors):
<img src="https://render.githubusercontent.com/render/math?math=Y=BX+\varepsilon">

Given some X and Y, we would like to find the B values. To do this, we will use the following method:
1. Random B values ​​are randomly selected
2. We will stochastically sample only some of the data points
3. We will use B on the subset we have chosen and check with Y what the value of its errors is
4. Using the errors we collected, we will update the B values
5. We will repeat the previous steps several times

At the end of the process, if we repeat it enough times, we will get values ​​close to the true B values. An illustration of the iterative process can be seen in the following GIF:

The process of updating the B values ​​we do through the graduate of the distribution of our model, and through it we update our weights, as can be seen in the following equation:

When N describes the learning rate of the change each time. This equation describes the simple method of updating the weights, while there are also other methods for even better optimization. One of them is ADAM, an algorithm designed to find the values ​​of the weights in a particularly efficient and fast way, based on adjusting the learning rate for each weight individually, as can be seen in the following equation:

As mentioned, when we compare the two methods, it can be seen that ADAM really achieves better performance than the simple method (the lr values ​​selected for this illustration are optimal for each of the methods):

It should be noted that the code is intended for application for the multivariate linear model, but of course is also suitable for implementation for cases of polynomials, as can be seen here:

For more on linear-polynomial regression, see here:

The attached code also includes a dedicated plot command for displaying the loss values ​​generated by the code so that its performance can be tested:

## Libraries
The code uses the following library in Python:

**matplotlib**

**numpy**

## Application
An application of the code is attached to this page under the name: 

[**implementation.**]()

The examples are also attached here [data](https://github.com/EtzionR/My-TF-AutoEncoder/tree/main/data).


## Example for using the code
To use this code, you just need to import it as follows:
``` sh
# import code

# load data

# define variables

# using the code

# fitting the model

# get prediction

```

When the variables displayed are:

**lr:** float, the learning rate  (defualt = .001)

## License
MIT © [Etzion Harari](https://github.com/EtzionR)

