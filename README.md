# This page not ready yet!

# Linear-SGD
Linear Regression SGD Optimization Implementation

## Overview
The **SGD** algorithm used as **machine learning** method for weights optimization in a given statistical model. The method based on iterative process, when in each iteration the model learns from the prediction  erorr in order to get better weight values. The code [**'sgd.py'**](https://github.com/EtzionR/Linear-SGD/sgd.py) used as such SGD-Algorithm method in from-scarch implementation for a linear model.

The familiar linear model is based on matrix X and weights B, that for their product we get the Y values (in addition to bulit-in errors):

<img src="https://latex.codecogs.com/svg.image?Y=BX&space;&plus;&space;\varepsilon" title="Y=BX + \varepsilon" />

Given some X and Y, we would like to use the SGD method to find the B values. To do this, we will follow this steps:
1. Sample randomly the B values.
2. Stochastically sample only **subset** from the given data points.
3. We will use B on the selected subset and calculated the **errors** between the prediction to Y.
4. Using the errors we calculated, we will **update** the B values.
5. **Repeat** the 2-4 steps until we get the required iteration times

At the end of the process, if we used enough iterations, we will get close <img src="https://render.githubusercontent.com/render/math?math=\widehat{B}"> values to the real B values. An illustration of the iterative process can be seen in the following GIF:

![iterations](https://github.com/EtzionR/Linear-SGD/blob/main/pictures/iterations.gif)

To update the weight in the currect direction, we need to minimize the errors between our prediction to the actual values of Y. to do so, we want to use the **derivative** of our model loss function (that measure the distance between the prediction and the actual values). our loss look as the following equation:

<img src="https://latex.codecogs.com/svg.image?Loss_{f}&space;=&space;\frac{1}{n}\sum_{i}^{n}(Y&space;-&space;\hat{B}X)^{2}" title="Loss_{f} = \frac{1}{n}\sum_{i}^{n}(Y - \hat{B}X)^{2}" />  (MSE loss)

So, the derivative can found by the following equation:

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;Loss_{f}&space;}{\partial&space;B}&space;=&space;\frac{-2}{n}\sum_{i}^{n}&space;X\cdot&space;(Y-BX)=&space;-2\sum_{i}^{n}&space;X\cdot&space;Errors" title="\frac{\partial Loss_{f} }{\partial B} = -2\sum_{i}^{n} X\cdot (Y-BX)= \frac{-2}{n}\sum_{i}^{n} X\cdot Errors" />

Now, we can use this derivative to update each of our B values:

<img src="https://latex.codecogs.com/svg.image?B_{i&plus;1}&space;=&space;B_{i}&space;-&space;\eta&space;\cdot&space;(\frac{\partial&space;Loss_{f}&space;}{\partial&space;B_{i}})=&space;B_{i}&space;-&space;\eta&space;\cdot&space;(&space;-\frac{2}{n}\sum_{i}^{n}&space;X_{\cdot&space;j}\cdot&space;Errors)" title="B_{i+1} = B_{i} - \eta \cdot (\frac{\partial Loss_{f} }{\partial B_{i}})= B_{i} - \eta \cdot ( -\frac{2}{n}\sum_{i}^{n} X_{\cdot j}\cdot Errors)" />

When <img src="https://render.githubusercontent.com/render/math?math=\eta"> describes the **learning rate** of the change each time (also called "step size"). This equation describes the simple method of updating the weights, while there are also other methods for even better optimization. One of them is **ADAM**, an algorithm designed to find the values of the weights in a particularly efficient and fast way, based on **adjusting the learning rate** for each weight individually, as can be seen in the following equation:

<img src="https://latex.codecogs.com/svg.image?M_{i&plus;1}&space;=&space;(\beta_{1}&space;\cdot&space;M_{i})&plus;&space;(1-\beta_{1})&space;\cdot&space;(\frac{\partial&space;Loss_{f}&space;}{\partial&space;B_{i}})" title="M_{i+1} = (\beta_{1} \cdot M_{i})+ (1-\beta_{1}) \cdot (\frac{\partial Loss_{f} }{\partial B_{i}})" />

<img src="https://latex.codecogs.com/svg.image?V_{i&plus;1}&space;=&space;(\beta_{2}&space;\cdot&space;V_{i})&plus;&space;(1-\beta_{2})&space;\cdot&space;(\frac{\partial&space;Loss_{f}&space;}{\partial&space;B_{i}})^{2}" title="V_{i+1} = (\beta_{2} \cdot V_{i})+ (1-\beta_{2}) \cdot (\frac{\partial Loss_{f} }{\partial B_{i}})^{2}" />

<img src="https://latex.codecogs.com/svg.image?\hat{M_{i+1}}&space;=&space;\frac{M_{i+1}}{1-\beta_{1}^{i+1}},&space;&space;&space;&space;&space;\hat{V_{i+1}}&space;=&space;\frac{V_{i+1}}{1-\beta_{2}^{i+1}}" title="\hat{M_{i+1}} = \frac{M_{i+1}}{1-\beta_{1}^{i+1}}, \hat{V_{i+1}} = \frac{V_{i+1}}{1-\beta_{2}^{i+1}}" />

<img src="https://latex.codecogs.com/svg.image?B_{i&plus;1}&space;=&space;B_{i}&space;-&space;\eta&space;\cdot&space;(\frac{\hat{M_{i&plus;1}}}{\sqrt{\hat{V_{i&plus;1}}}&plus;\varepsilon})" title="B_{i+1} = B_{i} - \eta \cdot (\frac{\hat{M_{i+1}}}{\sqrt{\hat{V_{i+1}}}+\varepsilon})" />

when **i** is the iteration number and <img src="https://latex.codecogs.com/svg.image?\beta_{1}&space;=&space;0.9,&space;\beta_{2}&space;=&space;0.999,&space;M_{0}&space;=&space;0,&space;V_{0}&space;=&space;0,&space;\varepsilon&space;=&space;epsilon,&space;\eta&space;=&space;learning-rate" title="\beta_{1} = 0.9, \beta_{2} = 0.999, M_{0} = 0, V_{0} = 0, \varepsilon = epsilon, \eta = learning-rate"/>

As mentioned, when we compare the two methods, it can be seen that **ADAM** achieves better performance than the simple method (the lr values for this comparsion selected for this illustration are optimal for each of the methods):

![compare](https://github.com/EtzionR/Linear-SGD/blob/main/pictures/adam_vs_simple.png)

It should be noted that the code writed for application on the **multivariate linear model**, but of course is also suitable for implementation for cases of polynomials, as can be seen here:

![poly](https://github.com/EtzionR/Linear-SGD/blob/main/pictures/simple_predicted.png)

For more on the math behind the linear-polynomial regression model, you can see here: [Poly Regression](https://github.com/EtzionR/Polynomial-Regression-Optimizer)

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

