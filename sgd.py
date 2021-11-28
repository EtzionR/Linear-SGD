# Create by Etzion Harari
# https://github.com/EtzionR

# Load libraries:
import matplotlib.pyplot as plt
import numpy as np

# Define useful functions:
linear_gradient = lambda error, ary: -2 * np.mean(ary * error)
rmse = lambda errors: np.sqrt(np.mean(np.power(errors, 2)))

# Define Adam Optimizer:
class ADAM:
    """
     Adaptive Moment Optimization
    """
    def __init__(self,lr,b1=.9,b2=.999,eps=1e-9):
        """
        object initilize
        :param eta: step size (eta)
        :param b1: beta 1 (default: .9)
        :param b2: beta 2 (default: .999)
        :param eps: epsilon (default: 1e-9)
        """
        self.t   = 1
        self.m   = 0
        self.v   = 0

        self.b1  = b1
        self.b2  = b2
        self.lr  = lr
        self.eps = eps

    def optimize(self,grd):
        """
        optimize learning rate and required weight change
        :param grd: gradient
        :return: required change in the weight
        """
        self.m = (self.b1*self.m) + ((1-self.b1)*grd)
        self.v = (self.b2*self.v) + ((1-self.b2)*grd*grd)
        m = self.m/(1-(self.b1**self.t))
        v = self.v/(1-(self.b2**self.t))
        self.t+=1
        return (self.lr*m)/((v**.5)+self.eps)


# Define Linear SGD Object:
class LinearSGD:
    """
    Linear SGD Regreesor Calculator
    Bulid to use sgd to predict the betas vector,
    using stochastic gradient descent method
    """
    def __init__(self, lr=1e-2, iters=100, sample_rate=0.5,adam=True):
        """
        initilize the Linear SGD object
        :param lr: float, the requied learning rate (default: .5)
        :param iters: int, the required number of algorithm iterations (default: 100)
        :param sample_rate: float, the % of samples for gradient calculation
                                   in each iteration (default: .5)
        """
        # model variables
        self.iters = iters
        self.adam  = adam
        self.loss  = []
        self.sr    = sample_rate
        self.lr    = lr

        # fitting variables
        self.adams = None
        self.betas = None
        self.n     = None
        self.m     = None

    def create_matrix(self, x):
        """
        create x matrix from given input (vector/matrix)
        :param x: input features data
        :return: x matrix
        """
        x = np.array(x)
        if len(x.shape) == 1:
            return np.array([np.ones(x.shape[0]), x]).T
        else:
            return np.array([np.ones(x.shape[0])] + [xi for xi in x.T]).T

    def update_weights(self, subset_x, errors):
        """
        update the betas values by using the sgd
        :param subset_x: subset of x matrix
        :param errors: vector of error between prediction and y values
        """
        self.loss.append(rmse(errors))
        for j in range(self.m):
            if self.adam:
                self.betas[j] -= self.adams[j].optimize(linear_gradient(errors,subset_x[:,j]))
            else:
                self.betas[j] -= self.lr*linear_gradient(errors,subset_x[:, j])

    def fit(self, x, y):
        """
        fitting the betas values using sgd
        :param x: input x values
        :param y: input y values
        """

        # prepere the data for fitting
        x = self.create_matrix(x)
        y = np.array(y)
        self.n = x.shape[0]
        self.m = x.shape[1]
        self.betas = np.random.uniform(0, 1, self.m)

        # if we want use adam optimizer, we should inilize the ADAM objects
        if self.adam:
            self.adams = [ADAM(self.lr) for _ in range(self.m)]

        # fitting the betas to the data
        for i in range(self.iters):
            # select the subset of x & y and calculate the errors
            selection = np.random.random(self.n) < self.sr
            subset_x = x[selection, :]
            errors = y[selection] - np.dot(subset_x, self.betas)

            # update the betas using the loss
            self.update_weights(subset_x, errors)

        return self

    def predict(self, x):
        """
        predict the y value using the calculated betas
        :param x: input x features
        :return: y - output prediction
        """
        return np.dot(self.create_matrix(x), self.betas)

    def plot_loss(self, size=10):
        """
        plot the loss by iterations
        :param size: plot size
        """
        plt.figure(figsize=(size, size * .6))
        plt.title(f'Prediction Loss by number of Iterations:', fontsize=15)
        plt.plot(np.arange(self.iters), self.loss, color='r', label='loss')
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)
        plt.ylim(0, self.loss[0] * 1.05)
        plt.show()

# License
# MIT © Etzion Harari