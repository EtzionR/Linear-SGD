# load sgd object and numpy
from sgd import LinearSGD
import numpy as np


# example 1
x = np.load(r'examples\example_1_x.npy')
y = np.load(r'examples\example_1_y.npy')

loss = LinearSGD(iters=500).fit(x,y).loss[-1]
print(f'Loss for the last iteration = {round(loss,3)}')

# example 2
x = np.load(r'examples\example_2_x.npy')
y = np.load(r'examples\example_2_y.npy')

print(f'Predicted Betas = {LinearSGD(iters=500).fit(x,y).betas},  Actual Betas = [-2, 1.5]')


# example 3
x = np.load(r'examples\example_3_x.npy')
y = np.load(r'examples\example_3_y.npy')

LinearSGD(iters=300,lr=0.02).fit(x,y).plot_loss()