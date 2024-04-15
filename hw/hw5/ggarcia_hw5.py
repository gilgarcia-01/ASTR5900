'''
Gilberto Garcia
Computational Physics
April 15 2024
HW5 - Neural Networks
'''

#import required libraries
import numpy as np

####
#Q1#
####


    #a#

#hard coding a neural network

#we first hard code the pre-defined weights
w_x1n1, w_x1n2, w_x1n3 = -0.91, 0.09, 0.72
w_x2n1, w_x2n2, w_x2n3 = 0.72, 0.34, 0.62
w_n1y, w_n2y, w_n3y = -0.56,0.94,-0.47
b_n1,b_n2,b_n3,b_y = 0.29,0.05,-0.99,-0.25
#define our activation function

#now we define the fxns that connect our nodes
def neural_network(activation_fxn,x1,x2):
    n1 = activation_fxn((w_x1n1 * x1) + (w_x2n1 * x2) + b_n1)
    n2 = activation_fxn((w_x1n2 * x1) + (w_x2n2 * x2) + b_n2)
    n3 = activation_fxn((w_x1n3 * x1) + (w_x2n3 * x2) + b_n3)
    y = activation_fxn((w_n1y*n1) + (w_n2y*n2) + (w_n3y*n3) + b_y)
    return y


    #b#

#use your neural network on the following points:
# point1(0,0),point2(7.5,2.5),point3(-5,-2)


point1 = neural_network(np.tanh,0,0)
print(point1)






