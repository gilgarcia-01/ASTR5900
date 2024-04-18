'''
Gilberto Garcia
Computational Physics
April 15 2024
HW5 - Neural Networks
'''

#import required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm 

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
point2 = neural_network(np.tanh,7.5,2.5)
point3 = neural_network(np.tanh,-5,-2)


    #c#
N = 100
value_matrix = np.zeros((N,N))
x,y = np.linspace(-10,10,N),np.linspace(-10,10,N)

for i in range(value_matrix.shape[0]):
    for j in range(value_matrix.shape[1]):
        value_matrix[i,j] = neural_network(np.tanh,x[i],y[j])

#X,Y = np.meshgrid(x,y)


cs = plt.contourf(x,y,value_matrix, levels=N,cmap='terrain')
cbar = plt.colorbar(cs) 
plt.show() 



    #d#

#we change the weights now for fun
w_x1n1, w_x1n2, w_x1n3 = -0.5, 0.21, 0.9
w_x2n1, w_x2n2, w_x2n3 = 0.4, 0.4, 0.2
w_n1y, w_n2y, w_n3y = -0.33,-0.94,-0.5
b_n1,b_n2,b_n3,b_y = 0.5,0.3,-0.1,-0.75


N = 100
value_matrix = np.zeros((N,N))
x,y = np.linspace(-10,10,N),np.linspace(-10,10,N)

for i in range(value_matrix.shape[0]):
    for j in range(value_matrix.shape[1]):
        value_matrix[i,j] = neural_network(np.tanh,x[i],y[j])

cs = plt.contourf(x,y,value_matrix, levels=N,cmap='terrain')
cbar = plt.colorbar(cs) 
plt.show() 





####
#Q3#
####

import tensorflow as tf
from sklearn.preprocessing import normalize
import pandas as pd
#read in our data first
magic04 = pd.read_csv('magic04.data',sep=',',index_col=None,header=None)
#we randomize our rows
magic04 = magic04.sample(frac=1)

#we want to create training and testing data sets
#to do this, we split our magic04 dataset into these two components
#we will use 80% of our data for the training set and 20% for the testing set
sample_percent = 0.8
training_index = int(sample_percent*magic04.shape[0])
testing_index = magic04.shape[0] - training_index
training = magic04[:training_index]
testing = magic04[testing_index:]
#we split both of them into sample and label data sets
#the sample is the data (x values) and the label is the output (y values, detection or non-detection)
training_sample,training_label = training.iloc[:,:-1],training.iloc[:,-1] 
testing_sample,testing_label =   testing.iloc[:,:-1],testing.iloc[:,-1] 


#we have our data ready, we now want to create and ready our model
#we will create a model with 1 hidden layer of 100 nodes and 1 outer layer with 2 nodes
#we use the code from class to make the model, compile, and test it
counter = 0
for nodes in [100,50,100,100]:
    if counter == 3:
        training_sample = normalize(training_sample)
        testing_sample = normalize(testing_sample)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(10,)),
        tf.keras.layers.Dense(nodes, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    #we ready our model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    #we can now pass our data to the model and train it
    if counter == 0:
        epochs = 10
    else:
        epochs = 20
    counter +=1
    batch_size = 32
    validation_split = 0.2
    history = model.fit(training_sample,
                        training_label,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)


    #we can evaluate the model and print our accuracy
    test_loss, test_acc = model.evaluate(testing_sample, testing_label)
    print(f"Test Accuracy: {test_acc:.3f}")

    #let's plot the accuracy across epochs
    plt.xlabel("Training Epoch")
    plt.ylabel("Accuracy")
    plt.plot(history.epoch, history.history['accuracy'], marker = '.', label = 'Train Accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], marker = '.', label = 'Val Accuracy')
    plt.legend()
    plt.show()


