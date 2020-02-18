# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:16:49 2019

@author: furkan
"""

#%%

import pandas as pd
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import losses

#%% activation functions 

def relu(x):
    y = np.copy(x)
    for i in range(len(y)):
        y[i] = max(0            ,y[i])
    return y
def sigmoid(x):
    y = np.copy(x)
    for i in range(len(y)):
        y[i] = 1/(1+math.exp(-y[i]))

    return y
def empty(x):
    return x

    # derivations of the activation functions

def d_relu(x):
    y = np.copy(x)
    for i in range(len(y)):
        if(y[i]>0):
            y[i] = 1
        else:
            y[i] = 0
    return y
def d_sigmoid(x):
    return np.multiply(sigmoid(x),np.multiply(sigmoid(x),-1)+1)
def d_empty(x):
    return 0


#%%
class Sequence:

  def __init__(self):
    self.w = []
    self.b = []
    self.func = []
    self.act = {'relu' : relu,'sigmoid' : sigmoid, 'empty' : empty}
    self.d_act = {'relu' : d_relu,'sigmoid' : d_sigmoid, 'empty' : d_empty}
    self.err = []
    self.inputs = []
    self.errorRate = []
    self.sum = []
    self.averageError = sys.maxsize

    # adding layers that are initialized in dense function to the corresponding lists 

  def add(self,w,b,func,err):
    self.w.append(w)
    self.b.append(b)
    self.func.append(func)
    self.err.append(err)
  
    # initializing the layers, inputs and outputs

  def Dense(self, numberOfNodes = 2, input = 0, activation = 'relu'):
    if (input == 0):
        input = self.w[-1].shape[0]
    w = np.random.rand(numberOfNodes, input)
    b = np.resize(np.random.rand(numberOfNodes,1),(numberOfNodes,))
    func = activation
    err = np.zeros((numberOfNodes,1))
    return w,b,func,err

    # feedforward is generate output from input

  def feedForward(self, input):
    self.inputs = []
    self.sum = []
    for i in range(len(self.w)):
        self.inputs.append(np.resize(input,(len(input),1)))
        summed = np.add((self.w[i]@input) , self.b[i] )
        summed = np.resize(summed,(len(summed),1))
        
        self.sum.append(summed)
        input = self.act[ self.func[i] ](summed)
    return input

    # error function for neural network

  def calculateError(self, target, output):
     error = np.sum(np.multiply(np.subtract(output,target),np.subtract(output,target))) / len(target)
     
     self.err[len(self.err)-1] = np.multiply( self.d_act[self.func[len(self.func)-1]](self.sum[len(self.sum)-1]), error)
     self.err[len(self.err)-1] = np.resize(self.err[len(self.err)-1], (self.err[len(self.err)-1].shape[0],1))
     for i in range((len(self.err)-2),-1,-1):
         dot = np.transpose(self.w[i+1])@self.err[i+1]
         self.err[i] = np.negative(np.multiply(self.d_act[self.func[len(self.func)-1]](self.sum[i]), np.resize(dot, (dot.shape[0],1))))

     return np.subtract(output,target)
     
#  def backpropogateError(self,lr):
#      for i in range(len(self.inputs)):
#          self.w[i] = np.subtract(self.w[i], np.multiply(self.err[i]@np.transpose(self.inputs[i]),lr))
#          self.b[i] = np.resize(np.subtract(np.resize(self.b[i], (self.b[i].shape[0],1)), np.multiply(self.err[i],lr)) , (self.b[i].shape[0],))
  
    # backpropogation for updating neural network's weight and biases

  def backpropogateError(self,lr):
      for i in range(len(self.inputs)):
          self.w[i] = np.subtract(self.w[i], np.multiply(self.err[i]@np.transpose(self.inputs[i]),lr))
          self.b[i] = np.resize(np.subtract(np.resize(self.b[i], (self.b[i].shape[0],1)), np.multiply(self.err[i],lr)) , (self.b[i].shape[0],))

    # calculate error for updating error rate

  def calculateErrorRate(self,errorRate, iteration):
      errorRate = np.asarray(errorRate)
      self.averageError = (self.averageError*(iteration-1) + math.sqrt(np.sum(np.multiply(errorRate,errorRate))))/iteration
      self.errorRate.append(self.averageError)
      return self.averageError
  
    # train and test functions to train the model and evaluate it

  def train(self, x_train, y_train, error = 0.01, learning_rate = 0.1, epoch_l = 200000):
      self.errorRate = []
      errorRate = sys.maxsize
      epoch = 0
      iteration = 0
      i = 0
      target = y_train[i]
      output = target*2
      #print("train: errorRate: " + str(errorRate))
      while((errorRate > error) and epoch < epoch_l):
          iteration += 1
          if(i < len(x_train)-1):
              i += 1
          else:
              i = 0
              epoch += 1
          output = self.feedForward(x_train[i])
          errorRate = self.calculateErrorRate(self.calculateError(y_train[i], output), iteration)
          print("x,y: " + str(x_train[i]) + " | output generated: " + str(output) + " <- y_train: "+ str(y_train[i]))
          self.backpropogateError(learning_rate)
          print("-- epoch: " + str(epoch) + " -- iter: " + str(iteration) + " errorRate: " + str(errorRate))
      return self.errorRate
  
  def test(self, x_test, y_test, error = 0.01):
      perc = 0
      for i in range(len(x_test)):
          output = self.feedForward(x_test[i])
          errorRate = self.calculateError(y_test[i], output)
          e = np.sum(np.multiply(errorRate,errorRate))/len(errorRate)
          if(e <= error):
              perc += 1
              print("-- x_test --")
              print(x_test[i])
              print("output")
              print(output)
              
      perc /= len(x_test)
      return perc*100

# split function for spliting the data to train and test sets

  def split(self, x, y, percentage = 0.2):
      x_train = np.asarray(x.head(int(x.shape[0]*(1-percentage))).values)
      y_train = np.asarray(y.head(int(y.shape[0]*(1-percentage))).values)
      x_test = np.asarray(x.tail(int(y.shape[0]*percentage)).values)
      y_test = np.asarray(y.tail(int(y.shape[0]*percentage)).values)
      return x_train, y_train, x_test, y_test
  





#%% Data Spesifications are given by the Neural Network Course Project

def f(x , y):
    return (x*x) + (y*y)

def generateData(n):
  df = pd.DataFrame(columns = ['x', 'y', 'result'])
  for i in np.linspace((-1)*n,n,2*n+1):
      for j in np.linspace((-1)*n,n,2*n+1):
          x = i
          y = j
          result = f(x,y)
          df = df.append({'x' : x, 'y' : y, 'result' : result}, ignore_index = True)
  df = df.sample(frac=1).reset_index(drop=True)
  return df

#%% Data Generation, Normalization, Splitting Data to train and test sets

df = generateData(5)

scaler = preprocessing.MinMaxScaler()

x = df[['x', 'y']]
y = df[["result"]]

#scaled_x = scaler.fit_transform(x)
scaled_y = scaler.fit_transform(y)

#scaled_x = pd.DataFrame(scaled_x, columns=['x', 'y'])
scaled_y = pd.DataFrame(scaled_y, columns=['result'])

x_train, y_train, x_test, y_test = model.split(x,scaled_y)

#%% Model with my library

model = Sequence()

w,b,func,err = model.Dense(3,2,'relu')
model.add(w,b,func, err)

w,b,func,err = model.Dense(4,0,'sigmoid')
model.add(w,b,func, err)

w,b,func,err = model.Dense(2,0,'sigmoid')
model.add(w,b,func, err)

w,b,func,err = model.Dense(1,0,'sigmoid')
model.add(w,b,func, err)


#%% train and test with my library

error = model.train(x_train, y_train, error =0.01, learning_rate = 0.01, epoch_l = 800)
model.test(x_test,y_test,error = 0.05)




#%% Model with keras


model_k = Sequential()
model_k.add(Dense(16, input_dim=2, activation='relu'))

model_k.add(Dense(1, activation='sigmoid'))
    
sgd = SGD(lr=0.01, momentum=0.0, nesterov=False)
model_k.compile(loss=losses.mean_squared_error, optimizer='sgd')


#%% train and test with keras

model_k.fit(x_train, y_train, epochs=500, batch_size=1)
accuracy = model_k.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
