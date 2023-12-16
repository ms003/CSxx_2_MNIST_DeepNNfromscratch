#!/usr/bin/env python
# coding: utf-8

# # Assignment 02 Part 2: Neural Net Implementation
# 

# 
# ### Artificial Neural Net Class

#%% Loading the data

import numpy as np
import matplotlib.pyplot as plt
import sys
from os import system, name 
from time import sleep
import pandas as pd
import time

#%%


###Load file

image_size = 28
image_pixels = image_size * image_size
data_path = "./"
train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")




#%%
train_set = np.asfarray(train_data[:10, 1:])
test_set = np.asfarray(test_data[:10, 1:])

#Adding a column with teh bias and scaling the data
nrow_tr, ncol_tr = np.shape(train_set)
nrow_tst, ncol_tst = np.shape(test_set)



bias_tr= np.ones((nrow_tr, 1))
train_set = np.hstack((bias_tr, train_set))


bias_tst = np.ones((nrow_tst, 1))
test_set = np.hstack([bias_tst, test_set])

train_setscaled = (train_set - np.mean(train_set)) / np.std(train_set)
test_setscaled = (test_set - np.mean(test_set)) / np.std(test_set)

train_label = train_data[:10,0]
test_label = test_data[:10,0]

# In[1]:


import numpy as np
import random

class ANN(object):

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs=784, input_layer = 784, no_hidden_layers=2, hidden_layer_size=28, output_layer_size=1, max_iterations=20, learning_rate=0.02):

        self.no_inputs = no_inputs
        self.input_layer_size = input_layer
        self.no_hidden_layers = no_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    #===================================#
    # Performs the activation function. #
    # Expects an array of values of     #
    # shape (1,N) where N is the number #
    # of nodes in the layer.            #
    #===================================#
    def weights(self,input_layer_size, hidden_layer_size, output_layer_size): #no_hidden_layers): 
    
         bias = np.ones((self.hidden_layer_size, 1))
         
         self.weights_init = (2*np.random.random((self.hidden_layer_size, self.input_layer_size)) -1) / self.hidden_layer_size
         #self.weights_init = np.random.random((self.hidden_layer_size, self.input_layer_size)) * np.sqrt(1/self.hidden_layer_size)

         #self.weights_init = np.random.random((self.hidden_layer_size, self.input_layer_size)) * np.sqrt(1/self.hidden_layer_size)
         self.weights_init = np.hstack([bias, self.weights_init ])
         #print(self.weights_init.shape)
         
         self.weights_hidden = []
         for i in range(self.no_hidden_layers):
            hidden_weights = (2*np.random.random((self.hidden_layer_size, self.hidden_layer_size)) -1) / self.hidden_layer_size
            #hidden_weights = np.random.random((self.hidden_layer_size, self.hidden_layer_size)) * np.sqrt(1/self.hidden_layer_size)

            hidden_weights  = np.hstack([bias, hidden_weights ])
            self.weights_hidden.append(hidden_weights)
            #print('hidden weight shape', self.weights_hidden[i].shape)
         
         
         output_bias = np.ones((self.output_layer_size, 1))
         weights_output = (2*np.random.random((self.output_layer_size, self.hidden_layer_size)) -1)/ self.output_layer_size
         #weights_output = np.random.random((self.output_layer_size, self.hidden_layer_size)) * np.sqrt(1/ self.output_layer_size)
         self.weights_output = (np.hstack([output_bias, weights_output ]))
        
         
         return (self.weights_init, self.weights_hidden, self.weights_output)

    
    
    
       
    
    def sigm_activ(self, a):
        a = np.array(a, dtype = np.float128)
        sigmoid = 1 / (1 + np.exp(-1*a))
        return sigmoid
    
    def rectifier_activ(self,a):
       

        for index, i in np.ndenumerate(a):
            #print(index, i)
            if  i<=0:
                a[index]=0
        #print('this is activa', a, a.shape)
        return a
        
        


   
    def forward_phase_sigmoid(self, train_set, labels, back_propag):
        assert len(train_set) == len(labels)
      
        total_layers = self.no_hidden_layers + 2
        
        labels0 = labels[:,np.newaxis] 
        labels0 = labels0[:,np.newaxis] 

        train_set0 = train_set[:,np.newaxis,:]
        for j in range(self.max_iterations):
             
            self.last_output = []
            
            for i in range(total_layers):
                

                if  i == 0:
                    a = np.dot(train_set0 , self.weights_init.T)
                    a = a /np.average(a)
                    #print("inputlayer activa", a, a.shape)
               
                
                elif i < (total_layers - 1):
                   a = np.dot(self.last_output[i-1], self.weights_hidden[i-1].T)
                   a = a /np.average(a)
                   #print("hidden l shape", a.shape)
                else:
                    a = np.dot(self.last_output[i-1], self.weights_output.T)
                    a = a /np.average(a)
                    #print("output lay shape",a.shape )
                
                if i < (total_layers - 1):
                  
                    shape_ =self.sigm_activ(a).shape
                    add_bias = np.ones((shape_[0],1,1))
                    output = np.dstack((add_bias , self.sigm_activ(a)))
                
                    self.last_output.append(output)
                        
                        
    
                else:
                    last_shape = a.shape
                    add_last_bias = np.ones((last_shape[0],1,1))
                    last_output = np.dstack((add_last_bias, a))
                    self.last_output.append(last_output)
                     
                    if back_propag == True:
                        self.back_propagation_sigmoid(self.last_output, labels0, train_set0)                   
                        #     self.back_propagation_rectifier(self.last_output, labels0, train_set0)
                    elif  back_propag== False:
                            return self.last_output
                
            
    
            
            
            
    def back_propagation_sigmoid(self, output, labels, data_set):
        
        total_layers = self.no_hidden_layers + 2   
        self.output_error = self.last_output[-1][:,:,1:] - labels
        
        flat_out_error  = np.average(self.output_error, axis =0)
        self.flat_outputs = []
        for i in range(len(self.last_output)):
            self.flat_outputs.append(np.average(self.last_output[i],axis =0))
            
     
       
        "Output Gradient"
        self.output_gradient = np.dot(flat_out_error.T, self.flat_outputs[-2])# thi needs to be transposed probably to have it as(1000, 10,28)??
            
        
        
        
        self.df =[]
      
        for i in range(len(self.last_output) -1):
            self.df.append(self.flat_outputs[i][:,1:]*(1- self.flat_outputs[i][:,1:]))

  
        self.errors = []
        self.errors.append(np.dot(np.dot(self.df[2].T , flat_out_error) , self.weights_output[:,1:]))
        self.errors.append((np.dot(self.df[1], self.errors[0])* self.weights_hidden[1][:,1:]))
        self.errors.append((np.dot(self.df[0], self.errors[1])* self.weights_hidden[0][:,1:]))


        #Remember to reverse your list
        self.errors_rev = self.errors[::-1]
        
        
        self.flat_train = np.average(train_set, axis = 0)[:,np.newaxis]
        
        
        self.hidden_gradients = []
        for i in range(len(self.errors_rev)):
            if  i ==0:
                hidden_gradient =np.dot(self.flat_train ,self.errors_rev[i][:,np.newaxis])
                hidden_gradient = np.average(hidden_gradient, axis =2)
                #print('aver,hid grad', hidden_gradient.shape)
                self.hidden_gradients.append(hidden_gradient)
                #print('flat hid error new axis', self.errors_rev[i][:, np.newaxis].shape)
                #print('input layper gradient', self.hidden_gradients[i].shape)
            else:
                hidden_gradient = np.dot(self.flat_outputs[i-1].T , self.errors_rev[i][:, np.newaxis])
                hidden_gradient = np.average(hidden_gradient, axis =2)
                self.hidden_gradients.append(hidden_gradient)
            
        
            
        
        
      
        "Update Weights!!"
        self.weights_output = self.weights_output - self.learning_rate * self.output_gradient
        
        for i in range(total_layers-1):
            if i == 0:
                self.weights_init = self.weights_init - self.learning_rate * self.hidden_gradients[i].T
                #print(self.weights_init.shape)
            else:
                self.weights_hidden[i-1] = self.weights_hidden[i-1] -  self.learning_rate * self.hidden_gradients[i].T
                #print(self.weights_hidden[i-1].shape)
    
        return  self.weights_init, self.weights_hidden, self.weights_output
 


    def forward_phase_rectifier(self, train_set, labels, back_propagation):
        self.total_time  = time.perf_counter()
        
        assert len(train_set) == len(labels)
      
        
        total_layers = self.no_hidden_layers + 2
        
        for j in range(self.max_iterations):
            
            predictions =[]
            for train, label in zip(train_set, labels):
            
                
                self.last_output = []
                activations = []
                for i in range(total_layers):
                    
    
                    if  i == 0:
                        a = np.dot(train, self.weights_init.T)
                        a = a - np.std(a)
                        activations.append(a)
                        #print('a input',a)
                    
                    
                    elif i < (total_layers - 1):
                        a = np.dot(self.last_output[i-1], self.weights_hidden[i-1].T)
                        a = a - 2*np.std(a)
                        activations.append(a)
                        #print('a hidden', a)
                    else:
                        a = np.dot(self.last_output[i-1], self.weights_output.T)
                        a = a - np.std(a)
                        activations.append(a)
                        #print('a output', a)
                    
                    
                    
                    if i < (total_layers - 1):
                    
                        output = np.hstack(([1] , self.rectifier_activ(a)))
                
                        self.last_output.append(output)
                            
                            
                    else:
                        
                        last_output = np.hstack(([0.5] , a))
                        self.last_output.append(last_output)
                        
                        if back_propagation ==True:
                            self.back_propagation_rectifier(self.last_output, label, train, activations)
                        elif  back_propagation== False:
                            predictions.append(a)
            return predictions
                    
            
    


           
    def back_propagation_rectifier(self, output, labels, train,activations):
        
        total_layers = self.no_hidden_layers + 2   
        
        self.output_error = self.last_output[-1][1:] - labels
        
        if self.output_error ==0:
            print('NN has learnt')
        else:
            
  
            print('Start Back Propagation')
         
           
            self.output_gradient = self.output_error * self.last_output[-2]# thi needs to be transposed probably to have it as(1000, 10,28)??
                
            
            
            
            self.df =[]
          
            for i in range(len(activations) -1):
                list_ =[]
                for j in activations[i]:
                    #print('j is', j)
                    if j<= 0:
                        list_.append(0)
                        
                    else :
                        list_.append(1)
                self.df.append(list_)
                #print(list_)
                        
                    
        
            self.errors = []
            self.errors.append(self.df[2] * self.output_error * self.weights_output[:,1:])
            self.errors.append(self.df[1]* self.errors[0]* self.weights_hidden[1][:,1:])
            self.errors.append(self.df[0] * self.errors[1] * self.weights_hidden[0][:,1:])
    
            #Remember to reverse your list
            self.errors_rev = self.errors[::-1]
            
            
      
    
         
            
            self.hidden_gradients = []
            for i in range(len(self.errors_rev)):
                if  i ==0:
                    hidden_gradient = np.dot(train[:,np.newaxis] ,self.errors_rev[i][:,np.newaxis])
                   
                    self.hidden_gradients.append(hidden_gradient)
                else:
                    hidden_gradient = np.dot(self.last_output[i-1][:,np.newaxis] , self.errors_rev[i][:, np.newaxis])
                    self.hidden_gradients.append(hidden_gradient)
                
            
                
            flat_gradients =[]
            for i in range(len(self.hidden_gradients)):
                flat_gradients.append(np.average(self.hidden_gradients[i], axis =2))
           
            
            
            "Update weights !!!!"
            self.weights_output = self.weights_output - self.learning_rate * self.output_gradient
            #print('weights out after',self.weights_output )
            for i in range(total_layers-1):
                if i == 0:
                    #print('weights ini before', self.weights_init)
                    self.weights_init = self.weights_init - self.learning_rate * flat_gradients[i].T
                    #print('weights ini after', self.weights_init)
                else:
                    #print('weights hid before', self.weights_hidden[i-1])
                    self.weights_hidden[i-1] = self.weights_hidden[i-1] -  self.learning_rate * flat_gradients[i].T
                    #print('weights hid after', self.weights_hidden[i-1])
            return  self.weights_init, self.weights_hidden, self.weights_output, self.total_time
        
       
            
       
    def predict_rectifier(self, data_set, data_labels):
        assert len(test_set) == len(data_labels)
    
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
       
        
    
    
            
        predictions  = self.forward_phase_rectifier(data_set, data_labels, False)
        #print('predictions', predictions)
     
        for label, prediction in zip(data_labels, predictions):
            print('label, prediction', label, prediction)
            
            if label==prediction :
                #print('true positive', prediction)
                true_positive +=1
            elif label != prediction:
                true_negative +=1
            elif label != label and prediction == label:
                false_positive +=1
            elif label == label and prediction != label:
                false_negative +=1

        
        accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_positive +false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive /(true_positive + false_negative)   
            
        print("Accuracy:\t"+str(accuracy))
        print("Precision:\t"+str(precision))
        print("Recall:\t"+str(recall))
        print("true_positives", true_positive)
        print('true_negatives', true_negative)
        print('false_positives', false_positive)
        print('false_negatives', false_negative)
        return accuracy, precision, recall
    
    def predict_batch(self, test_set, test_labels):
        assert len(test_set) == len(test_labels)
    
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
       
     
        self.predictions = self.forward_phase_sigmoid(test_set, test_labels, 0)
      

        
        for test_label, prediction in zip(test_labels, self.predictions[3][:, :,1:]):
        

          
            for i in prediction:
                for j in i:
                    test_label=1
                    j = round(j, 2)
                    if test_label ==1 and j ==1 :
                        print('true positive', j)
                        true_positive +=1
                    elif test_label == 1 and j !=1:
                        true_negative +=1
                    elif test_label !=1 and j ==1:
                        false_positive +=1
                    elif test_label == 1 and j !=1:
                        false_negative +=1
               

            
                    
        accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_positive +false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive /(true_positive + false_negative)   
            
        print("Accuracy:\t"+str(accuracy))
        print("Precision:\t"+str(precision))
        print("Recall:\t"+str(recall))
        print("true_positives", true_positive)
        print('true_negatives', true_negative)
        print('false_positives', false_positive)
        print('false_negatives', false_negative)
        return accuracy, precision, recall  



#%%

print("NN Sigmoid")
nn=ANN(784,)


nn.weights(784,28,10)


nn.forward_phase_sigmoid(train_set, train_label, True)
nn.predict_batch(train_set, train_label)



#%%

print('NN with Rectifier Function')
nn.forward_phase_rectifier(train_set, train_label, True)


nn.predict_rectifier(train_set, train_label)



