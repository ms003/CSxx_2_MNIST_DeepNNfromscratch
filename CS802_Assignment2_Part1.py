#!/usr/bin/env python
# coding: utf-8

# # Assignment 02 Part 1: Perceptron Template
# 
# This file contains the template code for the Perceptron.
# 
# ### Perceptron Class

# In[1]:


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


train_set = np.asfarray(train_data[:, 1:])
test_set = np.asfarray(test_data[:, 1:])

#Adding a column with the bias and scaling the data
nrow_tr, ncol_tr = np.shape(train_set)
nrow_tst, ncol_tst = np.shape(test_set)

bias_tr= np.ones((nrow_tr, 1))
train_set = np.hstack([train_set, bias_tr])

bias_tst = np.ones((nrow_tst, 1))
test_set = np.hstack([test_set, bias_tst])

train_setscaled = (train_set - np.mean(train_set)) / np.std(train_set)
test_setscaled = (test_set - np.mean(test_set)) / np.std(test_set)



train_label = train_data[:,0]
test_label = test_data[:,0]


#Isolate 7s
train_label_7= []
for i in train_data:
    #print(i)
    if i[0] == 7.0:
        train_label_7.append(1)
    else:
        train_label_7.append(-1)
        
train_label_7 = np.asarray(train_label_7)


test_label_7= []
for i in test_data:
    #print(i)
    if i[0] == 7.0:
        test_label_7.append(1)
    else:
        test_label_7.append(-1)

test_label_7 = np.asarray(test_label_7)

#%%
#Part 1.1 - 1.2
class Perceptron(object):

    #==========================================#
    # The init method is called when an object #
    # is created. It can be used to initialize #
    # the attributes of the class.             #
    #==========================================#
    def __init__(self, no_inputs, no_nodes =10, max_iterations=20, learning_rate=0.1):
        self.no_nodes = no_nodes
        self.no_inputs = no_inputs
        self.weights = (np.random.random(no_inputs) -1) / no_inputs
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
       

    #=======================================#
    # Prints the details of the perceptron. #
    #=======================================#
    def print_details(self):
        print("No. inputs:\t" + str(self.no_inputs))
        print("Max iterations:\t" + str(self.max_iterations))
        print("Learning rate:\t" + str(self.learning_rate))

    #=========================================#
    # Performs feed-forward prediction on one #
    # set of inputs.                          #
    #=========================================#
    def predict(self, traindata, label):
        traindata = traindata.transpose()
        total_sum = np.dot(traindata, self.weights)
        activation = total_sum
        if activation > 0:
            #print(activation)
            activation = label
            
        else:
            activation = -1
            #print(activation)
        return activation


    #======================================#
    # Trains the perceptron using labelled #
    # training data.                       #
    #======================================#
        
        
        
        
    def train_batch (self, traindata, train_labels):
        total_train_time  = time.time()
        start_time  = time.time()
        
        

        for iter_ in range(self.max_iterations):
            self.prediction = np.array(())
            for x, label in zip(traindata, train_labels):
                     prediction = self.predict(x, label)
                     
                     difference = (label - prediction)
                     if difference != 0:
                         self.weights = self.weights + (self.learning_rate*difference*x)
                      
                         
                     else:
                         #print("perceprton has learnt", label)
                        
                         end_time = time.time()
                  
                     
        print("Learning time is", round(end_time -start_time, 2 ) )  
        print("Total time is",round(total_train_time,2 ))     
    
    def percept_rule_nodes (self, traindata, train_labels):
         total_train_time  = time.time()
         start_time  = time.time()
         for i in range(self.no_nodes):
             for j in range(self.max_iterations):
                 for x , label in zip(traindata, train_labels):
                     prediction = self.predict(x, label)
                     
                     difference = (label - prediction)
                     
                     if difference != 0:
                         self.weights = self.weights + (self.learning_rate*difference*x)
                         #self.weights  = self.weights / len(x)
                     else:
                         print("perceprton has learnt", label)
                         end_time = time.time()
         print("Learning time is", round(end_time -start_time, 2 ) )  
         print("Total time is",round(total_train_time,2 ))  
                
                         

                         
    def sigmoid_predict(self, traindata, train_labels):
        total_sum  = np.dot(traindata, self.weights)
        predicted_label = 1 / (1 + np.exp(-total_sum))
        print("sigmoid output", predicted_label)
        return predicted_label, total_sum
        
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def percept_rule_sigmoid(self, traindata, train_labels):
         total_train_time  = time.time()
         start_time  = time.time()
         
         self.predicts = []    
         for j in range(self.no_nodes):
             for i in range(self.max_iterations):
                 for x, label in zip(traindata, train_labels):
                     
                     label = 1 
                     
                     sigm_sum  = np.dot(x, self.weights)
                     predict  = 1 / (1 + np.exp(-sigm_sum))
                     self.predicts.append(predict)
                     #print("predictions", predict)
                     
                     gradient_vector = (predict*(label-predict)*(1-predict)*label)/len(traindata)

                     if label - predict != 0 :
                         
                         self.weights = self.weights - (gradient_vector * self.learning_rate)
                         #print(self.weights)
                         
                     
                     else:
                         print("sigmoid has learnt perfectly", label)
                         end_time = time.time()
                     
         
         print("Learning time is", round(end_time -start_time, 2 ) )  
         print("Total time is",round(total_train_time,2 ))  
                 
         
                          
         
           
    #=========================================#
    # Tests the prediction on each element of #
    # the testing data. Prints the precision, #
    # recall, and accuracy of the perceptron. #
    #=========================================#
    def test(self, test_set, test_labels, func):
        assert len(test_set) == len(test_labels)
    
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        accuracy = 1
        
        for  data, label in zip(test_set, test_labels):
            prediction = func(data, label)
            #print(label, prediction)
           
            
            
            
            if label == prediction and label == label:
                true_positive += 1
            if  label == prediction  and label == -1:
                true_negative  +=1
            if label!= prediction and prediction == label:
                false_positive += 1
            if label != prediction and prediction == -1:
                false_negative +=1
        
        accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_positive +false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive /(true_positive + false_negative)   
            
        print("Accuracy:\t"+str(accuracy))
        print("Precision:\t"+str(precision))
        print("Recall:\t"+str(recall))
        
        print('true positive', true_positive)
        print('true negative', true_negative)
        print('false positive', false_positive)
        print('false negative', false_negative)
        return true_positive, true_negative, false_positive, false_negative
    
    
    
    def test_sigmoid(self, test_set, test_labels, func):
        assert len(test_set) == len(test_labels)
    
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        
        for  data, label in zip(test_set, test_labels):
            prediction = func(data, label)
            label = 1
            #print(label, prediction[0])
            
            
            if prediction[0] ==1 and label ==1:
                true_positive += 1
            if prediction[0] != 1  and label != 1:
                true_negative  +=1
            if prediction[0] ==1 and label != 1:
                false_positive += 1
            if prediction[0] != 1 and label == 1:
                false_negative +=1
        
        accuracy = (true_positive + true_negative)/(true_negative + true_positive + false_positive +false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive /(true_positive + false_negative)   
            
        print("Accuracy:\t"+str(accuracy))
        print("Precision:\t"+str(precision))
        print("Recall:\t"+str(recall))
        
        print('true positive', true_positive)
        print('true negative', true_negative)
        print('false positive', false_positive)
        print('false negative', false_negative)
      
        return true_positive, true_negative, false_positive, false_negative
    
    def visualization(self, train_set, weights):
        for t in test_set:
            dat = t[1:].reshape((28, 28)) 
            for k in range(28):
                for j in range(28): 
                    if dat[k][j]>0:
                        sys.stdout.write("#") 
                    else:
                        sys.stdout.write(".") 
                    sys.stdout.flush()
                sys.stdout.write("\n") 
            sleep(0.2)
            system('clear')
            
        new_weights  = weights[1:].reshape(28,28)
        plt.imshow(new_weights,cmap=plt.cm.gray, )
        plt.show()   
            
        
    
#%%        
print("Training 7s")
p = Perceptron(28*28+1)
p.print_details()
weights = p.weights
p.train_batch(train_setscaled, train_label_7)
p.test(test_setscaled, test_label_7, p.predict)

#%%
print("Batch Training")

p.train_batch(train_setscaled, train_label)
p.test(test_setscaled, test_label, p.predict)

#%%
print('Nodes Training')
p.percept_rule_nodes(train_setscaled, train_label)
p.test(test_setscaled, test_label, p.predict)

#%%
print("Sigmoid Training")
p.percept_rule_sigmoid(train_setscaled, train_label)
p.test_sigmoid(test_setscaled, test_label, p.sigmoid_predict)
#%%

print('Viusalizations')
p.visualization(test_set, weights)

#%%

    
   
                         









    



