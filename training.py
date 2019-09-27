#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 10:58:50 2019
@author: Raj Kumar Shrivastava

Recipient Analysis: Training
This code performs the following tasks:
1. generate vocabs.py generates words and email dictionaries for senders and receivers
2. generate_training_dataset.py prepares training data for the Neural Network
3. Training (text -> recipient)
"""
import numpy as np
import pickle
import time
from generate_vocabs import generate_vocabs
import generate_training_dataset
import random
import math
from scipy.sparse import vstack
import tensorflow as tf

class NeuralNet():
    def __init__(self, config):                           #initializing hyperparameters
        self.mini_batch_size = config['mini_batch_size']
        self.epochs = config['epochs']
        
        #Loading word vocabulary dictionary
        self.word_index = pickle.load(open('word_index.txt','rb'))
        self.word_voc_size = len(self.word_index)
        
        #Loading sender email ids vocabulary dictionaries
        self.sender_email_index = pickle.load(open('sender_email_index.txt','rb'))
        self.sender_index_email = pickle.load(open('sender_index_email.txt','rb'))
        self.sender_email_voc_size = len(self.sender_email_index)
        
        #loading receiver email ids vocabulary dictionaries
        self.receiver_email_index = pickle.load(open('receiver_email_index.txt','rb'))
        self.receiver_index_email = pickle.load(open('receiver_index_email.txt','rb'))
        self.receiver_email_voc_size = len(self.receiver_email_index)
        
            
    def train_network(self):
        #Unpickling (loading) the data generated by generate_training_dataset.py
        text_data            = pickle.load(open('text_data.txt','rb'))                  #input
        receiver_data        = pickle.load(open('receiver_data.txt','rb'))              #output 
        
        #tShuffling the data before splitting into training and validaton set
        temp = list(zip(text_data, receiver_data))      
        random.shuffle(temp) 
        text_data, receiver_data = zip(*temp)
        
        length = len(receiver_data)   
        print('->Length of training dataset: ', length)
        
        #Assigning 90% of the data for training, 10% for training.
        #This ratio can vary depending on the availability of the total data
        idx1 = int(length*0.9)  
        text_data_train     = text_data[:idx1]
        receiver_data_train = receiver_data[:idx1]
        
        train_len = len(receiver_data_train)
        
        text_data_valid     = text_data[idx1:]
        receiver_data_valid = receiver_data[idx1:]
        
        valid_len = len(receiver_data_valid)
        
        print("->Length of training data: ", train_len)
        print("->Length of validation data: ", valid_len)
        
        print("->Minibatch size: ", self.mini_batch_size)
        print("->Sender_email vocabulary size: ", self.sender_email_voc_size)
        print("->Receiver_email vocabulary size: ", self.receiver_email_voc_size)
        print("->Word vocabulary size: ", self.word_voc_size)
        
        #graph architecture
        input_len = text_data[0].shape[1]       #length of each tfidf vector
        print("input_len: ", input_len)
        hidden_len1 = 10000             #This is the size of hidden layer.This parameter can be varied.
        
        #Now, we create the architecture of the neural network and define the operations to be performed
        graph=tf.Graph()
        with graph.as_default():
            tf.set_random_seed(17)            
            X = tf.placeholder(tf.float32, [None, input_len], name='X')     #input
            Y = tf.placeholder(tf.int32, [None, 1], name='Y')               #output
            
            self.w1   = tf.Variable(tf.random_uniform([input_len, hidden_len1], -1.0, 1.0),name='w1')
            self.b1   = tf.Variable(tf.truncated_normal([hidden_len1], mean=0, stddev=0.1), name='b1')
            
            self.nce_w   = tf.Variable(tf.truncated_normal([self.receiver_email_voc_size, hidden_len1], stddev=1.0/math.sqrt(hidden_len1)), name='nce_w')
            self.nce_b   = tf.Variable(tf.zeros([self.receiver_email_voc_size]),name='nce_b')
             
            h1_layer = tf.add(tf.matmul(X, self.w1), self.b1)
            h1_layer = tf.nn.relu(h1_layer)
            print("---->h1: ",h1_layer.get_shape())
            
            output_layer =  tf.add(tf.matmul(h1_layer, tf.transpose(self.nce_w)), self.nce_b, name = 'output_layer')        #Used during testing
            
            #Here we are using noise contrastive estimation to find the loss
            loss = tf.reduce_mean(tf.nn.nce_loss(weights = self.nce_w,
                                                 biases = self.nce_b,
                                                 labels = Y,            
                                                 inputs = h1_layer,
                                                 num_sampled = 64,
                                                 num_classes = self.receiver_email_voc_size))
            
            #optimizer = tf.train.AdamOptimizer(0.03).minimize(loss)
            optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
            
       #graph architecture ends

        mini_batches_text_data_valid      = [text_data_valid[k:k+self.mini_batch_size] for k in range(0, valid_len, self.mini_batch_size)]
        mini_batches_receiver_data_valid  = [receiver_data_valid[k:k+self.mini_batch_size] for k in range(0, valid_len, self.mini_batch_size)]
                
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())     #Initializing all the graph variables
            saver = tf.train.Saver()        #For saving the model after training
            
            print("->Training started at ", time.ctime(time.time()) )
            for epo in range(self.epochs):
                
                #Re-shuffling the training data for each epoch to improve generalization
                temp = list(zip(text_data_train, receiver_data_train))
                random.shuffle(temp) 
                text_data_train, receiver_data_train = zip(*temp)
                
                mini_batches_text_data_train      = [text_data_train[k:k+self.mini_batch_size] for k in range(0, train_len, self.mini_batch_size)]
                mini_batches_receiver_data_train  = [receiver_data_train[k:k+self.mini_batch_size] for k in range(0, train_len, self.mini_batch_size)]
                
                #training
                train_loss_sum = 0                
                for mini_count in range(len(mini_batches_receiver_data_train)):
                    batch_x = vstack(mini_batches_text_data_train[mini_count]).toarray()   #training inputs
                    
                    batch_y = mini_batches_receiver_data_train[mini_count]
                    batch_y = np.array(batch_y).reshape(len(batch_y),1)                    #training outputs
                    
                    _, mini_loss = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                    train_loss_sum += mini_loss
                    
                train_loss = train_loss_sum/len(mini_batches_receiver_data_train)
                
                #validation
                valid_loss_sum = 0
                for mini_count in range(len(mini_batches_receiver_data_valid)):
                    batch_x = vstack(mini_batches_text_data_valid[mini_count]).toarray()    #validation inputs
                    
                    batch_y = mini_batches_receiver_data_valid[mini_count]
                    batch_y = np.array(batch_y).reshape(len(batch_y),1)                     #validation outputs
                    
                    mini_loss = sess.run(loss, feed_dict={X:batch_x, Y: batch_y})
                    valid_loss_sum += mini_loss
                    
                valid_loss = valid_loss_sum/len(mini_batches_receiver_data_valid)
                    
                print("\n-->Epoch", epo+1, " completed at ",time.ctime(time.time()) )                
                print("\tTrain loss = {:.2f}\tValid loss = {:.2f}".format(train_loss, valid_loss))
                
            print("->Training completed at ", time.ctime(time.time()))
            
            print('->Saving model...')
            saver.save(sess, './saved_model/my-model', global_step =2)
            
#DRIVER CODE
if __name__=='__main__':
    np.random.seed(7)    
    
    #Feeding the name (with path, if in a different folder) of the training data file in csv format.
    #Each row is in the following format : [sender email id, receiver email ids, subject, message]
    training_data = 'training_data.csv'                 
    
    print("Generating vocabulary dictionaries...", time.ctime(time.time()))
    generate_vocabs(training_data)
    
    print("\nGenerating training dataset...", time.ctime(time.time()))
    generate_training_dataset.generate_dataset(training_data)     

    config={'mini_batch_size':128,'epochs':25}   #final      
    model = NeuralNet(config)
    
    print("\nTraining model...", time.ctime(time.time()))
    model.train_network()
    
    print("Process completed.")