#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:32:13 2019
@author: Raj Kumar Shrivastava

Recipient Analysis: Testing
This code performs the following tasks:
2. generate_testing_dataset.py prepares testing data for the Neural Network
3. Testing (Neural network + filters)
"""
import numpy as np
import pickle
import time
import generate_testing_dataset
import tensorflow as tf
from scipy.sparse import vstack

class NeuralNet():
    def __init__(self):                           #initializing hyperparameters
        self.mini_batch_size = 512  #can be reduced if memory error occurs
        
        self.word_index = pickle.load(open('word_index.txt','rb'))
        
        #Loading sender dictionaries
        self.sender_email_index = pickle.load(open('sender_email_index.txt','rb'))
        self.sender_index_email = pickle.load(open('sender_index_email.txt','rb'))
        self.sender_email_voc_size = len(self.sender_email_index)
        
        #Loading receiver dictionaries
        self.receiver_email_index = pickle.load(open('receiver_email_index.txt','rb'))
        self.receiver_index_email = pickle.load(open('receiver_index_email.txt','rb'))
        self.receiver_email_voc_size = len(self.receiver_email_index)
    
    def test_network(self, sal_flag):
        
        def true_count(outputs, senders, true_receivers, true_salutations):
            correct_pred_count = 0          #for correct predictions
            new_receiver_count = 0          #for any new receiver encountered
            new_sender_count   = 0          #for any new sender enountered
                
            for i in range(len(outputs)):                
                sender_index = senders[i]                               
                if sender_index not in sender_dic:          #checking for new_sender
                    new_sender_count += 1
                    continue 
                
                true_receiver_index = true_receivers[i]                
                
                #true_chars = ''                                                    #Accuracy for no given character of true recipient email_id
                #true_chars = self.receiver_index_email[true_receiver_index][:1]    #Accuracy for 1st given character of true recipient email_id
                true_chars = self.receiver_index_email[true_receiver_index][:2]     #Accuracy for 1st two given characters of true recipient email_id
                
                prediction_vector = outputs[i]
                idxs =  prediction_vector.argsort()[:][::-1]
                                               
                sal_list = sender_dic[sender_index][0]      #list of salutation used for recipients
                rec_list = sender_dic[sender_index][1]      #list of corresponding recipients (indexes)
                
                #creating filtered lists based on the beginning characters of the recipient email id and the the past interaction
                sal_list_filtered = []
                rec_list_filtered = []
                
                #l=int(len(idxs)*0.7)            #only considering the top 70% ranked predictions
                #idxs = idxs[:l]
                
                for idx in idxs:
                    if self.receiver_index_email[idx].startswith(true_chars) and idx in rec_list:
                        pos = rec_list.index(idx)
                        rec_list_filtered.append(idx)
                        sal_list_filtered.append(sal_list[pos])
                           
                if true_receiver_index not in rec_list_filtered:    #checking for new_receiver
                    new_receiver_count += 1
                    continue
                
                if len(rec_list_filtered) == 1:
                    if rec_list_filtered[0] == true_receiver_index:correct_pred_count += 1
                    continue
                
                #Salutation filtering
                #It may or may not give improved results depending on the dataset of emails.
                if sal_flag==1:         #if salutation is to be checked
                    true_salutation   = true_salutations[i]  
                    if len(true_salutation)>=2:              
                        if true_salutation in sal_list_filtered:       #Filtering based on exact salutation matching (hard filter)
                            pos = sal_list_filtered.index(true_salutation)
                            if rec_list_filtered[pos] == true_receiver_index: correct_pred_count += 1
                            continue
                
                #Salutation not present/Salutation not matched. Predicting the top rec_index as correct.
                if rec_list_filtered[0] == true_receiver_index:
                    correct_pred_count += 1
                
            return correct_pred_count, new_receiver_count, new_sender_count
         
        #Loading testing data_generated from generate_testing_data.py    
        sender_data = pickle.load(open('sender_data_test.txt','rb'))
        text_data = pickle.load(open('text_data_test.txt','rb'))
        receiver_data = pickle.load(open('receiver_data_test.txt','rb'))
        salutation_data     = pickle.load(open('salutation_data_test.txt','rb'))       #Unpickling
        sender_dic = pickle.load(open('sender_dic.txt','rb'))
        
        test_len = len(sender_data)       
        print('Length of testing dataset: ', test_len)
        
        #creating mini-batches for testing in batches at a time
        mini_batches_sender_data    = [sender_data[k:k+self.mini_batch_size] for k in range(0, test_len, self.mini_batch_size)]
        mini_batches_text_data      = [text_data[k:k+self.mini_batch_size] for k in range(0, test_len, self.mini_batch_size)]
        mini_batches_receiver_data  = [receiver_data[k:k+self.mini_batch_size] for k in range(0, test_len, self.mini_batch_size)]
        mini_batches_salutation_data= [salutation_data[k:k+self.mini_batch_size] for k in range(0, test_len, self.mini_batch_size)]
        
        with tf.Session() as sess:
            #Reloading the trained model
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('./saved_model/my-model-2.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./saved_model'))
            graph = tf.get_default_graph()
            X= graph.get_tensor_by_name("X:0")
            output_layer = graph.get_tensor_by_name('output_layer:0')
            print("Model restored.")
            
            correct_pred_count_sum = 0
            new_receiver_count_sum = 0
            new_sender_count_sum  =  0
            
            for mini_count in range(len(mini_batches_sender_data)):
                print("mini_count = {} at {}".format(mini_count, time.ctime(time.time()) ) )
                test_x = vstack(mini_batches_text_data[mini_count]).toarray()                
                outputs = sess.run(output_layer, feed_dict={X:test_x})
                
                correct_pred_count, new_receiver_count, new_sender_count = true_count(outputs, mini_batches_sender_data[mini_count],\
                                                                                      mini_batches_receiver_data[mini_count],\
                                                                                      mini_batches_salutation_data[mini_count])
                correct_pred_count_sum += correct_pred_count
                new_receiver_count_sum += new_receiver_count
                new_sender_count_sum += new_sender_count
            
            #Test samples where a new sender or receiver is encountered are not considered
            actual_test_len = test_len - new_receiver_count_sum - new_sender_count_sum
            acc = 100*correct_pred_count_sum/actual_test_len
                
            print("\nTest Accuracy: {:.1f}".format(acc))

#DRIVER CODE
if __name__=='__main__':
    np.random.seed(0)   
    model = NeuralNet()
    
    #Feeding the name (with path, if in a different folder) of the testing data file in csv format.
    #Each row is in the following format : [sender email id, receiver email ids, subject, message]
    testing_data_file = 'testing_data.csv'    
    
    sal_flag = input('Press 1 if salutation filters are to be included, else 0: ')
    print("Generating testing dataset...", time.ctime(time.time()))
    generate_testing_dataset.generate_dataset(testing_data_file)
      
    print("Testing model...", time.ctime(time.time()))
    model.test_network(sal_flag)
    
    print("Process completed.", time.ctime(time.time()))