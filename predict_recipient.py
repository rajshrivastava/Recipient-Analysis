#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:32:13 2019
Recipient Analysis - Testing
@author: Raj Kumar Shrivastava
"""
import pickle
import tensorflow as tf
import re
from sal_parser import extract_salutation
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class NeuralNet():
    def __init__(self):                           #initializing hyperparameters        
        self.word_index = pickle.load(open('word_index.txt','rb'))
        
        #Loading sender dictionaries
        self.sender_email_index = pickle.load(open('sender_email_index.txt','rb'))
        self.sender_index_email = pickle.load(open('sender_index_email.txt','rb'))
        self.sender_email_voc_size = len(self.sender_email_index)
        
        #Loading receiver dictionaries
        self.receiver_email_index = pickle.load(open('receiver_email_index.txt','rb'))
        self.receiver_index_email = pickle.load(open('receiver_index_email.txt','rb'))
        self.receiver_email_voc_size = len(self.receiver_email_index)
        
    def generate_data(self, sender, subject, message):
        flag = 0        #raise flag if any invalid data is entered
        salutation = ''
        porter = PorterStemmer()   #For stemming words to their root words
        
        #filtering sender email id
        if type(sender) == float:
            flag = 'invalid sender'
            return flag, sender, subject, message, salutation
        
        sender = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', sender)
        if len(sender) == 0:
            flag = 'invalid sender'
            return flag, sender, subject, message, salutation
        
        sender = sender[0]
        if sender not in self.sender_email_index:
            flag = 'new sender'
            return flag, sender, subject, message, salutation
        
        #Checking for readable subject and message
        if(type(subject) == float):
            subject = ''          
        
        if(type(message) == float):
            message = ''           
        
        #Extracting salutation, if any, from the message
        salutation = extract_salutation(message)
        
        #concatenating subject and the message
        text = subject + '\n' + message                                     #sub+msg
        text = text.lower()
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) #removing punctuations
        text = re.sub(r'(.)\1+', r'\1\1', text)     #remove repeating characters (for eg., yessss -> yes)

        stemmed_words = [porter.stem(word) for word in word_tokenize(text)]         #reducing words to their root words
       
        filtered_words = []            
        for word in stemmed_words:
            if word in self.word_index:
                filtered_words.append(word)
        
        if len(filtered_words) == 0:
            flag = 'Insufficient message length'
            return flag, sender, subject, message, salutation
        else:
            filtered_text = ' '.join(filtered_words)            #creating a string after filtering the words
      
        texts    = [filtered_text]
        
        #Creating tfidfs of the message
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("tfidf_features.pkl", "rb")))
        tfidf = transformer.fit_transform(loaded_vec.fit_transform(texts))
        
        return flag, self.sender_email_index[sender], subject, tfidf, salutation

    def predict(self, tfidf):            #training pairs generation
        with tf.Session() as sess:
            #Reloading the trained model
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('./saved_model/my-model-2.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./saved_model'))
            graph = tf.get_default_graph()
            X= graph.get_tensor_by_name("X:0")
            output_layer = graph.get_tensor_by_name('output_layer:0')
            print("Model restored.")
            
            test_x = tfidf.toarray()                
            outputs = sess.run(output_layer, feed_dict={X:test_x})
            return outputs
    
    def filter_predictions(self, prediction_vector, sender_index, salutation):
        self.sender_dic = pickle.load(open('sender_dic.txt','rb'))
        
        idxs =  prediction_vector.argsort()[:][::-1]
                                       
        sal_list = self.sender_dic[sender_index][0]      #list of salutation used for recipients
        rec_list = self.sender_dic[sender_index][1]      #list of corresponding recipients (indexes)
        
        #creating ordered and (later) filtered lists based on the beginning characters of the recipient email id and the the past interaction
        sal_list_filtered = []
        rec_list_filtered = []
        
        for idx in idxs:
            if idx in rec_list:
                pos = rec_list.index(idx)
                rec_list_filtered.append(idx)
                sal_list_filtered.append(sal_list[pos])
        
        #for no keys pressed yet
        print("\nPREDICTED RECIPIENT: ", self.receiver_index_email[rec_list_filtered[0]] )
        
        #after pressing key(s)
        keys_entered = ''
        while True:
            ch = input('Enter recipient characters: {}'.format(keys_entered))
            if ch == '':
                return keys_entered         #when the entire recipient email id has been entered
            #else
            sal_list_refiltered = []
            rec_list_refiltered = []
            beginning_with = keys_entered + ch
            for rec_idx in rec_list_filtered:
                if self.receiver_index_email[rec_idx].startswith(beginning_with):
                    pos = rec_list_filtered.index(rec_idx)
                    rec_list_refiltered.append(rec_idx)
                    sal_list_refiltered.append(sal_list_filtered[pos])
            
            if len(rec_list_refiltered) == 0:
                print("No email found. Press another key")
                continue
            #Salutation filtering
            #It may or may not give improved results depending on the dataset of emails.
            '''
            if len(salutation)>=2:              
                if salutation in sal_list_refiltered:       #Filtering based on exact salutation matching (hard filter)
                    pos = sal_list_refiltered.index(salutation)
                    predicted_rec = self.receiver_email_index[rec_list_refiltered[pos] ]   
                    print("\nPREDICTED RECIPIENT: ", predicted_rec)
                    continue                          
            '''
            predicted_rec =  self.receiver_index_email[ rec_list_refiltered[0] ]
            print("\nPREDICTED RECIPIENT: ", predicted_rec)
            keys_entered += ch
            sal_list_filtered = sal_list_refiltered.copy()
            rec_list_filtered = rec_list_refiltered.copy()
            
            
    def warn(self, entered_rec, sender):
        
        if entered_rec in self.receiver_email_index:
            idx = self.receiver_email_index[entered_rec]
            if idx not in self.sender_dic[sender][1]:
                print("WARNING: Potenitally incorrect recipient")
        else:
            print("WARNING: Potenitally incorrect recipient")
#DRIVER CODE
if __name__=='__main__':
    model = NeuralNet() 
    sender = input("Sender email id: ")
    subject = input("Subject: ")
    message = input("Message: ")
    
    #Following are sample inputs
    #'''
    sender = 'ted.murphy@enron.com'
    subject = 'Schedule'
    message =  'Hello, The meeting is scheduled on Wednesday next week. Thank you'
    #'''
    flag, sender, subject, message, salutation = model.generate_data(sender, subject, message)
    
    if flag == 0:
        output_layer = model.predict(message)
        entered_receiver = model.filter_predictions(output_layer[0], sender, salutation)    
        #model.warn(entered_receiver, sender)
    else:
        print(flag)