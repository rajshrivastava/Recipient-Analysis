#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:11:18 2019
@author: Raj Kumar Shrivastava

Generate dataset for training data.
Dataset file in csv format is fed to this method.
It returns sender_data, text_data, receiver_data, salutation_data.
It also creates a sender dictionary which stores the past recipients
along with the respective salutations used in those mails, if any.
"""
import pandas as pd
import re
from sal_parser import extract_salutation
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time

#method to create a dictionary of senders with their respecive recipients+salutations
def sender_receivers(sender_data, salutation_data, receiver_data):
    sender_dic  = {}
    for i in range(len(sender_data)):
        sender_index   = sender_data[i]
        salutation = salutation_data[i]
        receiver_index = receiver_data[i]
        if sender_index not in sender_dic:
            sender_dic[sender_index] = {(salutation, receiver_index)}
        else:
            sender_dic[sender_index].add((salutation, receiver_index))
    
    sender_dic_modified = {}
    for key, tups in sender_dic.items():
        sal_list = []
        rec_list = []
        for tup in tups:
            sal, rec = tup
            sal_list.append(sal)
            rec_list.append(rec)
        sender_dic_modified[key] = [sal_list, rec_list]
    return sender_dic_modified

def generate_dataset(data_file_name):            #training pairs generation
    print("Reading training data")
    data = pd.read_csv(data_file_name)       
    print("Raw length of training dataset: ", len(data))
    
    #Loading dictionaries
    sender_email_index = pickle.load(open('sender_email_index.txt','rb'))             #generated from generate_vocabs.py
    receiver_email_index = pickle.load(open('receiver_email_index.txt','rb'))         #generated from generate_vocabs.py
    word_index = pickle.load(open('word_index.txt','rb'))                             #generated from generate_vocabs.py
    
    porter = PorterStemmer()   #For stemming words to their root words
    
    #Loading the columns of the dataset in seperate variables    
    sender_col      = data.iloc[:, 0]
    receivers_col   = data.iloc[:, 1]        
    subject_col     = data.iloc[:, 2] 
    message_col     = data.iloc[:, 3]       
    
    groups = []
    for i in range(len(data)):     
        
        #filtering sender email id
        if type(sender_col[i]) == float:
            continue
        sender= re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', sender_col[i])
        if len(sender) == 0:
            continue
        sender = sender[0]
        if sender not in sender_email_index:
            continue
       
        #Checking for readable subject and message
        if(type(subject_col[i])!=float): subject = subject_col[i]           #subject
        else: subject = ''
        
        if(type(message_col[i])!=float): message = message_col[i]           #message
        else: message = ''
        
        #Extracting salutation, if any, from the message
        salutation = extract_salutation(message)
        
        #concatenating subject and the message
        text = subject + '\n' + message                                     #sub+msg
        text = text.lower()
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) #removing punctuations
        text = re.sub(r'(.)\1+', r'\1\1', text)     #remove repeating characters (for eg., yessss -> yes)

        stemmed_words = [porter.stem(word) for word in word_tokenize(text)]     #reducing words to their root words
       
        filtered_words = []            
        for word in stemmed_words:
            if word in word_index:
                filtered_words.append(word)
        
        if len(filtered_words) == 0:
            continue
        else:
            filtered_text = ' '.join(filtered_words)   #creating a string after filtering the words
   
        #filtering receiver email ids
        receivers = []
        if type(receivers_col[i]) == float:
            continue
        recvs= re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', receivers_col[i])
        if len(recvs) == 0:
            continue
        for recv in recvs:
            if recv in receiver_email_index:
                receivers.append(recv)
        if len(receivers) == 0:
            continue
      
        groups.append([sender, filtered_text, receivers, salutation])
        
    print('->vectorizing', time.ctime(time.time()))
    texts    = [row[1] for row in groups]
    
    #Creating tfidfs of the messages
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("tfidf_features.pkl", "rb")))
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(texts))
    
    sender_data = []
    text_data = []
    receiver_data = []
    salutation_data = []
    
    for i in range(len(groups)):            
        #for recvr in groups[i][2]:
        sender_data.append(sender_email_index[groups[i][0]])
        text_data.append(tfidf[i])
        receiver_data.append(receiver_email_index[ groups[i][2][0] ])           #Storing only the first recipient email id for meainingful training
        salutation_data.append(groups[i][3])

    sender_dic = sender_receivers(sender_data, salutation_data, receiver_data)
    
    with open("sender_data.txt", "wb") as fp:       #Pickling
        pickle.dump(sender_data, fp)
        
    with open("text_data.txt", "wb") as fp:         #Pickling
        pickle.dump(text_data, fp)
        
    with open("receiver_data.txt", "wb") as fp:     #Pickling
        pickle.dump(receiver_data, fp)
        
    with open("salutation_data.txt", "wb") as fp:   #Pickling
        pickle.dump(salutation_data, fp)
        
    out_file=open('sender_dic.txt','wb')            #Pickling
    pickle.dump(sender_dic, out_file)
    out_file.close()