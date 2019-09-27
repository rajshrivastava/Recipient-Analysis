#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 10:58:50 2019
@author: Raj Kumar Shrivastava

Word and email id vocabularies generator.
"""
import pandas as pd
import pickle
import time
import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter 

class NeuralNet():
    
    def __init__(self, config):                           
        self.top_n_words = config['top_n_words']        #Threshold for the count of word
       
    def create_vocabs(self, data):
        #list of insignificant words to be eliminated from the email message
        stop_words=['k','m','t','d','e','f','g','h','i','u','r','I','im',\
                    'ourselves', 'hers', 'between', 'yourself', 'again', \
                    'there', 'about', 'once', 'during', 'out', 'very', \
                    'having', 'with', 'they', 'own', 'an', 'be', 'some', \
                    'for', 'do', 'its', 'yours', 'such', 'into', 'of', \
                    'most', 'itself', 'other', 'off', 'is', 's', 'am', \
                    'or', 'who', 'as', 'from', 'him', 'each', 'the', \
                    'themselves', 'until', 'below', 'are', 'we', 'these', \
                    'your', 'his', 'through', 'don', 'nor', 'me', 'were', \
                    'her', 'more', 'himself', 'this', 'should', 'our', \
                    'their', 'while', 'above', 'both', 'to', 'ours', 'had', \
                    'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',\
                    'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',\
                    'yourselves', 'then', 'that', 'because', 'what', 'over', \
                    'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you',\
                    'herself', 'has', 'just', 'where', 'too', 'only', 'myself',\
                    'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',\
                    'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',\
                    'how', 'further', 'was', 'here', 'than'];
        
        porter = PorterStemmer()   #For stemming words to their root words
        
        #Loading the columns of the dataset in seperate variables
        sender_col      = data.iloc[:, 0]
        receivers_col   = data.iloc[:, 1]        
        subject_col     = data.iloc[:, 2] 
        message_col     = data.iloc[:, 3]
        
        sender_email_id_counts = {}      #dictionary to store all sender_email ids and their counts
        receiver_email_id_counts = {}    #dictionary to store all receiver_email ids and their counts
        word_counts = {}                 #dictionary to store all words and their counts
                
        for i in range(len(data)):
            if i%100000 == 0: print(i, time.ctime(time.time()))
            
            #Creating sender email ids dictionary
            if type(sender_col[i]) != float:                                    
                sender= re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', sender_col[i])
                if len(sender)!=0:
                    sender = sender[0]
                    if sender in sender_email_id_counts:
                        sender_email_id_counts[sender] += 1
                    else:
                        sender_email_id_counts[sender] = 1

            #Creating receiver email ids dictionary
            if type(receivers_col[i]) != float:                              
                recvrs = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', receivers_col[i])   
                if len(recvrs)!=0:
                    for recvr in recvrs:
                        if recvr in receiver_email_id_counts:
                            receiver_email_id_counts[recvr] += 1     #for existing email id
                        else:
                            receiver_email_id_counts[recvr] = 1      #for new email id
                        
            #checking for readable subject
            if(type(subject_col[i])!=float): subject = subject_col[i]           
            else: subject = ''
            
            #checking for readable message
            if(type(message_col[i])!=float): message = message_col[i]           
            else: message = ''
            
            #concatenating subject and the message
            text = subject + '\n' + message            
            text = text.lower()
            text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) #removing punctuations
            text = re.sub(r'(.)\1+', r'\1\1', text)     #remove repeating characters (for eg., yessss -> yes)

            stemmed_words = [porter.stem(word) for word in word_tokenize(text)]         #reducing words to their root words
           
            for word in stemmed_words:
                if (word not in stop_words) and word.isdigit() == False:                                        
                    if word in word_counts:
                        word_counts[word] += 1   #for existing word
                    else:
                        word_counts[word] = 1     #for new word
        #for loop ends

        self.sender_email_voc_size = len(sender_email_id_counts)
        print("Sender email_id vocabulary size = ",self.sender_email_voc_size)
        
        self.sender_email_index = dict((email, i) for i, email in enumerate(sender_email_id_counts))        #sender_email -> index
        self.sender_index_email = dict((i, email) for i, email in enumerate(sender_email_id_counts))        #index -> sender_email
        
        self.receiver_email_voc_size = len(receiver_email_id_counts)
        print("Receiver email_id vocabulary size = ",self.receiver_email_voc_size)
        
        self.receiver_email_index = dict((email, i) for i, email in enumerate(receiver_email_id_counts))    #receiverer_email -> index
        self.receiver_index_email = dict((i, email) for i, email in enumerate(receiver_email_id_counts))    #index -> receiver_email
        
        #Assuming that words which have occured rarely or insignificant or noisy words,
        #we eliminate the less frequent words
        k = Counter(word_counts)
        high = k.most_common(self.top_n_words)
        word_counts = dict(high)
        
        self.word_voc_size = len(word_counts)
        print("Word vocabulary size = ",self.word_voc_size)
        
        self.word_index = dict((word, i) for i, word in enumerate(word_counts))     #word -> index
        self.index_word = dict((i, word) for i, word in enumerate(word_counts))     #index -> word
        
        #Saving all the dictionaries to the disk.
        #Will be used for training and testing and creating their respective datasets.
        out_file=open('word_index.txt','wb')
        pickle.dump(self.word_index,out_file)
        out_file.close()    
        
        out_file=open('index_word.txt','wb')
        pickle.dump(self.index_word,out_file)
        out_file.close()
        
        out_file=open('sender_email_index.txt','wb')
        pickle.dump(self.sender_email_index,out_file)
        out_file.close()    
        
        out_file=open('sender_index_email.txt','wb')
        pickle.dump(self.sender_index_email,out_file)
        out_file.close()
        
        out_file=open('receiver_email_index.txt','wb')
        pickle.dump(self.receiver_email_index,out_file)
        out_file.close()    
        
        out_file=open('receiver_index_email.txt','wb')
        pickle.dump(self.receiver_index_email,out_file)
        out_file.close()
        
        #Storing the vocabulary words to disk.
        #The same vocabulary will be used for training and testing to create Tfidfs
        vectorizer = CountVectorizer(decode_error="replace")
        vec_train = vectorizer.fit_transform(word_counts)
        pickle.dump(vectorizer.vocabulary_,open("tfidf_features.pkl","wb"))


#DRIVER CODE
def generate_vocabs(data_file_name):
    
    config={'top_n_words':20000} 
     
    model = NeuralNet(config)
    
    print("Reading training data...")
    data=pd.read_csv(data_file_name)   
    
    print("Length of dataset: ", len(data))
    
    print("Generating corpus...", time.ctime(time.time()))    
    model.create_vocabs(data)   
    
    print("Vocabulary generation completed at ", time.ctime(time.time()))