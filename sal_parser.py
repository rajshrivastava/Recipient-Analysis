#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 13:10:24 2019
Salutation Parser

@author: Raj Kumar Shrivastava
"""
def extract_salutation(msg):
    greets = ['dear', 'respected', 'hi', 'hey', 'hello',\
              'dr.', 'mr.', 'mrs.', 'dr', 'mr', 'miss' , 'mrs',\
              'sir','madam','ma\'am']    #Arranged according to priority
    
    msg_length = len(msg)
    salutation = ''

    for greet in greets:
        idx = msg.find(greet)
        if idx != -1 and msg[idx-1].isalnum() == False:               #greet found
            index_after_greet = idx+len(greet)
            if index_after_greet >= msg_length:
                salutation += greet
                return salutation
            
            if msg[index_after_greet] in ['\n', ',', '!', ':']:
                salutation += greet
                return salutation
            
            if msg[index_after_greet] == ' ':                #if match found, append salutation
                salutation += greet
                msg = msg[index_after_greet:]
                msg_length = len(msg)
                break               
    
    if idx != -1:                         #if match found, find next salutation word
        for greet in greets:
            idx = msg.find(greet)
            if idx != -1:               #greet found                                        
                index_after_greet = idx+len(greet)
                if index_after_greet >= msg_length:
                    salutation += ' ' + greet
                    return salutation
                
                if msg[index_after_greet] in ['\n', ',', '!', ':']:
                    salutation += ' ' + greet
                    return salutation
                
                if msg[index_after_greet] == ' ':                #if match found, append salutation
                    salutation += ' ' + greet
                    msg = msg[index_after_greet:]
                    msg_length = len(msg)
                    break
         
        puncs = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' + '\n'    #find name followed by salutation
        idx = 1
        whitespaces_count = 0
        whitespaces_lim = 2   #can tolerate upto 3 words for name
        while (idx < msg_length and whitespaces_count <= whitespaces_lim and msg[idx] not in puncs):
            if msg[idx] == ' ': whitespaces_count += 1
            idx += 1
        
        if idx >= msg_length or whitespaces_count > whitespaces_lim:
            return salutation
        
        if msg[idx] in puncs:
            salutation += msg[:idx]             #append salutation, from beginning (including whtespace) till end of name
         
    else:                                                       #find name without any salutation
        puncs = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' + '\n'    
        idx = 0
        whitespaces_count = 0
        whitespaces_lim = 1   #can tolerate upto 2 words for name (Assumption: Only first name is used for messages beginning directly with receiver's name)
        while (idx < msg_length and whitespaces_count <= whitespaces_lim and msg[idx] not in puncs):
            if msg[idx] == ' ': whitespaces_count += 1
            idx += 1
    
        if idx >= msg_length or whitespaces_count > whitespaces_lim:
            return salutation
        
        if msg[idx] in puncs:
            salutation += msg[:idx]             #append name
     
    return salutation