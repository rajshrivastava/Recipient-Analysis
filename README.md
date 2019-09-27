# Recipient-Analysis
Prediction of recipient email id based on the content of the message, subject and its sender. Uses Natural Language Processing. Codes in Python

Steps:
1. Download training dataset at https://drive.google.com/file/d/1efpbJLL6JFmgbtKaYAXCdKpHxKxNQ-x0/view?usp=sharing
2. Download testing dataset at https://drive.google.com/file/d/1GzZkct6ZZFC5VllKi0J5AAADtrm4NqL6/view?usp=sharing
3. Run training.py to train the model.
4. Run testing.py to test the model. 
5. predict_recipient.py : After training, run this code to predict the recipient for user input  data.

Other files description:
generate_vocabs.py  : to generate dictionaries for sender email ids, receiver email ids and word vocabulary.\
generate_training_dataset.py : to prepare lists of inputs and outputs for training.\
sal_parser.py : to extract salutations from emails.\
generate_testing_dataset.py : to prepare lists of inputs and outputs for testing.\
training_data.csv : training data of 439790 samples in the following format: <sender_email_id, receiver_email_id, subject, message>\
testing_data.csv : testing data of 77611 samples in the following format: <sender_email_id, receiver_email_id, subject, message>\


Notes
1.	During testing, the user will be prompted to use salutation filters or not for predicting outputs. This has been kept optional since salutation filters may or may not work depending on the emails in the dataset.
2.	All the trailing past threads in an email message have been trimmed off to train only on the message being sent.

