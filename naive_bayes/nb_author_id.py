#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB() 
t0 = time() #set timer
clf.fit(features_train, labels_train) #training
print "training time:", round(time()-t0, 3), "s" #end timer
t1 = time() #set second timer
print(clf.score(features_test,labels_test)) # accuracy =points classified correctly/total points
print "testing time:", round(time()-t1, 3), "s" #end second timer
#########################################################


