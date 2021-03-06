#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

from sklearn import svm
# clf=svm.SVC(kernel='linear')
clf=svm.SVC(kernel='rbf', C=10000.0) # usarlo solo con una small sample, is too slow
t0 = time() #set timer

#reduce de dataset
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
#

clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s" #end timer
t1 = time() #set timer
print(clf.score(features_test,labels_test))
print "training time:", round(time()-t1, 3), "s" #end timer

pred = clf.predict(features_test)
print(pred[10], pred[26], pred[50])
print(sum(pred))
#########################################################


