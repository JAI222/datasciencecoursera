# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:39:04 2020

@author: junti
"""


import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import NullFormatter
import pandas as pd
from sklearn import preprocessing


df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')
#print(df.head())

#Convert to date time object
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
print(df.head())

#data visualization and preprocessing
#print(df['loan_status'].value_counts())

import seaborn as sns

#preprocessing
df['dayofweek'] = df['effective_date'].dt.dayofweek
#set a threshold
df['weekend'] = df['dayofweek'].apply(lambda x:1 if (x>3) else 0)
#print(df.head())

#convert categorical features to numerical
#print(df.groupby(['Gender'])['loan_status'].value_counts(normalize = True))
df['Gender'].replace(to_replace=['male', 'female'], value = [0,1], inplace = True)
#print(df.head())

df.groupby(['education'])['loan_status'].value_counts(normalize = True)

#use one hot encoding technique to conver categorical variable to binary variables
Feature = df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
Feature = pd.concat([Feature, pd.get_dummies(df['education'])], axis = 1)
Feature.drop(['Master or Above'], axis = 1, inplace = True)
#print(Feature.head())

#define feature sets, X
X = Feature
#print(X[0:5])
y = df['loan_status'].values
#print(y[0:5])

#Normalize data
X = preprocessing.StandardScaler().fit(X).transform(X)



#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
print('Train set:' , x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = []
for n in range(1, Ks):
    #train model and predict
    neigh = KNeighborsClassifier(n_neighbors =n).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
#print(mean_acc)

plt.plot(range(1,Ks), mean_acc, 'g')
plt.fill_between(range(1,Ks), mean_acc -1*std_acc, mean_acc +1*std_acc, alpha = 0.1)
plt.legend(('Accuracy', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of nabors(K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

k = 7
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
neigh


#Modeling Decision Tree
from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier(criterion = 'entropy', max_depth =4)
#print(Tree)
#fit the data
Tree.fit(X, y)

#Modeling SVM
from sklearn import svm
clf = svm.SVC(kernel = 'rbf')
clf.fit(X, y)

#Modeling LR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver ='liblinear').fit(X, y)


#Dealing with test set
test_df = pd.read_csv('loan_test.csv')
print(test_df.head())


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])

test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x:1 if (x>3) else 0)

test_df['Gender'].replace(to_replace=['male', 'female'], value = [0,1], inplace = True)
test_df.groupby(['education'])['loan_status'].value_counts(normalize = True)

Feature_test = test_df[['Principal', 'terms', 'age', 'Gender', 'weekend']]
Feature_test = pd.concat([Feature_test, pd.get_dummies(test_df['education'])], axis = 1)
Feature_test.drop(['Master or Above'], axis = 1, inplace = True)

X_test = Feature_test

X_test =preprocessing.StandardScaler().fit(X_test).transform(X_test)
y_test = test_df['loan_status'].values



#predicting KNN
yhat = neigh.predict(X_test)

from sklearn.metrics import f1_score
F1_KNN = f1_score(y_test, yhat, average = 'weighted')
print('F1_score_KNN', F1_KNN )
    
from sklearn.metrics import jaccard_similarity_score
Jaccard_KNN = jaccard_similarity_score(y_test, yhat)
print('jaccard_score_KNN', Jaccard_KNN)


#Predicting Decision Tree
predTree = Tree.predict(X_test)
F1_Tree = f1_score(y_test, predTree, average= 'weighted')
Jaccard_Tree = jaccard_similarity_score(y_test, predTree)
print('F1_score_Tree', F1_Tree )
print('jaccard_score_Tree', Jaccard_Tree)

#Predicting SVM
yhat_SVM = clf.predict(X_test)
F1_SVM = f1_score(y_test, yhat_SVM, average = 'weighted')
Jaccard_SVM = jaccard_similarity_score(y_test, yhat_SVM)
print('F1_score_SVM_rbf', F1_SVM)
print('jaccard_score_SVM_rbf', Jaccard_SVM)

#Predicting LR
yhat_LR = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

#Evaluating
F1_LR =  f1_score(y_test, yhat_LR, average = 'weighted')
Jaccard_LR = jaccard_similarity_score(y_test, yhat_LR)
print('F1_score_LR',F1_LR)
print('jaccard_score_LR', Jaccard_LR)

from sklearn.metrics import log_loss
Logloss = log_loss(y_test, yhat_prob)
print('Log_loss_LR', Logloss)



result = {'Algorithm':['KNN', 'DecisionTree', 'SVM', 'LogisticRegreesion'],
          'Jaccard': [Jaccard_KNN, Jaccard_Tree, Jaccard_SVM, Jaccard_LR],
          'F1-score': [F1_KNN, F1_Tree, F1_SVM, F1_LR],
          'LogLoss': ['NA', 'NA', 'NA', Logloss]}

result_df = pd.DataFrame(result, columns = ['Algorithm', 'Jaccard', 'F1-score', 'LogLoss'],index = None)
r_csv = result_df.to_csv(header = True, index = True)
print(result_df)



