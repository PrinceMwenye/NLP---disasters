# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:18:50 2020

@author: Prince
"""

import pandas as pd 
import numpy as np
import re
import nltk #for stop words e.g the, a , but, irrelvant words
nltk.download('stopwords') 
from nltk.corpus import stopwords #corpus has stop words
from nltk.stem.porter import PorterStemmer #to apply stemming into our views e.g loved to love because means the sames

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test['target'] = 'NA'

full = pd.concat([train, test], axis = 0, keys = ['train', 'test'])
full = full.fillna('missing')

full['text'] = full['text'].map(lambda i: str(i))

#full = train.drop(['location','id'], axis = 1)
full.isna().sum()

#replace missing keyword and missing values

corpus = []
for i in range(0,10876):
    review = re.sub('[^a-zA-Z]', ' ', full['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer #some words still not relevant so we need to take most frequent word
cv = CountVectorizer(max_features= 15000) #remove some words, hence 18K most freq words
x = cv.fit_transform(corpus).toarray()
y = full['target'].values

len(x[0]) #number of words

#split data into test/train sets

x_train = x[:7613,].astype('int')
x_test = x[7613:,].astype('int')
y_train = y[:7613,].astype('int')
#y_test = y[7613:, -1].astype('int') #algorithm wants to read array in int

'''from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train) 


y_pred_1 = classifier.predict(x_test) #54 %

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = kernel_classifier, scoring = 'f1',
                             X = x_train, y = y_train, cv = 3)
accuracies.mean()
accuracies.std()'''


#naive bayes = 61% average
#K nearest = 0 average
#SVC = 0 average
#deep learning
#xgboos = 53%


#NLP feature engineering practices

 

# Initializing the ANN


import tensorflow as tf
tf.__version__


# Adding the input layer and the first hidden layer
#activation function in a fully connected neural netwrok must be rectifier "relu
from keras.wrappers.scikit_learn import KerasClassifier #for cross validation
from sklearn.model_selection import GridSearchCV

def kerasclassifier():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=20, activation='relu'))   #DENSE class? #hyperparameters,  units = how many neurons
    ann.add(tf.keras.layers.Dense(units=18, activation='relu'))  #relu because of linear combination of inputs
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #output should be one because its binary. If more than 1, say a,b,c we need 3
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #for binary classification, always binary_crossentropy, #for none,
    return ann


model = KerasClassifier(build_fn = kerasclassifier, epochs = 100, batch_size = 100)

# Training the ANN on the Training set
model.fit(x_train, y_train)


#cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 4)
accuracies.mean()
accuracies.std()

#gridsearch 
'''
from sklearn.model_selection import GridSearchCV
parameters = [{'batch_size': [25,32,40],
               'epochs': [100, 150],
               }]
   
gs = GridSearchCV(estimator = model,
                  param_grid = parameters,
                  scoring = 'accuracy',
                  cv = 6,
                  n_jobs = -1)

'''

#predictions
y_pred = model.predict(x_test)
y_pred = y_pred[:, 0]



y_pred = pd.Series(y_pred)

pred_values = lambda i: 0 if i < 0.5 else 1
y_pred = y_pred.map(pred_values)
final = pd.concat([test, y_pred], axis = 1)


final.to_csv('final.csv', index = False)

import getpass
print(getpass.getuser())

