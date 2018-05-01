import time
start_time = time.time()

import pandas
import sys
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import sys

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


from sklearn import preprocessing
nm = preprocessing.Normalizer()
normalized_X_train = nm.fit_transform(X_train)
normalized_X_test = nm.transform(X_test)
print(normalized_X_train)

import keras
y_train_encoded = keras.utils.to_categorical(y_train, 10)
y_test_encoded = keras.utils.to_categorical(y_test, 10)

# sys.exit()
########################################################
# from sklearn.model_selection import cross_val_predict
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm

# knn = svm.SVC()
# # knn = KNeighborsClassifier(n_neighbors=5)
# # knn = RandomForestClassifier(n_estimators=50,max_depth=32,n_jobs=-1)
# predictions = cross_val_predict(knn, X_train, y_train, cv=10) 
# # knn.fit(X_train,y_train)
# predictions = knn.predict(X_test)
##########################################################
#####################KERAS-Training#######################
# MODEL
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=784))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.add(Activation('relu'))

# COMPILATION
from keras import optimizers
opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(normalized_X_train, y_train_encoded, epochs=10, batch_size=60000)
# score = model.evaluate(normalized_X_train, y_train, batch_size=32)
# print(score)
predictions = model.predict(normalized_X_test) 
print(predictions)
predictions = model.predict_classes(normalized_X_test) 
print(predictions)


# save = np.c_[X_test[:,0],predictions]
# # save = np.vstack(([['PERID,Criminal']['']],save))
# print('shape',X_test[:,0].shape)
# print(save.shape)
# np.savetxt('submission.csv',save,delimiter=',',fmt='%10.20f')

sys.exit()
##########################################################
##########################################################
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
from sklearn.metrics import precision_score
print(precision_score(y_test,predictions))
sys.exit()

save = np.c_[X_test[:,0],predictions]
# save = np.vstack(([['PERID,Criminal']['']],save))
print('shape',X_test[:,0].shape)
print(save.shape)
np.savetxt('submission.csv',save,delimiter=',',fmt='%10.0f')
