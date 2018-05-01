
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Preparing Data'''
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

'''Cropping Data'''
X_train, y_train = X[:4000], y[:4000]


# from sklearn.model_selection import cross_val_score


# In[4]:


# a = XTest[2]
# print(neigh.predict([a]))
# plt.imshow(np.reshape(a,(28,28)))


# In[5]:


# from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier


# In[6]:

'''Model Preparing'''
'''
neigh = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
# neigh = MLPClassifier(solver='sgd',learning_rate='constant', learning_rate_init=0.1, hidden_layer_sizes=(10, 10))
# neigh = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10), max_iter=20, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-8,
#                     learning_rate_init=.1)
# neigh = GaussianNB()
# neigh.fit(XTrain, YTrain) 
# print("Training set score: %f" % mlp.score(XTrain, YTrain))
# print("Test set score: %f" % mlp.score(XTest, YTest))
cross_val_score(neigh, X_train, y_train, cv=2, scoring="accuracy")
# predictions = neigh.predict(XTest)
# print(accuracy_score(YTest, predictions))


# In[ ]:


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(neigh, X_train, y_train, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_train_pred)


# In[ ]:


# y_train_pred[6000:6090]
y_train[1000:1090]


# In[ ]:


from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train, y_train_pred))
print(recall_score(y_train, y_train_pred))
from sklearn.metrics import f1_score
f1_score(y_train, y_train_pred)

'''