

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
# X_train, y_train = X_train[:60000], y_train[:60000]



from sklearn.naive_bayes import GaussianNB,MultinomialNB
# from sklearn.model_selection import GridSearchCV

# param_grid = [

# ]

clf = MultinomialNB()

# grid_search = GridSearchCV(clf, param_grid, cv=10)
# print(grid_search.fit(X_train, y_train))
# from sklearn.model_selection import cross_val_score
# cross_val_scor = cross_val_score(clf, X_train, y_train, cv=10)
# np.savetxt("cross_val_score.txt", cross_val_scor, delimiter="," , fmt='%10.5f')
clf.fit(X_train, y_train)
# In[12]:


# cvres = grid_search.cv_results_

# from pprint import pprint
# pprint(grid_search.cv_results_)

# import csv

# with open('grid_search_details.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file)
#     for key, value in grid_search.cv_results_.items():
#        writer.writerow([key, value])
    
# print(grid_search.best_estimator_)
# np.savetxt("best_estimator.txt", [grid_search.best_estimator_],fmt="%s" )


# In[13]:


# final_model = grid_search.best_estimator_
# final_predictions = final_model.predict(X_test)
final_predictions = clf.predict(X_test)

# In[18]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(y_test, final_predictions)
print(accuracy_score(y_test, final_predictions))
print(conf_mx)
np.savetxt("accuracy_score.txt", [accuracy_score(y_test, final_predictions)], delimiter="," , fmt='%10.5f')
np.savetxt("conf_mat.csv", conf_mx, delimiter="," , fmt='%10.0f')

data = conf_mx

image_product = 70
new_data = np.zeros(np.array(data.shape) * image_product)
for j in range(data.shape[0]):
    for k in range(data.shape[1]):
        new_data[j * image_product: (j+1) * image_product, k * image_product: (k+1) * image_product] = data[j, k]
# plt.imshow(new_data, cmap=plt.cm.gray)
# plt.show()
plt.imsave("confusion_mat.jpg" , new_data, dpi=1000,cmap=plt.cm.gray)


# In[ ]:


# from sklearn.model_selection import cross_val_predict,cross_val_score
# clf = KNeighborsClassifier(n_jobs=-1)
# y_train_pred = cross_val_score(clf, X_train, y_train, cv=10)
# print(y_train_pred)
# y_train_pred = cross_val_predict(clf, X_train, y_train, cv=10)
# print(y_train_pred)

