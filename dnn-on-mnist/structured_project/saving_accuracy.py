'''best estimator prediction'''
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)

'''saving result'''
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

