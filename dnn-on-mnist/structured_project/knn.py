from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [
{'n_neighbors': [3, 5, 7, 30]},
]

clf = KNeighborsClassifier(n_jobs=-1)

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X_train, y_train)
