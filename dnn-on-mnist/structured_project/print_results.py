
cvres = grid_search.cv_results_

from pprint import pprint
pprint(grid_search.cv_results_)

import csv

with open('grid_search_details.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in grid_search.cv_results_.items():
       writer.writerow([key, value])
    
print(grid_search.best_estimator_)
np.savetxt("best_estimator.txt", [grid_search.best_estimator_],fmt="%s" )
