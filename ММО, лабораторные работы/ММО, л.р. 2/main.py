from Preprocessing import *
from Methods       import *
from Analysis      import *



path             = "MLM, dataset.xlsx"
compound_range   = "B:V"
properties_range = "W:AD"

X_train, X_test, y_train, y_test = loading_data(path, compound_range, properties_range)



#X_train, X_test, y_train, y_test = normalization(X_train, X_test, y_train, y_test)

print(ensemble(method_ElasticNet, 2000, 0.01, X_train, X_test, y_train, y_test))
