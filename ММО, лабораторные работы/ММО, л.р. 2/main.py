from Preprocessing import *
from Methods       import *
from Analysis      import *



variant = 2

if variant == 1:
    path             = "MLM, dataset 1.xlsx"
    compounds_range  = "B:V"
    properties_range = "W:AD"
else:
    path             = "MLM, dataset 2.xlsx"
    compounds_range  = "A:T"
    properties_range = "U:AB"



X_train, X_test, y_train, y_test = loading_data(path, compounds_range, properties_range)

properties_names = pd.read_excel(path, nrows=0, usecols=properties_range)
properties_names = properties_names.columns.tolist()



scores = ensemble(method_SVR, X_train, X_test, y_train, y_test, False, 2000, 0.01)
metrics(scores, properties_names)
schedule(scores, properties_names)
