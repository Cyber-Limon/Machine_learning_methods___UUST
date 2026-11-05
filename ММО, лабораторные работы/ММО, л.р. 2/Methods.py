import numpy as np

from Preprocessing   import augmentation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



method_names = ["ElasticNet", "SVR", "RandomForest", "XGBoost", "MLP", "kNN", "ExtraTrees", "GaussianProcess", "CatBoost"]



def method_ElasticNet(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import ElasticNet

    model = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000, random_state=42)

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_SVR(X_train, X_test, y_train, y_test):
    from sklearn.svm         import SVR
    from sklearn.multioutput import MultiOutputRegressor

    svr   = SVR(kernel='rbf')
    model = MultiOutputRegressor(svr)

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_SVR_ensemble(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVR

    model = SVR(kernel='rbf')

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_RandomForest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_jobs=-1)

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_XGBoost(X_train, X_test, y_train, y_test):
    from xgboost import XGBRegressor

    model = XGBRegressor(n_jobs=-1)

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_MLP(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPRegressor

    params = {
        'hidden_layer_sizes': (256, 128, 64, 32),
        'max_iter': 500,
        'random_state': 42,
        'verbose': True,
        'activation': 'relu',
        'solver': 'adam',
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.001,
        'alpha': 0.0001,
        'batch_size': 'auto',
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'max_fun': 15000
    }

    model = MLPRegressor(**params)

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_kNN(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsRegressor

    model = KNeighborsRegressor(n_neighbors=5, weights='uniform')

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_ExtraTrees(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import ExtraTreesRegressor

    model = ExtraTreesRegressor(n_estimators=100, max_features='sqrt')

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_GaussianProcess(X_train, X_test, y_train, y_test):
    from sklearn.gaussian_process import GaussianProcessRegressor

    model = GaussianProcessRegressor()

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def method_CatBoost(X_train, X_test, y_train, y_test, loss_function='MultiRMSE'):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        loss_function=loss_function,
        random_seed=42,
        verbose=100
    )

    return model_evaluation(model, X_train, X_test, y_train, y_test)



def model_evaluation(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = []
    score.append(r2_score(y_test, y_pred))
    score.append(mean_absolute_error(y_test, y_pred))
    score.append(mean_squared_error(y_test, y_pred) ** 0.5)

    return score



def all_methods(X_train, X_test, y_train, y_test):
    scores = [
        method_ElasticNet(X_train, X_test, y_train, y_test),
        method_SVR(X_train, X_test, y_train, y_test),
        method_RandomForest(X_train, X_test, y_train, y_test),
        method_XGBoost(X_train, X_test, y_train, y_test),
        method_MLP(X_train, X_test, y_train, y_test),
        method_kNN(X_train, X_test, y_train, y_test),
        method_ExtraTrees(X_train, X_test, y_train, y_test),
        method_GaussianProcess(X_train, X_test, y_train, y_test),
        method_CatBoost(X_train, X_test, y_train, y_test)
    ]

    return scores



def ensemble(method, quantity, bias, X_train, X_test, y_train, y_test):
    if method == method_SVR:
        method = method_SVR_ensemble

    R_2  = []
    MAE  = []
    RMSE = []

    for material_property in range(len(y_train[0])):
        X_train_current, y_train_current = augmentation(X_train, y_train, quantity,      bias, material_property)
        X_test_current,  y_test_current  = augmentation(X_test,  y_test,  quantity // 4, bias, material_property)

        X_train_current = np.array(X_train_current)
        X_test_current  = np.array(X_test_current)
        y_train_current = np.array(y_train_current)
        y_test_current  = np.array(y_test_current)

        if method != method_CatBoost:
            score = method(X_train_current, X_test_current, y_train_current, y_test_current)
        else:
            score = method(X_train_current, X_test_current, y_train_current, y_test_current, "RMSE")

        R_2.append(score[0])
        MAE.append(score[1])
        RMSE.append(score[2])

    return [sum(R_2) / len(R_2), sum(MAE) / len(MAE), sum(RMSE) / len(RMSE)]



def all_ensembles(quantity, bias, X_train, X_test, y_train, y_test):
    scores = []

    for method in method_names:
        method = globals()["method_" + method]
        scores.append(ensemble(method, quantity, bias, X_train, X_test, y_train, y_test))

    return scores
