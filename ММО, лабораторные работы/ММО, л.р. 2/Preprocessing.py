import random

import numpy  as np
import pandas as pd

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split



def loading_data(path, compound_range, properties_range):
    compound  = pd.read_excel(path, usecols=compound_range)
    compound  = compound.values.tolist()
    compound  = np.array(compound)

    properties = pd.read_excel(path, usecols=properties_range)
    properties = properties.values.tolist()
    properties = np.array(properties)

    X_train, X_test, y_train, y_test = train_test_split(
        compound, properties,
        test_size=0.2,
        random_state=1107)

    return X_train, X_test, y_train, y_test



def normalization(X_train, X_test, y_train, y_test):
    scaler_X = StandardScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_test   = scaler_X.transform(X_test)

    scaler_Y = StandardScaler()
    y_train  = scaler_Y.fit_transform(y_train)
    y_test   = scaler_Y.transform(y_test)

    return X_train, X_test, y_train, y_test



def augmentation(X, y, quantity, bias, material_property):
    X_current = []
    y_current = []

    for material in range(len(X)):
        if y[material][material_property] != 0:
            X_current.append(X[material])
            y_current.append(y[material][material_property])

    X_aug = list(X_current.copy())
    y_aug = list(y_current.copy())

    while len(X_aug) < quantity:
        material  = random.randint(0, len(X_current) - 1)
        x_current = X_current[material].copy()

        sample = []
        for i in range(len(x_current)):
            if x_current[i] > bias:
                sample.append(i)

        if len(sample) >= 6:
            number_of_changes = 6
        elif len(sample) >= 4:
            number_of_changes = 4
        elif len(sample) >= 2:
            number_of_changes = 2
        else:
            continue

        changes = random.sample(sample, number_of_changes)
        for change in range(len(changes)):
            if change % 2 == 0:
                x_current[changes[change]] += bias
            else:
                x_current[changes[change]] -= bias

        X_aug.append(x_current)
        y_aug.append(y_current[material])

    return X_aug, y_aug
