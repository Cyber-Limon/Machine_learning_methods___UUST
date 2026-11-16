import random

import numpy  as np
import pandas as pd

from pymatgen.core                    import Element, Composition
from sklearn.preprocessing            import StandardScaler
from sklearn.model_selection          import train_test_split
from matminer.featurizers.composition import ElementProperty, Meredig



def loading_data(path, compounds_range, properties_range, modification = None):
    compounds  = pd.read_excel(path, usecols=compounds_range)
    compounds  = compounds.values.tolist()
    compounds  = np.array(compounds)

    properties = pd.read_excel(path, usecols=properties_range)
    properties = properties.values.tolist()
    properties = np.array(properties)


    if modification is not None:
        compounds_names = pd.read_excel(path, nrows=0, usecols=compounds_range)
        compounds_names = compounds_names.columns.tolist()

        compositions = atomic_fractions(compounds, compounds_names)

        if modification == "magpie":
            feat = ElementProperty.from_preset(preset_name="magpie", impute_nan=True)
        else:
            feat = Meredig()

        feat.set_n_jobs(1)
        modifications = feat.featurize_many(compositions, ignore_errors=True)

        compounds = np.array(modifications)
        #compounds = np.hstack([compounds, modifications])


    X_train, X_test, y_train, y_test = train_test_split(
        compounds, properties,
        test_size=0.2,
        random_state=1107)

    return X_train, X_test, y_train, y_test



def atomic_fractions(compounds, compounds_names):
    result = []

    for compound in compounds:
        modified = []
        for i in range(len(compound)):
            modified.append(compound[i] / float(Element(compounds_names[i]).atomic_mass))

        total       = sum(modified)

        composition = {}
        for i in range(len(modified)):
            if modified[i] > 0:
                composition[compounds_names[i]] = modified[i] / total

        result.append(Composition(composition))

    return result



def normalization(X_train, X_test, y_train, y_test):
    X_train = np.asarray(X_train)
    X_test  = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    scaler_X = StandardScaler()
    X_train  = scaler_X.fit_transform(X_train)
    X_test   = scaler_X.transform(X_test)

    scaler_Y = StandardScaler()
    if y_train.ndim == 1:
        y_train = scaler_Y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test  = scaler_Y.transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_train  = scaler_Y.fit_transform(y_train)
        y_test   = scaler_Y.transform(y_test)

    return X_train, X_test, y_train, y_test



def cleaning(X, y, material_property):
    X_clean = []
    y_clean = []

    for material in range(len(X)):
        if y[material][material_property] != 0:
            X_clean.append(X[material])
            y_clean.append(y[material][material_property])

    return X_clean, y_clean



def augmentation(X, y, quantity, bias, material_property):
    X_current, y_current = cleaning(X, y, material_property)
    X_aug,     y_aug     = X_current.copy(), y_current.copy()

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
