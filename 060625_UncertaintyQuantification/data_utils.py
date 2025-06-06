from pymatgen.core.composition import Composition
import numpy as np
def get_samples_w_element_X(data, col_w_atomic_formula, element):

    has_element = data['formula'].apply(
        lambda x: element in [str(i) for i in Composition(x).elements]
        )
    test = data[has_element]

    train = data.drop(index=test.index)

    train = clean_data(train)
    test = test[train.columns]

    return train, test

def format_dataset(data):
    data = data.dropna(axis=1)
    for col in data.columns:
        if type(data[col].values[0]) == str:
            data = data.drop(axis=1, columns=col)
        elif type(data[col].values[0]) == np.bool_:
            data = data.drop(axis=1, columns=col)

    data = data.drop(axis=1, columns='atoms')


def get_target_label(data, target):
    y = data[target].values
    X = data.drop(axis=1, columns=target).values
    return X, y

def clean_data(data):
    for col_name in data.columns:
        if any([type(i)==str for i in data[col_name].values]): 
            data = data.drop(columns=col_name)
        elif any([type(i)==dict for i in data[col_name].values]):
            data = data.drop(columns=col_name)
        elif any([type(i)==np.bool_ for i in data[col_name].values]):
            data = data.drop(columns=col_name)
    data = data.dropna(axis=1, how='any')

    return data
