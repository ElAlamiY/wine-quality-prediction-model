import pandas as pd
import os

def clean_data(data=None, scaling_factors=None):
    current_dir = os.path.dirname(__file__)
    data_file = os.path.join(current_dir, 'data', 'winequality-white.csv')

    if data.empty:
        data = data_file
    if scaling_factors is None:
        scaling_factors = {
            'volatile acidity': (1000, 100),
            'chlorides': (1000, 1),
            'density': (1000, 100)
        }

    for feature, (factor, threshold) in scaling_factors.items():
        data[feature] = data[feature].apply(lambda x: x/factor if x > threshold else x)

    return data
