import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def exclude_outliers(data, columns_to_exclude=None, lower_percentile=5, upper_percentile=95):
    """
    Exclude data points outside the specified percentiles for specific columns, while preserving others.

    Parameters:
    data (pd.DataFrame): Input data.
    columns_to_exclude (list): List of column names to exclude outliers from (default: None, excludes all columns).
    lower_percentile (float): Lower percentile boundary (default: 5).
    upper_percentile (float): Upper percentile boundary (default: 95).

    Returns:
    pd.DataFrame: Data with outliers removed in specified columns while preserving others.
    """
    if columns_to_exclude is None:
        columns_to_exclude = data.columns

    filtered_data = data.copy()

    for col in columns_to_exclude:
        col_data = data[col]
        lower_bound = np.percentile(col_data, lower_percentile)
        upper_bound = np.percentile(col_data, upper_percentile)
        filtered_data[col] = col_data[(col_data >= lower_bound) & (col_data <= upper_bound)]

    return filtered_data


def plot_feature_importance(features, importances):
    """
    Plot feature importances.

    Parameters:
    features (list): List of feature names.
    importances (array-like): Feature importances.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.show()

def feature_importance_rf(X, y, feature_names):
    """
    Compute feature importance using Random Forest and return it as a list of tuples,
    with each tuple containing the name of the feature and its importance.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    feature_names (list): List of feature names.

    Returns:
    list of tuples: Each tuple contains the name of the feature and its importance.
    """
    # Initialize Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # Fit the model
    rf.fit(X, y)
    # Get feature importances
    importances = rf.feature_importances_
    # Pair feature names with their importances
    feature_importance_pairs = [(feature, importance) for feature, importance in zip(feature_names, importances)]
    # Plot feature importances
    plot_feature_importance(feature_names, importances)
    return feature_importance_pairs


def process(data, target_column_name):
    """
    Separate features and target, and compute feature importances.

    Parameters:
    csv_path (str): Path to the CSV file.
    target_column_name (str): Name of the target column.
    """

    # Separate features and target
    X = data.drop(target_column_name, axis=1)
    y = data[target_column_name]
    feature_names = X.columns.tolist()
    
    # Compute and plot feature importances
    return feature_importance_rf(X, y, feature_names)
