from itertools import combinations
import featureselection
import rfmodel

def generate_feature_combinations(data):
    """
    Generate all possible combinations of features.

    Parameters:
    data (pd.DataFrame): The dataset containing features and target.

    Returns:
    list: List of lists representing all combinations of features.
    """
    # Extract all features from the DataFrame except the target column
    features = data.columns[data.columns != 'quality']

    # Generate all possible combinations of features
    all_combinations = []
    for r in range(1, len(features) + 1):
        all_combinations.extend([list(combo) for combo in combinations(features, r)])

    return all_combinations

def incremental_feature_training(data, sorted_features):
    """
    Incrementally trains models by adding one feature at a time based on feature importance.
    
    Parameters:
    data (pd.DataFrame): The input dataset.
    sorted_features (list of tuples): Sorted list of features and their importances, descending.
    
    Returns:
    tuple: The best performance metrics and the corresponding list of features.
    """
    # Extract just the feature names from the sorted feature list
    sorted_feature_names = [feat[0] for feat in sorted_features]

    performance_records = []

    for i in range(1, len(sorted_feature_names) + 1):
        # Select the top i features
        current_features = sorted_feature_names[:i]
        subdata = featureselection.select_features(data, current_features)
        
        # Train the model using the selected features
        model_performance = rfmodel.train_wine_quality_model(subdata)
        val_accuracy, test_accuracy = model_performance[1], model_performance[2]
        
        # Save the performance along with the current feature set
        performance_records.append((val_accuracy, test_accuracy, current_features))

    # Find the entry with the highest test accuracy
    best_performance = max(performance_records, key=lambda x: x[1])

    return best_performance


