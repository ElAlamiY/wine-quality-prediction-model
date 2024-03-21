def select_features(data, features_to_select):
    """
    Select specified features from the dataset and drop all others.

    Parameters:
    data (pd.DataFrame): The input dataset.
    features_to_select (list of str): The names of the features to keep.

    Returns:
    pd.DataFrame: The transformed dataset with only the selected features.
    """
    # Ensure features_to_select is a list to allow for single feature as a string
    if isinstance(features_to_select, str):
        features_to_select = [features_to_select]
    
     # Always include the target column "quality" if it's not already in the list
    if "quality" not in features_to_select:
        features_to_select.append("quality")
        
    # Select the specified features
    selected_data = data[features_to_select]
    return selected_data