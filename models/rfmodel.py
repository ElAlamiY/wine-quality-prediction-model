import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def train_wine_quality_model(data):
    """
    Trains a RandomForest model on the provided wine quality dataset.

    Parameters:
    data (pd.DataFrame): The cleaned wine quality dataset with features and target.

    Returns:
    The trained RandomForest model and the test accuracy as a float.
    """
    # Splitting the dataset into features and target variable
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split the data into training, validation, and test sets 
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=150, random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Validate the model
    y_val_pred = rf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    #print(f'Validation Accuracy: {val_accuracy}')

    # Test the model
    y_test_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    #print(f'Test Accuracy: {test_accuracy}')

    return rf, val_accuracy, test_accuracy

def train_wine_quality_model_enhanced(data):
    """
    Trains a RandomForest model on the provided wine quality dataset using a simplified approach.
    Includes validation accuracy in the output.

    Parameters:
    data (pd.DataFrame): The wine quality dataset with features and target.

    Returns:
    The trained RandomForest model, validation accuracy as a float, test accuracy as a float, and the best hyperparameters.
    """
    # Prepare the data
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the RandomForest model
    rf = RandomForestClassifier(random_state=42)

    # Define a simplified hyperparameter space
    param_distributions = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 15, 17, 20, 25, 30],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4],
    }

    # Perform randomized search with cross-validation
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Best model
    best_model = random_search.best_estimator_
    
    # Average validation accuracy from the cross-validation
    val_accuracy = random_search.best_score_
    
    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return best_model, val_accuracy, test_accuracy, random_search.best_params_
