import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def xgbmodel(data):
    # Separate features and target
    X = data.drop('quality', axis=1)
    y = data['quality']
    y = y - y.min()  # Ensure labels start from 0

    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train)

    # Validation predictions
    y_val_pred = model.predict(X_val_scaled)

    # Evaluate the model on the validation set
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    # Final test predictions (only after tuning and selecting the model)
    y_test_pred = model.predict(X_test_scaled)

    # Evaluate the model on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')


from sklearn.model_selection import GridSearchCV

def xgbmodel_optimized(data):
    # Separate features and target
    X = data.drop('quality', axis=1)
    y = data['quality']
    y = y - y.min()  # Ensure labels start from 0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Define the grid of hyperparameters to search
    param_grid = {
        'max_depth': [7],
        'learning_rate': [0.1],
        'n_estimators': [323],
        'subsample': [0.7]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and best score
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')


