from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def linear_regression_model_tuned(data):
    # Separate features and target
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for tuning
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}

    # Create the Ridge Regression model
    model = Ridge()

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    test_score = best_model.score(X_test_scaled, y_test)
    print(f'Best Model Test Score (R^2): {test_score:.4f}')

    return best_model
