from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def linear_regression_model(data):
    # Separate features and target
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate the model on the test set
    test_score = model.score(X_test_scaled, y_test)
    print(f'Test Score (R^2): {test_score:.4f}')

    # Get coefficient values for each feature
    coefficients = model.coef_
    feature_names = X.columns
    feature_coefficients = dict(zip(feature_names, coefficients))

    return model, feature_coefficients

# Example usage:
# model, coefficients = linear_regression_model(data)
# print(coefficients)
