import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint
import joblib

def train_model(X_train, y_train):
    # Define the parameter distributions for random search
    param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.1, 0.3),
        'n_estimators': randint(50, 200)
    }

    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
    y_train -= y_train.min()
    # Convert y_train to list
    y_train = y_train.tolist()

    # Perform random search
    random_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_

    # Initialize XGBoost classifier with the best parameters
    best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')

    # Train the best model on the entire dataset
    best_model.fit(X_train, y_train)

    return best_model

def validate_model(model, X_val, y_val):
    # Make predictions on the validation dataset
    y_pred = model.predict(X_val)

    # Evaluate the model on the validation dataset
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy:.4f}')

def test_model(model, X_test, y_test):
    # Make predictions on the test dataset
    y_pred = model.predict(X_test)

    # Evaluate the model on the test dataset
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.4f}')

def train_validate_test_and_save_model(data):
    # Split data into train, validation, and test sets
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_intermediate, y_train, y_intermediate = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.5, random_state=42)

    # Train the model
    trained_model = train_model(X_train, y_train)

    # Validate the model
    validate_model(trained_model, X_val, y_val)

    # Test the model
    test_model(trained_model, X_test, y_test)


