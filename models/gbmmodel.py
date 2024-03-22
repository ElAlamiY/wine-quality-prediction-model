import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train):
    # Initialize XGBoost classifier with default parameters and evaluation metric to avoid warnings
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    
    # Train the model on the training dataset
    model.fit(X_train, y_train)

    return model

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

def train_validate_test_model(data):
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

    # Optionally, save the model
    # joblib.dump(trained_model, 'xgb_model.joblib')

#
