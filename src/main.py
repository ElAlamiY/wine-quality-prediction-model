import pandas as pd
import sys
import os

# Get the directory path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'src' directory to the Python module search path
src_path = os.path.join(current_dir, "src")
sys.path.insert(0, src_path)

# Local imports
import datacleaning
import eda
import featureengineering

# Import the rfmodel module from the 'models' directory
models_path = os.path.join(current_dir, "..", "models")
sys.path.insert(0, models_path)

import rfmodel


def main():
    # We are working here wiht a file named "winequality-white.csv" in the data folder
    data_file_path = os.path.join(current_dir, '..', 'data', 'winequality-white.csv')
    data_df = pd.read_csv(data_file_path, sep=';')
    
    # Clean data
    data = datacleaning.clean_data(data_df)

    # Perform Exploratory Data Analysis (EDA)
    eda.plot_correlation_matrix(data)

    # Let's plot the importance scores
    target = 'quality'
    importances = featureengineering.process(data, target)
    sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
    print(f"Features sorted by their importance score {sorted_importances}")
    
    # Train model
    model, val_accuracy, test_accuracy = rfmodel.train_wine_quality_model(data)
    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Test Accuracy: {test_accuracy}')
    return model

if __name__ == "__main__":
    trained_model = main()
