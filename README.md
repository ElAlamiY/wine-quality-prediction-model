# Wine Quality Prediction Model

## Project Overview
This project focuses on building a predictive model for wine quality based on various physicochemical properties. The pipeline includes data cleaning, exploratory data analysis (EDA), feature engineering, feature selection, model selection, model training, and model evaluation. We aim to understand the underlying patterns that determine wine quality and develop a model that can predict the quality of wine with high accuracy.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Feature Selection](#feature-selection)
  - [Model Selection](#model-selection)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation
Provide instructions on setting up the project environment. Typically, this involves steps to clone the repository, install dependencies, and any necessary configuration steps.

```bash
git clone https://github.com/yassine-el-alami/wine-quality-prediction-model
cd wine-quality-prediction-model
pip install -r requirements.txt
```

## Usage

To run the wine quality prediction model, follow these steps:

1. Ensure you have navigated to the root directory of the project.
2. The necessary Python packages should already be installed as specified in the `requirements.txt` file. If you haven't installed the dependencies yet, please run:
   ```bash
   pip install -r requirements.txt
   ```
3. To execute the model, run the following command from the root directory:
   ```bash
   python src/main.py
   ```

This will initiate the process defined in `main.py`, including data preparation, model training, and evaluation steps. Make sure the `data` folder is properly set up as outlined in the [Data Setup](#data-setup) section.

"For an in-depth look at the model development process, including data preprocessing, feature engineering, and model evaluation, please refer to our detailed Jupyter notebook located at `/notebooks/pipeline.ipynb`."

## Project Structure
Please note that the data folder has to populated with own files.
```
/project-directory
    /src            # Source files for model training and evaluation
        main.py
        helper.py
        datacleaning.py
        eda.py
        featureengineering.py
        featureselection.py
    /models
        rfmodel.py  # Random Forest Model used in the project
        gbmmodel.py
    /notebooks      # Jupyter notebooks for EDA and analysis
        dataanalysis.ipynb
        pipeline.ipynb
    /data           # Directory for dataset storage, Make sure to add your own data in this folder
    /tests          
    README.md
    LICENSE.md
    requirements.txt
```

## Methodology

### Data Cleaning
The data cleaning process analyzes the dataset to handle missing values, and ensure data consistency.

### Exploratory Data Analysis (EDA)
Performed both through simple pandas and seaborn-based analysis for quick insights and a dedicated notebook `/notebooks/dataanalysis.ipynb` for deeper investigation into the dataset's distribution and relationships.

### Feature Engineering
Involved handling outliers and calculating feature importance scores to enhance model performance.

### Feature Selection
Selected the most relevant features for the model, primarily based on importance scores.

### Model Selection
For this project, Random Forest was chosen for its effectiveness and suitability.

### Model Training
The model was first trained without dropping any features, Later, in the evaluation step, feature selection was used to see how it affects validation and test scores.

### Model Evaluation
Evaluated the model by fine-tuning hyperparameters and assessing the impact of feature engineering and selection methods on its accuracy.

## Results
At the end of our journey through cleaning data, analyzing it, tweaking features, and training models, we've arrived at a valuable conclusion: our trained model is now ready to predict wine quality with good accuracy. Interestingly, our journey revealed that a simpler approach outperformed more complex ones.

Initially, we thought that carefully selecting and engineering features would lead to the best performance. However, it turned out that using a broader set of features without overcomplicating the model gave us better scores. This finding suggests that wine quality is influenced by a wide range of factors, and capturing as many of these as possible helps in making more accurate predictions.

In essence, our model, which embraces a wide lens on the data, is primed for predicting the nuanced qualities of wine, demonstrating the power of simplicity in tackling complex predictive tasks.

## Contributing


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## Acknowledgments
