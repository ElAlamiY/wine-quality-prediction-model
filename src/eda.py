import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D 

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_file = os.path.join(data_dir, 'winequality-white.csv')

def summarize_data(data=data_file):
    """Print summary statistics and information about the dataset."""
    print("Dataset Info:")
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())

def plot_distribution(data=data_file, feature="quality"):
    """Plot distribution for a categorical feature."""
    plt.figure(figsize=(10, 6))
    plt.title(f'Distribution of {feature}')
    sns.countplot(x=feature, data=data)
    plt.show()


def plot_correlation_matrix(data=data_file):
    """Plot the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_pairwise_relationships(data=data_file, columns=None):
    """Plot pairwise relationships in the dataset."""
    if columns is not None:
        sns.pairplot(data[columns])
    else:
        sns.pairplot(data)
    plt.show()

def plot_feature_by_quality(data=data_file, feature="alcohol"):
    """
    Plot the distribution of a specified feature by wine quality, including the 5th and 95th percentiles.
    
    Parameters:
    - data: pandas DataFrame containing the wine dataset.
    - feature: str, the name of the feature to plot.
    """
    
    # Calculate the 5th and 95th percentiles for the specified feature within each 'quality' group
    percentiles = data.groupby('quality')[feature].quantile([0.05, 0.95])

    # Visualize the relationship between the specified feature and 'quality'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quality', y=feature, data=data, showfliers=True)
    plt.title(f'{feature.capitalize()} by Wine Quality')
    plt.xlabel('Quality Score')
    plt.ylabel(feature.capitalize())

    # Plot the 5th and 95th percentiles
    qualities = data['quality'].unique()
    qualities.sort()
    for quality in qualities:
        y_10th = percentiles.loc[quality, 0.05]
        y_90th = percentiles.loc[quality, 0.95]
        xpos = list(qualities).index(quality)

        plt.hlines(y=y_10th, xmin=xpos-0.4, xmax=xpos+0.4, color='red', linestyles='--', linewidth=1)
        plt.hlines(y=y_90th, xmin=xpos-0.4, xmax=xpos+0.4, color='blue', linestyles='--', linewidth=1)

    # Add a legend for the 10th and 90th percentile lines
    legend_elements = [
        Line2D([0], [0], color='red', lw=1, linestyle='--', label='5th Percentile'),
        Line2D([0], [0], color='blue', lw=1, linestyle='--', label='95th Percentile')
    ]
    plt.legend(handles=legend_elements)

    plt.show()

def main():
    data = pd.read_csv(data_file, sep= ";")
    
    summarize_data(data)
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    plot_distribution(data, None)
    plot_correlation_matrix(data)
    plot_pairwise_relationships(data, numerical_columns[:4])  # Adjust as needed

if __name__ == "__main__":
    main()