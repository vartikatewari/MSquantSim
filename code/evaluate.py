import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Function to read and display features of the protein abundance matrix

def plot_correlation_matrix(data, figsize=(10, 8), cmap='coolwarm', annot=True, fmt='.2f'):
    """
    Plots a correlation matrix heatmap for the given DataFrame.

    Parameters:
    - data (pd.DataFrame): The input data to calculate the correlation matrix.
    - figsize (tuple): Size of the heatmap figure. Default is (10, 8).
    - cmap (str): Colormap for the heatmap. Default is 'coolwarm'.
    - annot (bool): Whether to annotate each cell with the correlation coefficient. Default is True.
    - fmt (str): String formatting for annotations. Default is '.2f' (two decimal places).

    Returns:
    - None: Displays the heatmap of the correlation matrix.
    """
    try:
        # Compute the correlation matrix
        correlation_matrix = data.corr()
        print("Correlation matrix computed successfully.")

        # Plot the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, cmap=cmap, annot=annot, fmt=fmt)
        plt.title('Correlation Matrix Heatmap')
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting the correlation matrix: {e}")
        raise
