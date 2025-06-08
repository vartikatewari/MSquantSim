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


def plot_corr(df, size=10,font_scale=0.7):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    plt.rcParams.update({'font.size': 8 * font_scale})

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    corr = corr[corr <= 0.99]
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap=cmap)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6 * font_scale)
    ax.set_yticklabels(corr.columns, fontsize=6 * font_scale)
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    cbar.ax.tick_params(labelsize=6 * font_scale)

    # Adjust layout to make room for labels
    plt.tight_layout()

    return
