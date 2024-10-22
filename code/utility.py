import pandas as pd


def read_df(file_path, delimiter=','):
    """
    Reads a CSV file into a DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.
    - delimiter (str): Delimiter used in the CSV file. Default is ','.

    Returns:
    - pd.DataFrame: DataFrame containing the CSV data.

    Raises:
    - FileNotFoundError: If the file is not found at the specified path.
    - pd.errors.EmptyDataError: If the file is empty.
    - pd.errors.ParserError: If there is a parsing error (e.g., incorrect delimiter).
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        print(f"DataFrame loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: Parsing error encountered. Check the delimiter and file format.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def show_protein_abundance_features(file_path):
    """
    Reads a protein abundance matrix from a CSV file and displays its features.

    Parameters:
    - file_path (str): Path to the protein abundance CSV file.

    Returns:
    - data (pd.DataFrame): The DataFrame containing the protein abundance data.
    """
    # Read the protein abundance matrix
    data = pd.read_csv(file_path)

    # Display the first few rows
    print("First few rows of the protein abundance matrix:")
    display(data.head())

    # Display summary statistics
    print("\nSummary statistics:")
    display(data.describe())

    # Display column names and data types
    print("\nColumn names and data types:")
    display(data.info())

    return data