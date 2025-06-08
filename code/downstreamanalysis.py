import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def run_random_forest(data, sample_size, rdata, replace):
    # Assume 'condition' is the target variable column in your CSV
    X = data.drop('Condition', axis=1)
    y = data['Condition']
    accuracies = []
    errors = []
    min_accuracies = []
    max_accuracies = []
    #     filepath = 'crc/'
    #     rdata = pd.read_csv(filepath+'crc_silvia_data.csv')
    #     condition_mapping = {'Healthy': 0, 'CRC': 1}
    #     # Replace values in the "Condition" column
    #     rdata['Condition'] = rdata['Condition'].replace(condition_mapping)

    #     filepath = 'datapaper/'
    #     rdata = pd.read_csv(filepath+'meldataclean.csv')
    #     rdata = rdata.drop('Unnamed: 0',axis=1)
    #     rdata.rename(columns={'BioReplicate': 'Condition'}, inplace=True)

    #     rdata0 = pd.read_csv('egf/dataegf.csv')
    #     rdata1 = pd.read_csv('egf/dataegfintAKT0.csv')
    #     rdata0['condition'] = 0
    #     rdata1['condition'] = 1
    #     rdata = pd.concat([rdata0, rdata1], ignore_index=True)
    X_test = rdata.drop('Condition', axis=1)
    y_test = rdata['Condition']
    #     for _ in range(10):  # Run 10 times
    for _ in range(100):  # Randomly select samples 10 times
        # Randomly select 'sample_size' samples
        selected_samples = np.random.choice(len(data), size=sample_size, replace=replace)
        X_sampled = X.iloc[selected_samples]
        y_sampled = y.iloc[selected_samples]

        # Split the data into training and testing sets
        #             X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

        # Create a random forest classifier
        clf = RandomForestClassifier(random_state=42)

        # Train the classifier
        #             clf.fit(X_train, y_train)
        clf.fit(X_sampled, y_sampled)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate classification error
        error = 1 - accuracy

        accuracies.append(accuracy)
        errors.append(error)
        min_accuracies.append(np.min(accuracies))
        max_accuracies.append(np.max(accuracies))

    mina = np.min(accuracies)
    maxa = np.max(accuracies)

    avg_accuracy = np.mean(accuracies)
    avg_error = np.mean(errors)

    return avg_accuracy, mina, maxa


def identify_significant_proteins(data, condition_col='Condition', significance_level=0.05, print_results=True,
                                  require_all_significant=False):
    """
    Identify proteins with significant differences between conditions using t-tests.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing protein abundance data with proteins as columns
        and a column indicating conditions
    condition_col : str, default 'Condition'
        Name of the column containing condition labels
    significance_level : float, default 0.05
        Threshold p-value for determining significance
    print_results : bool, default True
        Whether to print the results
    require_all_significant : bool, default False
        If True, proteins must be significant across all conditions
        If False, proteins are considered significant if they differ in at least one condition

    Returns:
    --------
    tuple : (significant_proteins, not_significant_proteins, t_test_results)
        Lists of significant and non-significant protein names, and DataFrame with all p-values
    """
    import pandas as pd
    from scipy.stats import ttest_ind

    # Get protein columns (excluding the condition column)
    protein_cols = [col for col in data.columns if col != condition_col]

    # Identify conditions in the dataset
    conditions = data[condition_col].unique()

    # Create a DataFrame to store t-test results
    t_test_results = pd.DataFrame(index=protein_cols)

    # Perform t-tests for each protein between each condition and all others
    for condition in conditions:
        # Select data for the current condition vs other conditions
        condition_data = data.loc[data[condition_col] == condition, protein_cols]
        other_data = data.loc[data[condition_col] != condition, protein_cols]

        # Initialize p-value list
        p_values = []

        # Perform t-test for each protein between conditions
        for protein in protein_cols:
            # Skip if the protein has all NaN values in either group
            if condition_data[protein].isna().all() or other_data[protein].isna().all():
                p_values.append(1.0)  # Not significant if all values are missing
                continue

            # Calculate t-test (handling potential empty arrays)
            try:
                _, p_value = ttest_ind(
                    condition_data[protein].dropna(),
                    other_data[protein].dropna(),
                    equal_var=False  # Using Welch's t-test which doesn't assume equal variances
                )
                p_values.append(p_value)
            except:
                # If t-test fails (e.g., only one sample), assign non-significant p-value
                p_values.append(1.0)

        # Assign p-values to the results DataFrame
        t_test_results[f"{condition}_p_value"] = p_values

    # Identify significant proteins based on the selected approach
    if require_all_significant:
        # Must be significant in all conditions
        significant_mask = t_test_results.le(significance_level).all(axis=1)
    else:
        # Significant in at least one condition
        significant_mask = t_test_results.le(significance_level).any(axis=1)

    significant_proteins = t_test_results[significant_mask].index.tolist()
    not_significant_proteins = t_test_results[~significant_mask].index.tolist()

    # Display the results if requested
    if print_results:
        print(f"Proteins with Significant Differences ({len(significant_proteins)}):")
        print(significant_proteins)
        print(f"\nProteins with Non-Significant Differences ({len(not_significant_proteins)}):")
        print(not_significant_proteins)

    return significant_proteins, not_significant_proteins, t_test_results


def calculate_percentage_of_variation(data, size, n_iterations=100, n_components=2,
                                      condition_column=None, impute_strategy='mean',
                                      standardize=True, random_state=None, verbose=False,
                                      return_all_stats=False):
    """
    Calculate the percentage of variation explained by principal components across multiple
    bootstrapped samples of a dataset.

    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing protein abundance data with proteins as columns
    size : int
        Sample size to use for each bootstrap iteration
    n_iterations : int, default 100
        Number of bootstrap iterations to perform
    n_components : int, default 2
        Number of principal components to use
    condition_column : str, default None
        If provided, sampling will be done separately within each condition
    impute_strategy : str, default 'mean'
        Strategy for SimpleImputer ('mean', 'median', 'most_frequent', 'constant')
    standardize : bool, default True
        Whether to standardize the data before PCA
    random_state : int, default None
        Random seed for reproducibility
    verbose : bool, default False
        Whether to print progress information
    return_all_stats : bool, default False
        If True, returns all variation values across iterations

    Returns:
    --------
    dict or tuple
        If return_all_stats=True: Dictionary with statistics for each condition (or 'all')
        If return_all_stats=False: Tuple of (average, minimum, maximum) variation percentages
    """
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # If condition column is provided, process each condition separately
    if condition_column is not None and condition_column in data.columns:
        conditions = data[condition_column].unique()
        results = {}

        for condition in conditions:
            if verbose:
                print(f"Processing condition: {condition}")

            # Filter data for the current condition
            condition_data = data[data[condition_column] == condition].drop(columns=[condition_column])

            # Calculate variation for this condition
            if size > len(condition_data):
                if verbose:
                    print(
                        f"Warning: Sample size {size} is larger than available data for condition {condition} ({len(condition_data)})")
                condition_size = len(condition_data)
            else:
                condition_size = size

            condition_results = _calculate_variation_for_dataset(
                condition_data, condition_size, n_iterations, n_components,
                impute_strategy, standardize, verbose
            )
            results[condition] = condition_results

        return results
    else:
        # Process the entire dataset without considering conditions
        if condition_column is not None and condition_column not in data.columns:
            if verbose:
                print(f"Warning: Condition column '{condition_column}' not found in data")

        # If condition column exists but we're not using it, drop it
        if condition_column in data.columns:
            processed_data = data.drop(columns=[condition_column])
        else:
            processed_data = data

        # Calculate variation for the whole dataset
        return _calculate_variation_for_dataset(
            processed_data, size, n_iterations, n_components,
            impute_strategy, standardize, verbose, return_all_stats
        )


def _calculate_variation_for_dataset(data, size, n_iterations, n_components,
                                     impute_strategy, standardize, verbose, return_all_stats=False):
    """Helper function to calculate variation for a single dataset"""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    all_variations = []

    # Check if sample size is larger than dataset
    if size > len(data):
        if verbose:
            print(f"Warning: Sample size {size} is larger than available data ({len(data)})")
        size = len(data)

    for i in range(n_iterations):
        if verbose and i % 10 == 0:
            print(f"  Iteration {i}/{n_iterations}")

        try:
            # Take a random sample with replacement
            sampled_data = data.sample(n=size, replace=True)

            # Handle missing values if present
            if sampled_data.isna().any().any():
                if verbose and i == 0:
                    print("  Imputing missing values")

                # Impute missing values
                imputer = SimpleImputer(strategy=impute_strategy)
                sampled_data_imputed = pd.DataFrame(
                    imputer.fit_transform(sampled_data),
                    columns=sampled_data.columns
                )

                # Use imputed data for further processing
                processed_data = sampled_data_imputed
            else:
                processed_data = sampled_data

            # Remove columns with zero variance
            non_zero_var_cols = processed_data.columns[processed_data.std() != 0]

            if len(non_zero_var_cols) < 2:
                if verbose:
                    print(
                        f"  Warning: Found only {len(non_zero_var_cols)} columns with non-zero variance. Skipping iteration.")
                continue

            data_for_pca = processed_data[non_zero_var_cols]

            # Standardize if requested
            if standardize:
                data_for_pca = (data_for_pca - data_for_pca.mean()) / data_for_pca.std()

            # Apply PCA
            pca = PCA(n_components=min(n_components, len(non_zero_var_cols)))
            pca.fit(data_for_pca)

            # Calculate percentage of variation
            percentage_variation = sum(pca.explained_variance_ratio_) * 100
            all_variations.append(percentage_variation)

        except Exception as e:
            if verbose:
                print(f"  Error in iteration {i}: {str(e)}")

    # Calculate statistics
    if len(all_variations) == 0:
        if verbose:
            print("No valid iterations completed")
        return (0, 0, 0, 0, []) if return_all_stats else (0, 0, 0)

    avg_var = np.mean(all_variations)
    min_var = np.min(all_variations)
    max_var = np.max(all_variations)
    std_var = np.std(all_variations)

    if return_all_stats:
        return avg_var, min_var, max_var, std_var, all_variations
    else:
        return avg_var, min_var, max_var