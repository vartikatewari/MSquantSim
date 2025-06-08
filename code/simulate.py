def simulate_using_copula(dataset, num_simulations=1, samples_per_simulation=None,
                          output_dir=None, output_prefix='simulated',
                          return_data=True, random_state=None):
    """
    Simulate protein abundance data using Gaussian Copula models.

    Parameters:
    -----------
    dataset : pandas DataFrame
        Input dataset containing protein abundance data
    num_simulations : int, default 1
        Number of simulation iterations to perform
    samples_per_simulation : int, default None
        Number of samples to generate in each simulation
        If None, uses the same number of samples as the input dataset
    output_dir : str, default None
        Directory to save CSV files. If None, files are not saved
    output_prefix : str, default 'simulated'
        Prefix for output filenames
    return_data : bool, default True
        Whether to return the simulated data
    random_state : int, default None
        Random seed for reproducibility

    Returns:
    --------
    list or None
        If return_data=True: List of pandas DataFrames containing simulated data
        If return_data=False: None
    """
    from copulas.multivariate import GaussianMultivariate
    import pandas as pd
    import os
    import numpy as np

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Default to the same number of samples as the input dataset
    if samples_per_simulation is None:
        samples_per_simulation = len(dataset)

    # Create output directory if it doesn't exist
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List to store simulated datasets if return_data is True
    simulated_datasets = [] if return_data else None

    # Run simulations
    for i in range(num_simulations):
        # Create and fit the Gaussian copula model
        model = GaussianMultivariate()
        model.fit(dataset)

        # Sample from the model
        synthetic = model.sample(samples_per_simulation)

        # Save to CSV if output_dir is provided
        if output_dir is not None:
            output_path = os.path.join(output_dir, f"{output_prefix}_{i}.csv")
            synthetic.to_csv(output_path, index=False)

        # Store the simulated dataset if return_data is True
        if return_data:
            simulated_datasets.append(synthetic)

    return simulated_datasets





def simulate_using_tvae(data, metadata, output_dir, dataset_name, num_iterations=1000, num_rows=None):
    """
    Generate synthetic data using TVAE synthesizer and save to CSV files.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input data to fit the synthesizer on
    metadata : sdv.metadata.SingleTableMetadata
        The metadata object describing the data structure
    output_dir : str
        Directory path where synthetic data files will be saved (e.g., 'crc/tvae/')
    dataset_name : str
        Base name for the dataset (e.g., 'crc_0', 'egfAKT100', 'mel1')
    num_iterations : int, default=1000
        Number of synthetic datasets to generate
    num_rows : int, optional
        Number of rows to generate per synthetic dataset. If None, uses len(data)

    Returns:
    --------
    synthesizer : TVAESynthesizer
        The fitted synthesizer object (can be reused later)
    """
    from sdv.single_table import TVAESynthesizer
    import os
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize and fit the synthesizer
    print(f"Fitting TVAE synthesizer on {len(data)} rows...")
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(data)
    print("Fitting complete!")

    # Determine number of rows to generate
    if num_rows is None:
        num_rows = len(data)

    # Generate synthetic data
    for i in range(num_iterations):
        synthetic_data = synthesizer.sample(num_rows=num_rows)

        # Create filename and save
        filename = f"{dataset_name}*tvae*{i}.csv"
        filepath = os.path.join(output_dir, filename)
        synthetic_data.to_csv(filepath, index=False)

    return synthesizer





def simulate_using_per_protein(data, output_dir, dataset_name, num_iterations=100, num_rows=None):
    """
    Simulate synthetic data using per-protein Gaussian distributions.

    This function estimates the mean and variance for each protein (column) in the dataset
    and generates synthetic data by sampling from normal distributions with those parameters.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input data where each column represents a protein
    output_dir : str
        Directory path where synthetic data files will be saved (e.g., 'mel/perprotein/')
    dataset_name : str
        Base name for the dataset (e.g., 'mel1', 'egfAKT100', 'melsmall1')
    num_iterations : int, default=100
        Number of synthetic datasets to generate
    num_rows : int, optional
        Number of rows to generate per synthetic dataset. If None, uses len(data)
    return_stats : bool, default=False
        Whether to return the means and variances along with the function result

    Returns:
    --------
    dict or None
        If return_stats=True, returns a dictionary with 'means' and 'variances'
    """
    import numpy as np
    import pandas as pd
    import os
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Estimate the mean and variance for each column (protein)
    means = data.mean()
    variances = data.var()


    # Determine number of rows to generate
    if num_rows is None:
        num_rows = len(data)

    # Generate synthetic data

    for i in range(num_iterations):
        simulated_data = pd.DataFrame()

        # Generate data for each protein/column
        for column in data.columns:
            mean = means[column]
            std_dev = np.sqrt(variances[column])
            simulated_data[column] = np.random.normal(loc=mean, scale=std_dev, size=num_rows)

        # Save to CSV
        filename = f"{dataset_name}_pp_{i}.csv"
        filepath = os.path.join(output_dir, filename)
        simulated_data.to_csv(filepath, index=False)



