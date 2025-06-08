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