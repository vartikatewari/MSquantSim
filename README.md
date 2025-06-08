# MSquantSim
## Simulated mass spectrometry-based protein abundances enable effective experimental design

![Paper Flow Diagram from JPR Simulation](https://github.com/user-attachments/assets/bd9c5d85-a9f6-4e2b-bb8f-281c0fda8b53)

MSquantSim is a framework for generating realistic simulated mass spectrometry-based protein abundance datasets to optimize experimental design in quantitative proteomics.

## Installation
	 
```
# Clone the repository
git clone https://github.com/vartikatewari/MSquantSim.git
cd MSquantSim

# Install Dependencies from requirements.txt
pip install -r requirements.txt

```
	
## Key Features

- Realistic simulation: Preserves marginal distributions, correlations, and complex dependencies between proteins
- Robust to missing values: Effectively handles missing data common in proteomics experiments
- Technology-aware: Captures differences between acquisition technologies (DIA, MRM, etc.)
- Flexible design planning: Supports class prediction, class discovery, and class comparison objectives
- Computationally efficient: Outperforms deep generative models in efficiency and stability

## Case Studies
The repository includes three case studies demonstrating the application of MSquantSim:
### Colorectal Cancer (CRC)
- Blood-based protein markers for cancer detection
- Targeted mass spectrometry data with 67 proteins
- Low missing values and experimental validation cohort

### Pancreatic Ductal Adenocarcinoma (PDAC)
- Biomarker discovery using both DIA and MRM acquisition techniques
- 143 proteins with varying sample sizes
- Comparison between different acquisition technologies

### Single-cell Melanoma (MEL)
- Single-cell protein measurements with high missing values
- 33 proteins selected using prior knowledge network
- Demonstrates robustness to missing data

### Code Files
The repository contains several Python modules for simulating and analyzing mass spectrometry data:
downstreamanalysis.py: Implements random_forest for class prediction, PCA for discovery and p-val calculation for comparison.

evaluate.py: Contains evaluation functions:
plot_corr(): Visualizing correlation matrices
calculate_similarity_score(): calculates the similarity score

makepkn.py: Network visualization tools:
plot_indra(): Creates and visualizes Prior Knowledge Networks from INDRA TSV files
plot_learned_structure(): Plots learned network structures from CSV files

utility.py: Helper functions for data handling:
read_df(): Reads and validates CSV data files
show_protein_abundance_features(): Displays protein abundance matrix statistics

simulate.py: Contains the core simulation algorithms (copula-based, per-protein, and TVAE approaches)
## Detailed Notebooks
See the Jupyter notebooks in the notebooks/ directory for detailed examples:

- `crcdata.ipynb`: Colorectal cancer biomarker analysis
- `pdadata.ipynb`: Pancreatic cancer multi-technology analysis
- `meldata.ipynb`: Single-cell melanoma analysis with missing values

```
# Run the notebook 
jupyter-notebook notebookname
``` 
## Implementation Details
MSquantSim implements several simulation approaches:

- Copula-based simulation: Captures dependencies between proteins while maintaining marginal distributions
- Per-protein simulation: Simple baseline that models each protein independently
- Tabular Variational Autoencoder (TVAE): Deep learning approach for comparison


