# From HR to ML: Star Classification

This project was developed for Partial 1 of the Deep Learning course as part of my academic exchange at the Universidad Católica de Colombia (Bogotá, Colombia). It aims to automate the classification of stellar populations — Main Sequence, Giants, and White Dwarfs — using data from Gaia DR3.

The goal is to build an end-to-end supervised ML pipeline for stellar classification: query and preprocess Gaia photometry/astrometry (compute BP–RP color and absolute magnitude M_G), generate labels via physically motivated HR-diagram cuts, perform feature selection (Random Forest importance + RFE), and train three models (KNN, Random Forest, SVM). The pipeline evaluates models with Accuracy, Macro-F1, ROC-AUC (OvR), and confusion matrices, and produces summary visualizations (HR diagram colored by class, feature-importance plots) to support data-driven astrophysical analysis.

### Models Used

This project includes three supervised machine learning models for stellar classification, each with different characteristics:

1. **K-Nearest Neighbors (KNN)**  
   - Intuitive and simple to implement.
   - Classifies stars based on proximity in feature space (e.g., color BP–RP, absolute magnitude M_G). 
   - Recommended as a baseline model and for capturing local patterns in the data.

2. **Random Forest (RF)**  
   - Ensemble model based on decision trees.
   - Provides feature importance scores, useful for interpretability and feature selection.
   - Robust against noise and effective with heterogeneous astrophysical variables.

3. **Random Forest (RF)**  
   - Support Vector Machine (SVM)
   - Uses hyperplanes in high-dimensional space to separate stellar classes.
   - Well-suited for complex decision boundaries and smaller, imbalanced datasets.
   - Recommended when precision in distinguishing minority classes (e.g., Giants, White Dwarfs) is crucial.

By evaluating these three models under the same pipeline, the project compares their accuracy, Macro-F1, ROC-AUC, and confusion matrices. This offers insights into which algorithm performs best for stellar population classification while highlighting the trade-offs in interpretability, speed, and performance.


## Repository Structure
```
.
├── data/         # Original data (.csv).
├── environment/  # Conda environment configuration files.
├── modules/      # Python modules and scripts. Main entry point: main.py
├── notebooks/    # Jupyter notebooks for testing and experimentation.
├── plots/        # Generated plots and visualizations.
├── .gitignore    # Specifies files and folders to exclude from Git version control.
└── README.md     # Project documentation and usage instructions.

```

## Configuration: Pipeline Stages Description

### Stage 1: Data Importing, Cleaning, and Preprocessing 


**Input:** Raw CSV file (e.g., `.csv`)  
**Output:** 


### Stage 2: 


### Stage 3: 


**Input:** `.csv` (output from Stage 2)  
**Output:** `.csv` 


### Stage 4: 


## Conda environment setup

Inside directory `environment/` there is a file named `astrophysics.yml`. This file is used to set up a dedicated Conda environment with all the necessary dependencies for running the code in this repository.

To create the environment, first ensure you have **Anaconda** or **Miniconda** installed on your system. You can download it from [Anaconda's official website](https://www.anaconda.com/download). Then, open a terminal and run the following command:


```bash
conda env create -f astrophysics.yml
```

This command will create a new Conda environment named `astrophysics`, installing all required libraries and dependencies automatically.

#### Activating and Deactivating the Environment

Once the installation is complete, you can activate the new environment by running:


```bash
conda activate astrophysics
```

If you need to switch to another environment or deactivate it, simpy run:

```bash
conda deactivate
```

## File Format for CSV Files

The input CSV file should include the following columns:
```

+-----------+-----------+-----------+-----------+-----------+---------+
| Variable1 | Variable2 | Variable3 |    ...    | VariableK | Target  |
+-----------+-----------+-----------+-----------+-----------+---------+
|   0.82    |   4.76    |   7.13    |    ...    |   1.01    |   MS    |
|   1.35    |  -1.20    |   3.25    |    ...    |   1.02    |  Giant  |
|  -0.10    |  11.05    |  12.80    |    ...    |   1.03    |   WD    |
+-----------+-----------+-----------+-----------+-----------+---------+


```

---

By following this unified format, the data can be easily loaded, cleaned, and processed for analysis in the pipeline.


## Running the Clustering Script

The `modules/` directory contains the main script **`main.py`**, which serves as the primary entry point for executing the analysis process. To run the script, simply use the following command in the terminal (with activated conda environment):  

```bash
python main.py
```

#### Configuring Execution Stages

At the beginning of `main.py`, there is a specific line that defines which stages of the analysis process will be executed:

```python
stages = [1] will only run Stage 1.
stages = [2] will only run Stage 2.
stages = [3] will only run Stage 3.
stages = [4] will only run Stage 4.
```

### References and Citations

