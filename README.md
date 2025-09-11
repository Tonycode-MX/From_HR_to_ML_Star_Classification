# From HR to ML: Star Classification

This project was developed for Partial 1 of the Deep Learning course as part of my academic exchange at the Universidad Católica de Colombia (Bogotá, Colombia). It aims to automate the classification of stellar populations — Main Sequence, Giants, and White Dwarfs — using data from Gaia DR3.

The goal is to build an end-to-end supervised ML pipeline for stellar classification: query and preprocess Gaia photometry/astrometry (compute BP–RP color and absolute magnitude M_G), generate labels via physically motivated HR-diagram cuts, perform feature selection (Random Forest importance + RFE), and train three models (KNN, Random Forest, SVM). The pipeline evaluates models with Accuracy, Macro-F1, ROC-AUC (OvR), and confusion matrices, and produces summary visualizations (HR diagram colored by class, feature-importance plots) to support data-driven astrophysical analysis.

**Note on Methodology:**

While the HR diagram provides a direct and physically-grounded method for stellar classification based on derived parameters, this project deliberately employs supervised machine learning models. This approach was chosen not for its practical necessity in this specific case, but as a practical exercise to apply and evaluate machine learning techniques to a dataset of personal interest. The primary goal was to demonstrate proficiency in building an end-to-end ML pipeline, not to develop a novel astrophysical classification tool.

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

3. **Support Vector Machine (SVM)**  
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

This stage handles the acquisition and initial preparation of the stellar dataset by:

1. **Importing Data**  
   - `Option 1`: Queries the Gaia DR3 archive directly using astroquery. This is the default method.
   - `Option 2`: Loads a local CSV file from the /data directory.

2. **Imputing Missing Values**  
   - For each column, any missing values (NaN) are replaced with the mean of that column. This helps preserve the overall statistical properties of the data while preventing errors in subsequent modeling stages.

3. **Saving Cleaned Data**  
   - The cleaned data is saved as `cleaned_data.csv` in the catalog directory, ready for the next stages of the pipeline.

**Input:** A direct query to the Gaia DR3 database or raw CSV file (e.g., `dataset.csv`) in the /data folder.

**Output:** `cleaned_data.csv` in the /data folder.

### Stage 2: Data Classification

This stage is the core of the labeling process. It computes the necessary features from the raw Gaia data and classifies each star based on its position on the Hertzsprung-Russell (HR) diagram. The process involves:

1. **Adding Stellar Indices**  
   - Computes the color index (`BP-RP`) and the absolute magnitude (`M_G`) for each star. These indices are essential for plotting the HR diagram and classifying stars.

2. **Labeling Stars**  
   - The pipeline applies specific, physically-motivated cuts on the HR diagram to classify each star as a Main Sequence, Giant, or White Dwarf and assigns this label to a new `Target` column.

3. **Saving Classified Data**  
   The DataFrame, now including the new color, magnitude, and `Target` columns, is saved as `classified_dataset.csv` in the /data directory.

**Input:** The cleaned CSV file from Stage 1 (`cleaned_data.csv`) in the /data folder.

**Output:** `classified_dataset.csv` in the /data folder.

### Stage 3: 

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

The input CSV file must be formatted as shown below for the model training stage. Note that the `Target` column is optional for the initial data import (Stage 1), as the pipeline can generate it in **Stage 2**. However, this column must be present in the final dataset used to train the machine learning models.
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
To build the target variable, the dataset must contain the following columns:

* `phot_g_mean_mag`
* `phot_bp_mean_mag`
* `phot_rp_mean_mag`
* `parallax`

**Note:** If your dataset already includes a target column, you should skip the execution of **Stage 2** and proceed directly to **Stage 3**.
By following this unified format, the data can be easily loaded, cleaned, and processed for analysis in the pipeline.

## Running the Main Script

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

