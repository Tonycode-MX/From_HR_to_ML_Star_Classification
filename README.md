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

## Additional Options (One-vs-Rest Strategies)
The following models are available as optional configurations, designed to explore the impact of using a One-vs-Rest (OVR) meta-strategy on the multi-class problem.

The OVR approach decomposes the multi-class task into multiple binary classification tasks (one for each class), which can sometimes improve performance or handle imbalanced classes more effectively.

```
+--------------+-----------------+---------------+-----------------------------------------------------+
|   Model Key  | Base Classifier |    Strategy   |                       Purpose                       |
+--------------+-----------------+---------------+-----------------------------------------------------+
|   KNN_OVR    |       KNN       |  One-vs-Rest  | Applies KNN in a binary fashion for                 |
|              |                 |               | each class, potentially improving class separation. |
+--------------+-----------------+---------------+-----------------------------------------------------+
|   RF_OVR     |  Random Forest  |  One-vs-Rest  | Uses RF's robustness within the OVR framework.      |
+--------------+-----------------+---------------+-----------------------------------------------------+
|              |                 |               | Leverages SVM's precision in a multi-class scenario |
|   SVM_OVR    |       SVM       |  One-vs-Rest  | by training an independent classifier for each      |
|              |                 |               | stellar population.                                 |
+--------------+-----------------+---------------+-----------------------------------------------------+
```

KNN_OVR	KNN	One-vs-Rest	Applies KNN in a binary fashion for each class, potentially improving class separation.
RF_OVR	Random Forest	One-vs-Rest	Uses RF's robustness within the OVR framework.
SVM_OVR	SVM	One-vs-Rest	Leverages SVM's precision in a multi-class scenario by training an independent classifier for each stellar population.

## Repository Structure
```
.
├── data/           # Original data (.csv).
├── environment/    # Conda environment configuration files.
├── modules/        # Python modules and scripts. Main entry point: main.py
├── notebooks/      # Jupyter notebooks for testing and experimentation.
├── plots/          # Generated plots and visualizations.
├── .gitattributes  # Excludes \*.ipynb files from the repository's language statistics.
├── .gitignore      # Specifies files and folders to exclude from Git version control.
└── README.md       # Project documentation and usage instructions.

```

## Configuration: Pipeline Stages Description

### Stage 1: Data Importing, Cleaning, and Preprocessing

This stage handles the acquisition and initial preparation of the stellar dataset by:

1. **Importing Data**  
   - `Option 1`: Queries the Gaia DR3 archive directly using astroquery. This is the default method.
   - `Option 2`: Loads a local CSV file from the `/data` directory.

2. **Imputing Missing Values**  
   - For each column, any missing values (NaN) are replaced with the mean of that column. This helps preserve the overall statistical properties of the data while preventing errors in subsequent modeling stages.

3. **Saving Cleaned Data**  
   - The cleaned data is saved as `cleaned_data.csv` in the `/data` directory, ready for the next stages of the pipeline.

**Input:** A direct query to the Gaia DR3 database or raw CSV file (e.g., `dataset.csv`) in the `/data` folder.

**Output:** `cleaned_data.csv` in the `/data` folder.

### Stage 2: Data Classification

This stage is the core of the labeling process. It computes the necessary features from the raw Gaia data and classifies each star based on its position on the Hertzsprung-Russell (HR) diagram. The process involves:

1. **Adding Stellar Indices**  
   - Computes the color index (`BP-RP`) and the absolute magnitude (`M_G`) for each star. These indices are essential for plotting the HR diagram and classifying stars.

2. **Labeling Stars**  
   - The pipeline applies specific, physically-motivated cuts on the HR diagram to classify each star as a Main Sequence, Giant, or White Dwarf and assigns this label to a new `Target` column.

3. **Saving Classified Data**  
   - The DataFrame, now including the new color, magnitude, and `Target` columns, is saved as `classified_dataset.csv` in the `/data` directory.

**Input:** The cleaned CSV file from Stage 1 (`cleaned_data.csv`) in the `/data` folder.

**Output:** `classified_dataset.csv` in the `/data` folder.

### Stage 3: RF & REF-based Modeling
This stage focuses on feature selection and model training. It prepares the data for modeling, selects the most relevant features using two different methods, and then compares model performance using all variables versus a select subset of variables.

1. **Prepare Data for Modeling**

   - The raw data from the previous stage is processed to separate the target variable (`Target`) from the features. The data is then split into training, validation, and testing sets to ensure robust model evaluation.

2. **Feature Selection with RF & RFE**

   - The pipeline uses two powerful techniques to identify the most important features:

      - Random Forest (RF): A method that assesses the importance of each feature by measuring how much it contributes to the model's predictive power. The top K features are selected based on their importance score.

      - Recursive Feature Elimination (RFE): A method that recursively removes the least important features, trains a model on the remaining ones, and repeats the process until the desired number of top features is reached.

3. **Compare Model Performance**
   - The final step compares the performance of a model trained on all features against a model trained on the top K features selected by RFE. This validation step is crucial to confirm whether feature selection improves or maintains predictive accuracy while reducing model complexity.

**Input:** The classified CSV file from Stage 2 (`classified_dataset.csv`) in the `/data` folder.

**Output:** The following essential files are saved to the `/data` folder:

   - Six CSV files containing the split datasets: `X_train.csv`, `X_val.csv`, `X_test.csv` (Features) and `y_train.csv`}, `y_val.csv`, `y_test.csv` (Target).
   - Two PKL files storing the list of selected feature names for column filtering in subsequent stages: `topK_features_rfe.pkl` and `topK_features_rf.pkl`.

### Stage 4: Model Comparison and Selection (Validation)
This stage is dedicated to evaluating and selecting the best machine learning algorithm from a set of candidates. It leverages the pre-split validation data to make an informed decision on the final model to be tested.

1. **Load Filtered Datasets**

   - The stage begins by loading the pre-split training (`X_train.csv`, `y_train.csv`) and validation (`X_val.csv`, `y_val.csv`) datasets from the /data folder.
   - Crucially, the features are filtered upon loading using the list of the best features previously saved in ‘topK_features_rfe.pkl‘. This ensures only the optimal feature subset is used for model comparison.

2. **Model Initialization and Comparison**

   - A set of predefined machine learning models (e.g., KNN, RandomForest, SVM) is initialized.
   - Each model is trained on the filtered training set (X_train, y_train) and then immediately evaluated on the validation set (X_val, y_val).
   - The ‘compare_in_validation‘ function handles this process, tracking key performance metrics for each algorithm.

3. **Best Model Selection**

   - The algorithm that achieves the highest performance on the validation data (based on a primary metric like F1-Score or Accuracy) is selected as the optimal model for the final testing phase.
   - The name of this best-performing model is saved for use in the subsequent stage.

Input: The following files are loaded from the `/data` folder:

   - Four CSV files containing the split datasets: `X_train.csv`, `X_val.csv` (Features) and `y_train.csv`, `y_val.csv` (Target).
   - One PKL file containing the list of selected features: `topK_features_rfe.pkl`.

Output: One PKL file `best_model_name.pkl` in the `/data` folder.

### Stage 5: Final Model Retraining and Testing (Production Readiness)
This is the final stage of the modeling pipeline. Its purpose is to maximize the predictive power of the selected best model by retraining it on the combined training and validation data, then providing a final, unbiased evaluation using the test set, and finally, saving the model for deployment.

1. **Load All Filtered Datasets**

   - The stage loads all six filtered datasets: `X_train.csv`, `X_val.csv`, `X_test.csv`, and their respective target files.
   - All feature datasets (X) are strictly filtered using the feature list stored in `topK_features_rfe.pkl`.

2. **Identify, Prepare, and Retrain Best Model**

   - The name of the best-performing model, determined in Stage 4, is loaded from `best_model_name.pkl`.
   - The training and validation sets are combined (X_trfin, y_trfin) to utilize all available non-testing data for the final training phase.
   - The model is retrained on the combined data and then evaluated for the very first time on the completely unseen test set (X_test, y_test).

3. **Output and Model Persistence**

   - The `retrain_best_model` function is executed, returning the fully trained classifier object.
   - This object is immediately serialized and saved, making it ready for production use.

Input: The following files are loaded from the `/data` folder:

   - Six CSV files containing the split datasets: `X_train.csv`, `X_val.csv`, `X_test.csv` (Features) and `y_train.csv`, `y_val.csv`, `y_test.csv` (Target).
   - Two PKL files: `topK_features_rfe.pkl` (list of selected features) and `best_model_name.pkl` (name of the model to be used).

Output: The stage produces two main outputs:

   - A comprehensive performance report summarizing the final metrics (e.g., Accuracy, F1-Score) achieved on the test set.
   - One Joblib file containing the final, fully trained model: `final_production_model.joblib` in the `/data` folder. 


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

The `modules/` directory contains the main script **`main.py`**, which serves as the primary entry point for executing the analysis process. To run the script, you must first navigate to the ‘modules/‘ directory. Use the following commands in your terminal (with your Conda environment already activated):
   - 1. Navigate to the modules directory
```bash
cd modules/
```
   - 2. Run the main script
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
stages = [5] will only run Stage 5.
```