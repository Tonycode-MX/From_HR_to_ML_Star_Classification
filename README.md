# nlp_polarity_analysis

This project was developed as part of the **Verano Delfín 2025** research program in Bogotá, Colombia. It aims to automate the analysis of public opinion expressed in online news comments using Natural Language Processing (NLP) techniques.

The main goal is to create a complete pipeline for **sentiment polarity analysis** of news-related user comments, making it easier to study trends, audience reactions, and public discourse around specific topics or headlines. The pipeline processes raw CSV data of news headlines and associated comments, cleans and tokenizes the text, classifies sentiment, and generates summary visualizations to support data-driven analysis.

### Sentiment Models Used

This project includes **two different pre-trained NLP models** for sentiment classification:

1. **Three-Category Model**  
   - Simpler and faster.  
   - Classifies comments into three polarities: **Negative**, **Neutral**, and **Positive**.  
   - Recommended for coarse-grained, high-level analysis when fine detail is not required.

2. **Five-Category Model**  
   - More fine-grained and nuanced.  
   - Classifies comments into five polarities: **Very Negative**, **Negative**, **Neutral**, **Positive**, and **Very Positive**.  
   - Recommended for detailed analysis that captures subtle variations in sentiment.

By providing both options, the pipeline offers flexibility for different research needs and levels of analysis detail. Users can choose the model best suited for their specific application or compare results across granularity levels.


## Repository Structure
```
.
├── catalog/      # Processed data files and outputs (e.g., cleaned and tokenized CSVs).
├── data/         # Original data (.csv).
├── environment/  # Conda environment configuration files.
├── modules/      # Python modules and scripts. Main entry point: run_polarity.py
├── notebooks/    # Jupyter notebooks for testing and experimentation.
├── plots/        # Generated plots and visualizations.
├── .gitignore    # Specifies files and folders to exclude from Git version control.
└── README.md     # Project documentation and usage instructions.

```


## Configuration: Pipeline Stages Description

### Stage 1: Data Importing, Cleaning, and Preprocessing 

This stage loads the raw CSV data and prepares it for analysis by:

1. **Importing Data**  
   Loads the headlines and comments from the specified CSV file.

2. **Cleaning Text**  
   - Removes non-alphanumeric characters.
   - Converts text to lowercase for consistency.
   - Applies these transformations only to text columns starting from a configurable column index.

3. **Saving Cleaned Data**  
   The cleaned data is saved as `cleaned_data.csv` in the catalog directory, ready for further processing in Stage 2.

**Input:** Raw CSV file (e.g., `DATA_PRUEBA.csv`)  
**Output:** `cleaned_data.csv`


### Stage 2: Data Tokenization

This stage processes the cleaned text data by applying two main steps:

1. **Stopwords Removal**  
   Removes common words (stopwords) in the selected language (e.g., English) from each text column starting from a configurable column index. The goal is to reduce noise and focus on meaningful words.

2. **Stemming**  
   Applies stemming to the remaining words to reduce them to their root forms (e.g., "running" → "run"). This helps standardize word variations and improve the quality of text analysis.

The result is saved as a new CSV file where each processed cell contains a list of tokenized, cleaned, and stemmed words, ready for downstream NLP tasks such as classification or topic modeling.

**Input:** `cleaned_data.csv` (output from Stage 1)  
**Output:** `tokenized_data.csv` (each cell contains a list of tokens)


### Stage 3: Data Polarization (Sentiment Analysis)

This stage takes the tokenized data and performs automated sentiment analysis using a pre-trained model. It assigns a sentiment category to each comment and summarizes the results at the news-headline level.

1. **Load Tokenized Data**  
   Loads the `tokenized_data.csv` file containing the tokenized comments.

2. **Sentiment Analysis**  
   - Applies a pre-trained NLP model to classify the sentiment of each comment.
   - Supports multiple sentiment categories (including fine-grained 5-category classification).

3. **Saving Results**  
   - Generates a summary table aggregating sentiment predictions for each news item.
   - Saves the results as `polarized_data.csv` in the catalog directory for further analysis.

**Input:** `tokenized_data.csv` (output from Stage 2)  
**Output:** `polarized_data.csv` (sentiment labels per comment, aggregated by news item)


### Stage 4: Sentiment Pie Chart Plotting

This stage visualizes the aggregated sentiment data by creating a pie chart of sentiment distribution.

1. **Load Polarized Data**  
   Loads the `polarized_data.csv` file generated in Stage 3.

2. **Compute Sentiment Totals**  
   Calculates the total count of each sentiment category across all comments.

3. **Generate and Save Pie Chart**  
   - Creates a pie chart with customizable labels and colors to represent sentiment distribution.
   - Saves the chart as a PDF file in the `plots` directory for easy reporting or presentation.

**Input:** `polarized_data.csv` (output from Stage 3)  
**Output:** `sentiment_pie_chart.pdf` (visual summary of sentiment distribution)


## Conda environment setup

Inside directory `environment/` there is a file named `polarity_analysis_env.yml`. This file is used to set up a dedicated Conda environment with all the necessary dependencies for running the code in this repository.

To create the environment, first ensure you have **Anaconda** or **Miniconda** installed on your system. You can download it from [Anaconda's official website](https://www.anaconda.com/download). Then, open a terminal and run the following command:


```bash
conda env create -f polarity_analysis_env.yml
```

This command will create a new Conda environment named `polarity_analysis_env`, installing all required libraries and dependencies automatically.

#### Activating and Deactivating the Environment

Once the installation is complete, you can activate the new environment by running:


```bash
conda activate polarity_analysis_env
```

If you need to switch to another environment or deactivate it, simpy run:

```bash
conda deactivate
```

## File Format for CSV Files

The input CSV file should include the following columns:
```

+----+----------------------------------------+-------------------------------+------------------------------+------------------------------+
| ID | Link | Headliner | Comment 1 | Comment 2 |
+----+----------------------------------------+-------------------------------+------------------------------+------------------------------+
| 1 | "https://example.com/article1" | "Breaking News: Event X!" | "Great news!" | "Can't wait for more info." |
| 2 | "https://example.com/article2" | "New Discoveries in Science" | "This is groundbreaking." | "I love this!" |
+----+----------------------------------------+-------------------------------+------------------------------+------------------------------+

```

- **ID**: A unique identifier for each entry.
- **Link**: The URL or reference link to the original article.
- **Headliner**: The text of the news headline.
- **Comment n**: Each additional column represents an individual comment associated with the headline.

---

By following this unified format, the data can be easily loaded, cleaned, and processed for analysis in the pipeline.


## Running the Clustering Script

The `modules/` directory contains the main script **`run_polarity.py`**, which serves as the primary entry point for executing the polarity analysis process. To run the script, simply use the following command in the terminal (with activated conda environment):  

```bash
python run_polarity.py
```

#### Configuring Execution Stages

At the beginning of `run_polarity.py`, there is a specific line that defines which stages of the polarity analysis process will be executed:

```python
stages = [1] will only run Stage 1 (data importing - cleaning).
stages = [2] will only run Stage 2 (tokenization).
stages = [3] will only run Stage 3 (sentiment analysis).
stages = [4] will only run Stage 4 (sentiment pie chart).
```

## Acknowledgements

We would like to express our special thanks to:

- **Ivan Toledano** (Universidad de Guadalajara, [@IvTole](https://github.com/IvTole)), for teaching us how to work in a structured and professional way with Python and Git. Your guidance was essential for organizing this project clearly and reproducibly.

- **Norvey Fonseca** (Universidad Católica de Colombia, [@norvey2019](https://github.com/norvey2019)), for proposing the original idea for the project and for providing us with the necessary material and bibliography. Your support and direction were key to developing this analysis on a solid and relevant foundation.


### References and Citations

This project uses the following pre-trained models from Hugging Face:

- [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

- [pysentimiento/robertuito-sentiment-analysis](https://huggingface.co/pysentimiento/robertuito-sentiment-analysis)

- Pérez, Juan Manuel, et al. (2021).  
  *pysentimiento: a python toolkit for opinion mining and social NLP tasks.*  
  arXiv preprint arXiv:2106.09462.  
  [arXiv Link](https://arxiv.org/abs/2106.09462)

- Pérez, Juan Manuel, Furman, Damián Ariel, Alonso Alemany, Laura, and Luque, Franco M. (2022).  
  *RoBERTuito: a pre-trained language model for social media text in Spanish.*  
  Proceedings of the Thirteenth Language Resources and Evaluation Conference, Marseille, France.  
  [ACL Anthology Link](https://aclanthology.org/2022.lrec-1.785)

- García-Vega, Manuel, et al. (2020).  
  *Overview of TASS 2020: Introducing emotion detection.*  
  Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2020) co-located with the 36th Conference of the Spanish Society for Natural Language Processing (SEPLN 2020), Málaga, Spain.
