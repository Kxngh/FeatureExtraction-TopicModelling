# FeatureExtraction-TopicModelling
 
# Interview NLP & ML Project

## Project Overview
This project is designed for interview purposes and is divided into two main parts:

### Part A: Task Extraction and Categorization
#### Objective
Develop an NLP pipeline that processes unstructured text and extracts actionable tasks. A task is defined as a sentence describing an action that needs to be performed (e.g., *"Rahul has to buy snacks for all of us"*). Additionally, the pipeline extracts:

- **Agent:** The person or entity responsible for performing the task.
- **Deadline:** When the task is due, if mentioned (e.g., *"by 5 pm today"*).

#### Methodology
**Preprocessing:**
- Clean the text by removing punctuation, stop words, and irrelevant metadata.
- Tokenize the text into sentences.
- Use POS tagging to identify actionable verbs.

**Task Identification:**
- Apply heuristic-based rules to detect task sentences (e.g., imperative verbs, keywords like *"has to"*, *"must"*).

**Information Extraction:**
- Extract the agent using POS tagging (e.g., first proper noun in the sentence).
- Extract deadline information using regex patterns.

**Categorization:**
- Convert extracted task sentences into dense vectors using **BERT embeddings** (via the `SentenceTransformer` library).
- Cluster the tasks using **KMeans** (with 4 clusters) to dynamically discover useful categories.

#### Deliverables
- A **short video walkthrough** of the code and testing.
- Well-documented code (`Jupyter Notebook` or Python script) with modular functions for preprocessing, task extraction, and categorization.
- A **structured output (CSV file)** listing extracted tasks with details (agent, deadline, category).
- Insights and challenges faced during task implementation.

---

### Part B: Sentiment Classification of Customer Reviews
#### Objective
Develop a machine learning model to classify customer reviews (e.g., banking app reviews) as **positive** or **negative**.

#### Methodology
**Data Preprocessing:**
- Clean the review text (remove unnecessary characters, digits, punctuation, and stop words; convert to lowercase).
- Tokenize the text.

**Feature Extraction:**
- Use **TF-IDF** to convert text into numerical features.
- **Why TF-IDF?** Unlike Bag-of-Words, TF-IDF reduces the weight of common words and highlights more informative terms.

**Model Selection and Training:**
- Train a classification model (**Logistic Regression**; optionally use an ensemble with SVM, Random Forest, Gradient Boosting).
- Perform **hyperparameter tuning** (using `GridSearchCV`) and optimize recall alongside accuracy and precision.

**Evaluation:**
- Assess the model using **accuracy, precision, recall, and F1-score**.
- Save the final model and **TF-IDF vectorizer** using `joblib` for reuse.

#### Deliverables
- A **short video walkthrough** of the code and testing.
- Well-documented code (`Jupyter Notebook` or Python script) with clear modules for preprocessing, feature extraction, model training, and evaluation.
- Discussion on potential improvements (alternative models, additional data, hyperparameter tuning).

---

## Installation
Follow these steps to set up the environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/interview-nlp-ml-project.git
cd interview-nlp-ml-project

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

---

## Usage
### Running Task Extraction (Part A)
```bash
python task_extraction.py --input "sample_text.txt" --output "tasks.csv"
```

### Running Sentiment Classification (Part B)
```bash
python sentiment_analysis.py --input "reviews.csv" --output "sentiment_results.csv"
```

For detailed usage, refer to the Jupyter Notebooks in the `notebooks/` folder.

---

## Dependencies
Ensure you have the following Python packages installed:
```
numpy
pandas
scikit-learn
nltk
sentence-transformers
joblib
```


## Acknowledgments
- **NLTK** and **SentenceTransformer** for NLP processing.
- **Scikit-learn** for ML models and evaluation.
- Online NLP and ML resources that guided the development.



