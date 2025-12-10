# Email Spam Detection using Machine Learning

A comprehensive machine learning project for classifying SMS messages as spam or ham (legitimate). This project implements an end-to-end pipeline from data acquisition through model evaluation, comparing multiple classification algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow)
- [Model Results](#model-results)
- [Skills Demonstrated](#skills-demonstrated)

## 🎯 Overview
This project tackles the problem of email/SMS spam detection using Natural Language Processing (NLP) and Machine Learning techniques. The pipeline downloads the UCI SMS Spam Collection dataset, performs comprehensive text preprocessing, applies TF-IDF vectorization, and trains multiple classification algorithms to identify spam messages.

**Key Features:**
- Automated data download from Kaggle using API
- Comprehensive text preprocessing pipeline
- TF-IDF feature extraction with optimized parameters
- Comparison of 5 different ML algorithms
- Detailed evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Stratified train-test split to handle class imbalance

## 📊 Dataset
**Source:** UCI SMS Spam Collection Dataset (via Kaggle)  
**Original Size:** 5,572 messages  
**After Deduplication:** 5,169 messages  
**Class Distribution:**
- Ham (legitimate): ~87%
- Spam: ~13%

**Dataset Characteristics:**
- Real-world SMS messages in English
- Binary classification (spam vs ham)
- Class imbalance requiring stratified sampling

## 📁 Project Structure
```
email_spam/
├── data.py                  # Data acquisition, preprocessing, and feature engineering
├── train_models.py          # Model training and evaluation
├── app.py                   # Initial imports (placeholder for future expansion)
├── requirement.txt          # Python dependencies
├── README.md               # Project documentation
├── cleaned_data.csv        # Preprocessed dataset (generated)
├── tfidf_vectorizer.pkl    # Trained TF-IDF vectorizer (generated)
├── X_train.npz             # Training features (generated)
├── X_test.npz              # Testing features (generated)
├── y_train.csv             # Training labels (generated)
└── y_test.csv              # Testing labels (generated)
```

### File Descriptions
- **`data.py`**: Complete data pipeline including:
  - Dataset download via `kagglehub` API
  - Data cleaning (removing duplicates, null values, unnamed columns)
  - Text preprocessing (lowercase, punctuation removal, stopword removal)
  - Label encoding (ham=0, spam=1)
  - TF-IDF vectorization with tuned parameters
  - Train-test split (80/20, stratified)
  - Artifact serialization for model training

- **`train_models.py`**: Model training and comparison module:
  - Loads preprocessed data artifacts
  - Trains 5 classification algorithms
  - Evaluates each model on multiple metrics
  - Generates comparison table sorted by F1-score
  - Displays detailed classification report for best model

- **`requirement.txt`**: All Python package dependencies

## 🛠 Technologies Used
**Programming Language:** Python 3.x

**Libraries:**
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn (sklearn)
- **NLP:** NLTK (Natural Language Toolkit)
- **Data Acquisition:** kagglehub
- **Text Processing:** textblob, vaderSentiment
- **Visualization:** matplotlib, seaborn, wordcloud
- **Utilities:** scipy, pickle

**Algorithms Implemented:**
1. Logistic Regression
2. Linear Support Vector Classifier (LinearSVC)
3. Multinomial Naive Bayes
4. Decision Tree Classifier
5. Random Forest Classifier

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Kaggle account (for dataset download)

### Setup Steps

1. **Clone or download the project**
```powershell
cd c:\Users\thari\Desktop\email_spam
```

2. **Create and activate a virtual environment** (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*Note: If PowerShell blocks script execution, run:*
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

3. **Install dependencies**
```powershell
pip install -r requirement.txt
```

## 🔄 Pipeline Workflow

### Step 1: Data Acquisition & Preprocessing
Run the data pipeline to download, clean, and transform the dataset:
```powershell
python data.py
```

**What this script does:**
1. **Download Dataset**: Fetches SMS Spam Collection from Kaggle using `kagglehub`
2. **Data Cleaning**:
   - Removes unnamed columns with null values
   - Drops duplicate messages (403 duplicates removed)
   - Renames columns: `v1` → `target`, `v2` → `message`
3. **Text Preprocessing**:
   - Converts all text to lowercase
   - Removes special characters, numbers, and punctuation
   - Removes English stopwords (the, a, is, etc.)
   - Strips extra whitespace
4. **Feature Engineering**:
   - Encodes labels: `ham=0`, `spam=1`
   - Applies TF-IDF vectorization (max_features=3000)
   - Creates sparse feature matrix
5. **Data Splitting**:
   - 80% training, 20% testing
   - Stratified sampling to preserve class distribution
6. **Save Artifacts**:
   - `cleaned_data.csv` - preprocessed text and labels
   - `tfidf_vectorizer.pkl` - trained vectorizer for future use
   - `X_train.npz`, `X_test.npz` - sparse feature matrices
   - `y_train.csv`, `y_test.csv` - label arrays

**Output Summary:**
- Training set: 4,135 messages
- Testing set: 1,034 messages
- Feature dimensions: 3,000 TF-IDF features

### Step 2: Model Training & Evaluation
Train and compare multiple machine learning models:
```powershell
python train_models.py
```

**What this script does:**
1. Loads preprocessed data artifacts
2. Trains 5 different classification algorithms
3. Evaluates each model on test set
4. Calculates Accuracy, Precision, Recall, F1-score
5. Generates comparison table sorted by F1-score
6. Displays detailed classification report for best model

## 📈 Model Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **LinearSVC** | **0.9768** | **0.9735** | **0.8397** | **0.9016** |
| MultinomialNB | 0.9739 | 0.9906 | 0.8015 | 0.8861 |
| RandomForest | 0.9720 | 0.9636 | 0.8092 | 0.8797 |
| DecisionTree | 0.9613 | 0.8583 | 0.8321 | 0.8450 |
| LogisticRegression | 0.9458 | 0.9870 | 0.5802 | 0.7308 |

### Best Model: Linear Support Vector Classifier (LinearSVC)

**Test Set Performance:**
- **Accuracy:** 97.68%
- **Precision:** 97.35%
- **Recall:** 83.97%
- **F1-Score:** 90.16%

**Detailed Classification Report:**
```
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       903
        spam       0.97      0.84      0.90       131

    accuracy                           0.98      1034
   macro avg       0.98      0.92      0.94      1034
weighted avg       0.98      0.98      0.98      1034
```

### Model Analysis

**Why LinearSVC Performs Best:**
- Excellent balance between precision (97.35%) and recall (83.97%)
- High accuracy (97.68%) on imbalanced dataset
- Best F1-score (90.16%) indicating overall effectiveness
- Works well with high-dimensional sparse text data (TF-IDF features)
- Effective at finding linear decision boundaries in feature space

**Model Insights:**
1. **LinearSVC**: Best overall, strong at detecting spam while minimizing false positives
2. **MultinomialNB**: Highest precision (99.06%), very few false spam predictions
3. **RandomForest**: Good balance but slightly lower recall than LinearSVC
4. **DecisionTree**: Decent performance but prone to overfitting
5. **LogisticRegression**: Lowest recall (58.02%), misses many spam messages

**Evaluation Metrics Explained:**
- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: Of messages predicted as spam, how many are actually spam
- **Recall**: Of actual spam messages, how many did we detect
- **F1-Score**: Harmonic mean of precision and recall (balanced measure)

## ⚙️ Configuration & Parameters

### TF-IDF Vectorization
```python
TfidfVectorizer(
    max_features=3000,      # Top 3000 most important words
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.8,             # Ignore words in >80% of documents
    stop_words='english'    # Remove common English words
)
```

### Train-Test Split
- **Split Ratio:** 80% training, 20% testing
- **Random State:** 42 (for reproducibility)
- **Stratification:** Yes (preserves class distribution)

### Model Hyperparameters
- **LogisticRegression:** `max_iter=200`
- **LinearSVC:** Default parameters
- **MultinomialNB:** Default parameters
- **DecisionTree:** `random_state=42`
- **RandomForest:** `n_estimators=200`, `random_state=42`

## 🎓 Skills Demonstrated

### Technical Skills
- **Python Programming**: Modular code, virtual environments, debugging
- **Data Science Libraries**: pandas, numpy, scikit-learn, NLTK
- **NLP Techniques**: Tokenization, stopword removal, TF-IDF vectorization
- **Machine Learning**: Classification algorithms, model evaluation, hyperparameters
- **Feature Engineering**: Text preprocessing, label encoding, feature extraction
- **Model Evaluation**: Multiple metrics, classification reports, model comparison
- **File I/O**: CSV, pickle serialization, sparse matrix storage

### Data Science Workflow
- End-to-end ML pipeline design
- Data preprocessing best practices
- Handling imbalanced datasets with stratification
- Model selection and comparison methodology
- Documentation and reproducibility

### Problem-Solving
- Resolving encoding issues (UTF-8 vs latin-1)
- Handling deprecated library functions
- Debugging import and dependency errors
- Optimizing TF-IDF parameters for text classification

## 📊 Sample Output

### data.py Output
```
First 5 records: DataFrame with v1, v2, and Unnamed columns
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5572 entries, 0 to 5571

--- Starting Data Preprocessing ---
After removing duplicates: 5169 rows
Preprocessing complete!

--- Starting Data Transformation ---
Target encoding: ham=0, spam=1
Class distribution:
0    4516
1     653
Name: target, dtype: int64

Feature matrix shape: (5169, 3000)
Number of features: 3000

Training set size: 4135
Testing set size: 1034
✓ All artifacts saved successfully
```

### train_models.py Output
```
LogisticRegression
Accuracy : 0.9458
Precision: 0.9870
Recall   : 0.5802
F1-score : 0.7308

LinearSVC
Accuracy : 0.9768
Precision: 0.9735
Recall   : 0.8397
F1-score : 0.9016

MultinomialNB
Accuracy : 0.9739
Precision: 0.9906
Recall   : 0.8015
F1-score : 0.8861

DecisionTree
Accuracy : 0.9613
Precision: 0.8583
Recall   : 0.8321
F1-score : 0.8450

RandomForest
Accuracy : 0.9720
Precision: 0.9636
Recall   : 0.8092
F1-score : 0.8797

=== Model Comparison ===
             model  accuracy  precision  recall     f1
         LinearSVC    0.9768     0.9735  0.8397 0.9016
     MultinomialNB    0.9739     0.9906  0.8015 0.8861
      RandomForest    0.9720     0.9636  0.8092 0.8797
      DecisionTree    0.9613     0.8583  0.8321 0.8450
LogisticRegression    0.9458     0.9870  0.5802 0.7308
```

## 🔍 Future Improvements
- Add confusion matrix visualization
- Implement cross-validation for more robust evaluation
- Try advanced models (XGBoost, Neural Networks)
- Add hyperparameter tuning (GridSearchCV)
- Create web interface for real-time spam detection
- Add more text preprocessing techniques (lemmatization, n-grams)
- Implement model deployment with Flask/FastAPI

## 🐛 Troubleshooting

### Common Issues
1. **ModuleNotFoundError**: Run `pip install -r requirement.txt`
2. **Encoding errors**: Dataset uses `latin-1` encoding (handled in code)
3. **Kaggle API errors**: Ensure kagglehub is installed
4. **PowerShell execution policy**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`

## 📚 References
- Dataset: [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Scikit-learn Documentation: [sklearn.org](https://scikit-learn.org/)
- NLTK Documentation: [nltk.org](https://www.nltk.org/)

## 📝 License
This project is for educational purposes.

## 👤 Author
Created as part of a machine learning learning project.

---

## 🚀 How to Push to GitHub

### Step 1: Create GitHub Repository
1. Go to [github.com](https://github.com) and sign in
2. Click "New repository" (green button)
3. Name it (e.g., `email-spam-detection`)
4. Keep it public or private
5. **Do NOT initialize with README** (we already have one)
6. Click "Create repository"

### Step 2: Push Code from Local
In the project folder, run:
```powershell
# Initialize git repository
git init

# Add all files
git add .

# Commit with message
git commit -m "Initial commit: Email spam detection ML pipeline"

# Add remote origin (replace with your repo URL)
git remote add origin https://github.com/<your-username>/<repo-name>.git

# Push to GitHub (use 'main' or 'master' depending on your default branch)
git branch -M main
git push -u origin main
```

### Step 3: Add .gitignore (Optional)
Create `.gitignore` to exclude unnecessary files:
```
.venv/
__pycache__/
*.pyc
*.pkl
*.npz
*.csv
.DS_Store
```

Then:
```powershell
git add .gitignore
git commit -m "Add .gitignore"
git push
```

**Note:** Replace `<your-username>` and `<repo-name>` with your actual GitHub username and repository name.

---

**Project Status:** ✅ Complete  
**Last Updated:** December 2025
