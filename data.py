# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Set the path to the file you'd like to load
file_path = "spam.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "uciml/sms-spam-collection-dataset",
  file_path,
  pandas_kwargs={"encoding": "latin-1"},
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
#print concise summary about dataset.
df.info()

#lets delete Unnamed : 2 , Unnamed : 3 and Unnamed : 4 column because they are having zero values in almost entire column.
column_to_delete=[name for name in df.columns if name.startswith('Unnamed')]
df.drop(columns=column_to_delete,inplace=True)

#rename v1 column to target and v2 column to message
df.rename(columns=dict({"v1":"target","v2":"message"}),inplace=True)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Data Preprocessing
print("\n--- Starting Data Preprocessing ---")

# Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape[0]} rows")

# Remove extra whitespace
df['message'] = df['message'].str.strip()

# Convert to lowercase
df['message'] = df['message'].str.lower()

# Remove special characters and numbers (keep only letters and spaces)
df['message'] = df['message'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Remove extra whitespace (again)
df['message'] = df['message'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = text.split()
    return ' '.join([word for word in words if word not in stop_words])

df['message'] = df['message'].apply(remove_stopwords)

print("Preprocessing complete!")
print("\nFirst 5 preprocessed records:")
print(df.head())

# Save cleaned data
df.to_csv('cleaned_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_data.csv'")

#after deleting and renaming columns print last 5 records of the dataset
print(df.tail())

#print null values
print(df.isnull().sum())

# ============================================
# Data Transformation
# ============================================
print("\n--- Starting Data Transformation ---")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Encode target variable (ham=0, spam=1)
df['target'] = df['target'].map({'ham': 0, 'spam': 1})
print(f"Target encoding: ham=0, spam=1")
print(f"Class distribution:\n{df['target'].value_counts()}")

# TF-IDF Vectorization
print("\nApplying TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8, stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['target']

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# Train-Test Split (80% train, 20% test)
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print(f"Training set class distribution:\n{y_train.value_counts()}")
print(f"Testing set class distribution:\n{y_test.value_counts()}")

# Save transformed data and vectorizer
print("\nSaving transformed data and vectorizer...")
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

import scipy.sparse as sp
sp.save_npz('X_train.npz', X_train)
sp.save_npz('X_test.npz', X_test)

y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("✓ TF-IDF vectorizer saved to 'tfidf_vectorizer.pkl'")
print("✓ Training features saved to 'X_train.npz'")
print("✓ Testing features saved to 'X_test.npz'")
print("✓ Training labels saved to 'y_train.csv'")
print("✓ Testing labels saved to 'y_test.csv'")

print("\n--- Data Transformation Complete ---")

#print no of duplicate records
print("Total duplicated records in dataset are : {}".format(df.duplicated().sum()))

#lets remove duplicated records
df.drop_duplicates(inplace=True)
