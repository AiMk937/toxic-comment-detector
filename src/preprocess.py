import pandas as pd
import numpy as np
import nltk
import ssl
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Bypass SSL certificate verification for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_data(train_file):
    """Load training data from a CSV file"""
    df = pd.read_csv(train_file)
    return df

def preprocess_text(text):
    """Preprocess the comment text (remove special characters, stopwords, etc.)"""
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

def preprocess_data(df):
    """Apply preprocessing to the comment_text column"""
    df['cleaned_comment'] = df['comment_text'].apply(preprocess_text)
    return df

def split_data(df):
    """Split the dataset into training and validation sets"""
    X = df['cleaned_comment']
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def vectorize_data(X_train, X_val):
    """Vectorize the text data using TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    return X_train_tfidf, X_val_tfidf, vectorizer

if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data('data/train.csv')
    df = preprocess_data(df)
    
    # Split the data
    X_train, X_val, y_train, y_val = split_data(df)
    
    # Vectorize the data
    X_train_tfidf, X_val_tfidf, vectorizer = vectorize_data(X_train, X_val)
    
    print("Data preprocessing and vectorization complete.")
