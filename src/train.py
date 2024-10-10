import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from preprocess import load_data, preprocess_data, split_data, vectorize_data

def train_model(X_train_tfidf, y_train):
    """Train a OneVsRest Logistic Regression model for multi-label classification"""
    # Logistic Regression in OneVsRestClassifier to handle multi-label classification
    model = OneVsRestClassifier(LogisticRegression(max_iter=200))
    model.fit(X_train_tfidf, y_train)  # Fit the multi-label model on 2D target array
    return model

def evaluate_model(model, X_val_tfidf, y_val):
    """Evaluate the model on validation data"""
    y_pred = model.predict(X_val_tfidf)
    print("\nClassification Report:\n", classification_report(y_val, y_pred))

def save_model(model, vectorizer, model_path='models/toxic_comment_model.pkl'):
    """Save the trained model and vectorizer to a file"""
    with open(model_path, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data('data/train.csv')
    df = preprocess_data(df)
    
    # Split the data
    X_train, X_val, y_train, y_val = split_data(df)
    
    # Vectorize the data
    X_train_tfidf, X_val_tfidf, vectorizer = vectorize_data(X_train, X_val)
    
    # Train the model
    model = train_model(X_train_tfidf, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_val_tfidf, y_val)
    
    # Save the model
    save_model(model, vectorizer)
