import pickle
import pandas as pd
from preprocess import preprocess_text, vectorize_data

def load_model(model_path='models/toxic_comment_model.pkl'):
    """Load the saved model and vectorizer"""
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

def predict_comments(comments, model, vectorizer):
    """Predict whether the comments are toxic or not"""
    # Preprocess and vectorize the comments
    comments_preprocessed = [preprocess_text(comment) for comment in comments]
    comments_tfidf = vectorizer.transform(comments_preprocessed)
    
    # Make predictions
    predictions = model.predict(comments_tfidf)
    return predictions

if __name__ == "__main__":
    # Load the saved model
    model, vectorizer = load_model()

    # Sample comments to predict
    sample_comments = [
        "I hate you!",
        "You are amazing, great work!",
        "This is the worst thing ever."
    ]
    
    # Make predictions
    predictions = predict_comments(sample_comments, model, vectorizer)
    
    # Display predictions
    for comment, prediction in zip(sample_comments, predictions):
        print(f"Comment: {comment}")
        print(f"Toxicity labels (0=No, 1=Yes): {prediction}")
        print("----")
