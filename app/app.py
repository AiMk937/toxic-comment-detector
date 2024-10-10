import sys
import os
from flask import Flask, request, jsonify, render_template
import pickle

# Correct the import path for preprocess_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocess import preprocess_text

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the saved model and vectorizer
def load_model():
    model_path = '/Users/aimaankhan/ENGINEERING/SEM 7/Projects/NLP/ToxicCommentDetector/models/toxic_comment_model.pkl'
    try:
        with open(model_path, 'rb') as f:
            model, vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

model, vectorizer = load_model()

@app.route('/')
def home():
    # Serve the homepage (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions on user-provided comments."""
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(force=True)
    comments = data['comments']

    # Preprocess and vectorize the input comments
    try:
        comments_preprocessed = [preprocess_text(comment) for comment in comments]
        comments_tfidf = vectorizer.transform(comments_preprocessed)

        # Make predictions
        predictions = model.predict(comments_tfidf)
        
        # Convert predictions to list of dictionaries for each comment
        prediction_results = []
        for i, comment in enumerate(comments):
            result = {
                'comment': comment,
                'toxic': int(predictions[i][0]),
                'severe_toxic': int(predictions[i][1]),
                'obscene': int(predictions[i][2]),
                'threat': int(predictions[i][3]),
                'insult': int(predictions[i][4]),
                'identity_hate': int(predictions[i][5])
            }
            prediction_results.append(result)

        return jsonify({'predictions': prediction_results})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == "__main__":
    app.run(debug=True)
