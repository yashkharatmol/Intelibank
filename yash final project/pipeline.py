import numpy as np
import joblib
from app.utils import extract_keywords_and_generate_label

threshold = 0.7
adb_model = joblib.load("outputs/adb_model.pkl")
vectorizer = joblib.load("outputs/vectorizer.pkl")

def combined_pipeline(input_text):
    transformed_text = vectorizer.transform([input_text])
    probabilities = adb_model.predict_proba(transformed_text)
    max_probability = np.max(probabilities)
    predicted_class = adb_model.classes_[np.argmax(probabilities)]

    if max_probability >= threshold and predicted_class != "Open Intent":
        return f"Predicted category by ADB: {predicted_class}"
    else:
        intent = extract_keywords_and_generate_label(input_text)
        return f"Detected as Open Intent. Intent: {intent}"