from flask import Flask, request, jsonify, render_template
from collections import Counter
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64

# Load the trained model
model_path = 'aatmodel.pkl'
with open(model_path, 'rb') as file:
    model, X_train, y_train, scaler, pca = pickle.load(file)

# Extract individual models from the stacking ensemble
model1 = model.named_estimators_['log_reg']
model2 = model.named_estimators_['svm']

# Extract actual accuracies from training
model1_accuracy = model1.score(X_train, y_train)
model2_accuracy = model2.score(X_train, y_train)
ensemble_accuracy = model.score(X_train, y_train)

# Store accuracies in a dictionary for easier comparison
accuracies = {
    "Logistic Regression": model1_accuracy,
    "SVM": model2_accuracy,
    "Ensemble": ensemble_accuracy
}
# Determine the model with the highest accuracy
best_model_name = max(accuracies, key=accuracies.get)
best_model = {
    "Logistic Regression": model1,
    "SVM": model2,
    "Ensemble": model
}[best_model_name]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Apply scaling and PCA
    scaled_input = scaler.transform(final_features)  # Scale the input
    transformed_input = pca.transform(scaled_input)  # Apply PCA


    # If model outputs probabilities
    prediction1 = model1.predict_proba(transformed_input)[0]  # Probabilities for all classes
    prediction2 = model2.predict_proba(transformed_input)[0]  # Probabilities for all classes
    ensemble_prediction = model.predict_proba(transformed_input)[0]  # Probabilities for all classes

    # Extracting the maximum probability and predicting the class
    model1_result = np.argmax(prediction1)
    model2_result = np.argmax(prediction2)
    ensemble_result = np.argmax(ensemble_prediction)

    # Mapping numerical predictions to class labels
    class_mapping = {0: "Non-Diabetic", 1: "Pre-Diabetic", 2: "Diabetic"}

    model1_label = class_mapping[model1_result]
    model2_label = class_mapping[model2_result]
    ensemble_label = class_mapping[ensemble_result]

    # Use the best model for the final prediction
    best_model_prediction = best_model.predict(transformed_input)[0]
    best_model_label = class_mapping[best_model_prediction]

    # Display prediction results
    print(f"Model1 Prediction: {model1_label}")
    print(f"Model2 Prediction: {model2_label}")
    print(f"Ensemble Prediction: {ensemble_label}")
    print(f"Best Model ({best_model_name}) Prediction: {best_model_label}")

    fig, ax = plt.subplots()
    ax.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'purple'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_ylim([0, 1])  # Accuracy values range from 0 to 1

    # Convert the plot to a base64 string for displaying in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_image = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)  # Close the figure to free memory

    # Render the result page with predictions and graph
    return render_template(
        'comparison.html',
        model1_result=model1_label,
        model2_result=model2_label,
        ensemble_result=ensemble_label,
        best_model_name=best_model_name,
        best_model_result=best_model_label,
        graph_image=graph_image,
        model1_accuracy=model1_accuracy,
        model2_accuracy=model2_accuracy,
        ensemble_accuracy=ensemble_accuracy,
    )

if __name__ == "__main__":
    app.run(debug=True)
