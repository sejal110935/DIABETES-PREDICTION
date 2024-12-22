from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Load the trained model
model_path = 'finalaatmodel1.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Extract individual models from the stacking ensemble
model1 = model.named_estimators_['logreg']
model2 = model.named_estimators_['svm']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make predictions with each model
    prediction1 = model1.predict(final_features)[0]
    prediction2 = model2.predict(final_features)[0]
    ensemble_prediction = model.predict(final_features)[0]

    # Results for individual models
    model1_result = "Diabetic" if prediction1 == 1 else "Non-Diabetic"
    model2_result = "Diabetic" if prediction2 == 1 else "Non-Diabetic"
    ensemble_result = "Diabetic" if ensemble_prediction == 1 else "Non-Diabetic"

    # Simulate accuracy values or replace them with actual values from training
    accuracies = {
        "Logistic Regression": 0.85,  # Replace with actual training accuracy for model1
        "SVM": 0.88,  # Replace with actual training accuracy for model2
        "Ensemble": 0.90  # Replace with actual training accuracy for the ensemble model
    }

    # Generate accuracy comparison graph
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
        model1_result=model1_result,
        model2_result=model2_result,
        ensemble_result=ensemble_result,
        graph_image=graph_image
    )

if __name__ == "__main__":
    app.run(debug=True)
