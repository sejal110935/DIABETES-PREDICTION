from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'finalaatmodel.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)[0]

        # Return the prediction result
    if prediction == 1:
            result = "Diabetic"
    else:
            result = "Non-Diabetic"

    return render_template('index.html', prediction_text=f"Prediction: {result}")
    


if __name__ == "__main__":
    app.run(debug=True)