# 🧠 Diabetes Prediction Web App

This is a web-based machine learning application for predicting diabetes conditions using clinical data. It classifies patients into:

- **Non-Diabetic**
- **Pre-Diabetic**
- **Diabetic**

---

## 📊 Dataset Features

The model was trained using a dataset with the following columns:

| Column | Description |
|--------|-------------|
| ID | Unique patient identifier |
| No_Pation | Patient number |
| Gender | Male/Female |
| AGE | Patient age |
| Urea | Blood urea level |
| Cr | Creatinine |
| HbA1c | Glycated hemoglobin |
| Chol | Cholesterol |
| TG | Triglycerides |
| HDL | High-density lipoprotein |
| LDL | Low-density lipoprotein |
| VLDL | Very low-density lipoprotein |
| BMI | Body Mass Index |
| CLASS | Target label (0: Non-Diabetic, 1: Pre-Diabetic, 2: Diabetic) |

---

## 🚀 Features

- Input form to submit health parameters.
- Scales and applies PCA to data before prediction.
- Predicts using three models:
  - Logistic Regression
  - SVM
  - Stacking Ensemble
- Automatically picks the **best performing model**.
- Displays **prediction results** and **model comparison graph**.

---

## 🛠️ How to Run Locally

### 1. Clone the repository or download the ZIP
```bash
git clone https://github.com/your-username/diabetes-prediction-app.git
cd diabetes-prediction-app
