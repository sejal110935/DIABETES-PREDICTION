# 🧠 DiabML:  Diabetes Prediction Web App

This is a web-based machine learning application for predicting diabetes conditions using clinical data. It classifies patients into:

- **Non-Diabetic**
- **Pre-Diabetic**
- **Diabetic**

---

## 🧠 Why This Project?

Diabetes is a chronic and increasingly prevalent disease worldwide. Early prediction can help in timely intervention, lifestyle changes, and treatment planning — ultimately improving patient outcomes.

This application offers:

- ✅ A **simple web-based tool** for healthcare professionals or patients to assess diabetic risk.
- ✅ **Immediate, model-backed predictions** based on clinically relevant indicators.
- ✅ A **comparison of multiple ML models** to ensure reliable decision-making.

---

## 📌 Why These Features?

The selected features are **biomarkers commonly used in clinical diagnostics** of diabetes and metabolic health:

| Feature | Why it's important |
|--------|---------------------|
| AGE | Diabetes risk increases with age |
| Gender | Gender influences metabolic rates and hormonal profiles |
| HbA1c | Core marker for average blood glucose levels |
| Urea, Cr | Indicators of kidney function (affected by diabetes) |
| Chol, TG, HDL, LDL, VLDL | Lipid profile markers linked to metabolic syndrome |
| BMI | High BMI correlates with obesity-related insulin resistance |

These features were selected based on **medical literature, clinical relevance**, and **availability in real-world patient datasets**. They help capture a holistic view of a patient's metabolic and physiological state — crucial for accurate prediction of diabetes status.

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

---
![image](https://github.com/user-attachments/assets/ccba38fd-7e70-4cfd-bd76-9cd82bbced21)
![image](https://github.com/user-attachments/assets/d3893163-3ad6-416a-a5ac-e7c0e942fad8)


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

