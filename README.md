# Heart Failure Prediction – Logistic Regression (From Scratch)

This project predicts the likelihood of heart disease in patients using a logistic regression model built entirely from scratch using only **NumPy and pandas** — without any machine learning libraries like `scikit-learn` or `Keras`.

---

## 📊 Dataset

- Source: [Heart Failure Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Size: 918 patient records
- Target variable: `HeartDisease` (1 = disease, 0 = no disease)
- Features: Age, blood pressure, cholesterol, chest pain type, ECG results, etc.

---

## 🧪 ML Pipeline Overview

### 🔧 Preprocessing
- One-hot encoding of categorical variables (`Sex`, `ChestPainType`, etc.)
- Standardization of numeric variables using Z-score normalization
- Stratified train/validation/test split (60/20/20) using `train_test_split`

### 🧠 Model Implementation
- Logistic regression built **entirely from scratch**
- Sigmoid activation, binary cross-entropy loss
- Manual gradient descent implementation
- Training loss tracked over 1000 epochs

### 📈 Evaluation Metrics
| Dataset     | Accuracy | F1 Score |
|-------------|----------|----------|
| Validation  | 88.6%    | 89.4%    |
| Test        | 86.4%    | 87.9%    |

### 📊 Confusion Matrix (Test Set)
