# 🎓 Student Academic Stream Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning web application that analyzes student academic performance across multiple subjects to predict the most suitable educational stream (**Science, Commerce/Arts**, etc.) using a tuned Random Forest Classifier deployed via a Flask web server.

---

## ⚡ Project Overview

Choosing the right academic stream is a critical decision for students. This system leverages historical student marks data to build a predictive model. The workflow includes:
1. **Data Preprocessing & Feature Engineering:** Cleaning marks datasets, merging categories (e.g., Commerce & Arts), one-hot encoding categorical features like gender, and applying Min-Max scaling across all core subjects.
2. **Model Training & Hyperparameter Tuning:** Utilizing a `RandomForestClassifier` optimized via `GridSearchCV` to maximize prediction accuracy.
3. **Model Persistence:** Saving the trained model using `pickle` (`stream_predict_final.pkl`).
4. **Web Deployment:** A lightweight Flask application providing an intuitive web form for users to input grades and receive instant stream predictions.

---

## 📂 Project Structure

```text
Student-Stream-Prediction/
│
├── app.py                      # Flask web application server
├── stream_predict_final.pkl    # Serialized/Trained Random Forest model
├── studentmarksheetupdated.xlsx# Historical student dataset
├── templates/
│   ├── index.html              # Input form interface
│   └── prediction_result.html  # Result rendering template
└── README.md                   # Project documentation

```

---

## 🛠️ Tech Stack & Libraries

* **Core Language:** Python
* **Machine Learning:** Scikit-learn (`RandomForestClassifier`, `GridSearchCV`, `OneHotEncoder`, `MinMaxScaler`, train/test splitting, metrics)
* **Data Processing:** Pandas, NumPy
* **Backend & Web Framework:** Flask, Jinja2 Templates
* **Model Serialization:** Pickle

---

## 🚀 Getting Started & Local Setup

To run this application locally on your machine, follow these steps:

1. **Clone the repository:**
```bash
git clone [https://github.com/Vighnesh1045/Student-Stream-Prediction.git](https://github.com/Vighnesh1045/Student-Stream-Prediction.git)
cd Student-Stream-Prediction

```


2. **Install required dependencies:**
Make sure you have Python installed, then install the necessary libraries:
```bash
pip install pandas scikit-learn flask openpyxl

```


3. **Verify Dataset & Model Paths:**
Ensure `studentmarksheetupdated.xlsx` and `stream_predict_final.pkl` are located in your project directory (or update the file paths inside `app.py` to match your local file layout).
4. **Run the Flask Application:**
```bash
python app.py

```


5. **Access the App:**
Open your web browser and navigate to:
`http://127.0.0.1:5000/`

---

## 📊 Model Performance & Features

* **Features Evaluated:** Gender, Maths, Physics, Chemistry, English, Biology, Economics, History, and Civics.
* **Tuned Hyperparameters:** Optimized via `GridSearchCV` assessing `n_estimators`, `max_depth`, and `min_samples_split`.
* **Evaluation Metrics:** Evaluated using Accuracy Score, Confusion Matrix, Classification Report, and Feature Importance Rankings.

---

## 📫 Connect with Me

* **Portfolio:** [vighnesh1045.github.io/portfolio](https://vighnesh1045.github.io/portfolio/)
* **LinkedIn:** [linkedin.com/in/vighnesh-anant-mhatre](https://www.linkedin.com/in/vighnesh-anant-mhatre/)
* **GitHub:** [github.com/Vighnesh1045](https://www.google.com/search?q=https://github.com/Vighnesh1045)

```

```
