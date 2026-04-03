# Loan Approval Prediction System

## Overview

This project is an NLP-based loan approval prediction system that analyzes customer information and notes to determine whether a loan should be approved or rejected. It combines machine learning with rule-based logic to improve decision accuracy, especially for high-risk applicants.

The system supports both text input and file uploads, making it suitable for real-world financial workflows.

---

## Problem Statement

Financial institutions often rely on manual evaluation of customer profiles, which can be time-consuming and inconsistent. This project automates the process by analyzing customer notes and identifying risk patterns such as loan defaults, poor repayment history, and unstable income.

---

## Solution

The system uses Natural Language Processing (NLP) techniques along with machine learning models to classify loan applications. It also includes a rule-based layer to ensure that critical risk signals are not missed.

---

## Features

* Predict loan approval using customer notes
* Supports text input and file uploads (TXT, PDF, DOCX)
* NLP preprocessing with TF-IDF vectorization
* Multiple model training (Logistic Regression, Random Forest, Naive Bayes)
* Rule-based override for high-risk cases (e.g., loan defaults)
* Streamlit-based user interface
* FastAPI backend for scalable prediction API
* Handles real-world structured and unstructured data

---

## Tech Stack

* Python
* FastAPI
* Streamlit
* Scikit-learn
* NLTK
* Pandas, NumPy
* PyPDF2, python-docx

---

## Project Structure

```
backend/
  ├── api.py
  ├── preprocess.py
  ├── file_handler.py
  ├── model/
      ├── loan_model.pkl
      ├── tfidf_vectorizer.pkl

frontend/
  ├── app.py
```

---

## How It Works

1. User provides input via text or uploads a file.
2. FastAPI processes the request and extracts text (if file).
3. Text is cleaned and transformed using TF-IDF.
4. Machine learning model predicts approval or rejection.
5. Rule-based logic checks for high-risk keywords.
6. Result is returned to the Streamlit UI.

---

## API Endpoints

### POST /predict/text

Request:

```
{
  "customer_note": "customer has stable job and good salary"
}
```

Response:

```
{
  "prediction": "Approved",
  "confidence": 0.87
}
```

---

### POST /predict/file

* Accepts TXT, PDF, DOCX files
* Extracts text and returns prediction with preview

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/AnuragSinghParihar/Loan-Approval-Prediction-System-
cd Loan-Approval-Prediction-System-
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Application

### Start FastAPI backend

```
uvicorn api:app --reload
```

### Start Streamlit frontend

```
streamlit run app.py
```

---

## Model Details

* TF-IDF Vectorizer (n-grams up to 3)
* Logistic Regression (primary model)
* Class balancing applied
* Dataset enhanced with strong negative financial patterns

---

## Evaluation

The model is evaluated using:

* Accuracy score
* Classification report (precision, recall, F1-score)
* Confusion matrix

Special focus is given to correctly identifying high-risk (rejected) cases.

---

## Future Improvements

* Integration with structured financial data (income, credit score)
* Transformer-based models (BERT)
* Risk scoring system (0–100)
* Explainable AI for decision transparency
* Deployment on cloud infrastructure

---

## Author

Anurag Singh Parihar

---

## License

This project is for educational and development purposes.
