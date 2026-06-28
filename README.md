# Loan Approval Prediction System

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://esaxpvs7mhxsmappyptyt8k.streamlit.app/)

An NLP-based loan approval predictor that analyzes customer notes to instantly classify applications as Approved or Rejected. Built with Scikit-learn, TF-IDF vectorization, and a Streamlit UI.

---

## Live Demo

**[https://esaxpvs7mhxsmappyptyt8k.streamlit.app/](https://esaxpvs7mhxsmappyptyt8k.streamlit.app/)**

---

## How It Works

1. Enter a customer note (e.g. "stable income of 80,000/month, CIBIL score 780, no defaults").
2. The text is preprocessed — lowercased, stemmed, stop-words removed.
3. A TF-IDF vectorizer converts the text to numeric features.
4. A Logistic Regression model predicts Approved or Rejected with a confidence score.
5. A rule-based override catches critical signals ("default", "bad credit", "unemployed") to guarantee rejection.

For bulk analysis, upload a CSV file and predictions are run on every row at once.

---

## Features

- Single text prediction with confidence score and risk badge
- Bulk CSV upload — predicts all rows and lets you download the results
- Session history in the sidebar with approve/reject counts
- Export history as JSON

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| ML Model | Scikit-learn (Logistic Regression) |
| Text Features | TF-IDF (n-grams 1–3, 1500 features) |
| NLP Preprocessing | NLTK (stemming, stop-word removal) |
| Model Serialization | Joblib |
| Data | Pandas |

---

## Project Structure

```
Loan-Approval-Prediction-System-/
├── app.py                    # Streamlit UI
├── preprocess.py             # NLP preprocessing pipeline
├── train.py                  # Model training script
├── requirements.txt
├── loan_test_data_10k.csv    # Training dataset (10K samples)
└── model/
    ├── loan_model.pkl        # Trained Logistic Regression
    └── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
```

---

## Run Locally

```bash
git clone https://github.com/AnuragSinghParihar/Loan-Approval-Prediction-System-
cd Loan-Approval-Prediction-System-
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model on the dataset:

```bash
python train.py
```

---

## Model Details

- **Algorithm:** Logistic Regression with balanced class weights
- **Features:** TF-IDF with unigrams, bigrams, trigrams (max 1500)
- **Preprocessing:** Lowercase → remove non-alpha → stop-word filter (preserving negation words: "no", "not", "bad") → Porter stemming
- **Rule override:** Hard-rejects any text containing "default", "bad credit", "unemployed", "missed emi", "overdue", etc.

---

## Author

Anurag Singh Parihar

---

## License

For educational and development purposes.
