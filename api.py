import os
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import preprocess
from file_handler import extract_text_from_file

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}

app = FastAPI(title="Loan Approval Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "loan_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "model", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def run_prediction(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")
    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])
    label = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = float(max(proba))
    prediction = "Approved" if label == 1 else "Rejected"
    return prediction, confidence


class TextRequest(BaseModel):
    customer_note: str


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "Loan Approval NLP Classifier"}


@app.post("/predict/text")
def predict_text(body: TextRequest):
    prediction, confidence = run_prediction(body.customer_note)
    return {"prediction": prediction, "confidence": round(confidence, 4)}


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: txt, pdf, docx",
        )

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 5 MB limit.")

    try:
        raw_text = extract_text_from_file(content, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not extract text: {exc}")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in file.")

    prediction, confidence = run_prediction(raw_text)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "extracted_text_preview": raw_text[:200].strip(),
    }
