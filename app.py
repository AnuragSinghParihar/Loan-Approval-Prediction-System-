import io
import json
import time
import datetime
import joblib
import pandas as pd
import streamlit as st

from preprocess import preprocess

st.set_page_config(
    page_title="LoanAI — NLP Loan Approval",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f1629 50%, #0a0f1e 100%);
}

[data-testid="stSidebar"] {
    background: rgba(15, 22, 41, 0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}

.header-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 28px;
}

.header-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

.header-sub {
    color: #94a3b8;
    margin: 6px 0 0 0;
    font-size: 0.92rem;
}

.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06));
    border: 1.5px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: slideUp 0.4s ease;
}

.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(185,28,28,0.06));
    border: 1.5px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: slideUp 0.4s ease;
}

.result-label-approved {
    font-size: 2.6rem;
    font-weight: 800;
    color: #34d399;
    margin: 0;
}

.result-label-rejected {
    font-size: 2.6rem;
    font-weight: 800;
    color: #f87171;
    margin: 0;
}

.result-conf {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 6px;
}

.conf-pct-approved { color: #34d399; font-weight: 700; font-size: 1.1rem; }
.conf-pct-rejected { color: #f87171; font-weight: 700; font-size: 1.1rem; }

.risk-badge-low {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 10px;
}

.risk-badge-high {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 999px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 10px;
}

.preview-box {
    background: rgba(10,15,30,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 14px 16px;
    color: #94a3b8;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
    margin-top: 8px;
}

.history-item {
    background: rgba(21,30,54,0.7);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
}

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
}

.stat-box {
    background: rgba(21,30,54,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}

.stat-num { font-size: 1.5rem; font-weight: 700; color: #818cf8; }
.stat-lbl { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

div[data-testid="stTextArea"] textarea {
    background: rgba(21,30,54,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(21,30,54,0.6) !important;
    border: 2px dashed rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
}

div[data-testid="stTabs"] button {
    font-weight: 600 !important;
    color: #94a3b8 !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #6366f1 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #818cf8, #6366f1) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}

.stProgress > div > div { background-color: #6366f1 !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load("model/loan_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer


def rule_based_override(text: str):
    negative_keywords = [
        "default", "defaults", "poor repayment",
        "bad credit", "no income", "unemployed",
        "missed emi", "overdue", "defaulter"
    ]
    text_lower = text.lower()
    for word in negative_keywords:
        if word in text_lower:
            return "Rejected", 0.95
    return None

def predict(text: str, model, vectorizer):
    override = rule_based_override(text)
    if override:
        print(f"[DEBUG Rule Override] Triggered for text: {text[:50]}...")
        return override

    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])
    label = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = float(max(proba))
    prediction = "Approved" if label == 1 else "Rejected"
    
    # Print model thinking as requested
    print(f"[DEBUG ML Prediction] Approved: {proba[1]:.4f}, Rejected: {proba[0]:.4f}")
    
    return prediction, confidence


def show_result(prediction: str, confidence: float, preview: str = ""):
    pct = int(confidence * 100)
    approved = prediction == "Approved"

    if approved:
        st.markdown(f"""
        <div class="result-approved">
            <p class="result-label-approved">Loan Approved</p>
            <p class="result-conf">Confidence: <span class="conf-pct-approved">{pct}%</span></p>
            <span class="risk-badge-low">{'Very Low Risk' if pct >= 85 else 'Low Risk'}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <p class="result-label-rejected">Loan Rejected</p>
            <p class="result-conf">Confidence: <span class="conf-pct-rejected">{pct}%</span></p>
            <span class="risk-badge-high">{'Very High Risk' if pct >= 85 else 'High Risk'}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    conf_color = "#34d399" if approved else "#f87171"
    st.markdown('<p class="section-label">Confidence Score</p>', unsafe_allow_html=True)
    st.progress(confidence)
    st.markdown(
        f'<p style="text-align:right;color:{conf_color};font-weight:600;margin-top:-14px;">{pct}%</p>',
        unsafe_allow_html=True,
    )

    if preview:
        st.markdown('<p class="section-label">Extracted Text Preview</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="preview-box">{preview[:300]}</div>', unsafe_allow_html=True)


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("## LoanAI")
    st.markdown('<p style="color:#64748b;font-size:0.82rem;margin-top:-12px;">NLP Prediction Engine</p>', unsafe_allow_html=True)
    st.divider()

    total = len(st.session_state.history)
    approved_count = sum(1 for h in st.session_state.history if h["prediction"] == "Approved")
    rejected_count = total - approved_count

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-box"><div class="stat-num">{total}</div><div class="stat-lbl">Total</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#34d399">{approved_count}</div><div class="stat-lbl">Approved</div></div>
        <div class="stat-box"><div class="stat-num" style="color:#f87171">{rejected_count}</div><div class="stat-lbl">Rejected</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Recent History")

    if not st.session_state.history:
        st.markdown('<p style="color:#475569;font-size:0.84rem;text-align:center;padding:20px 0;">No predictions yet.</p>', unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history[-15:]):
            color = "#34d399" if item["prediction"] == "Approved" else "#f87171"
            src_label = "File" if item["source"] == "file" else "Text"
            st.markdown(f"""
            <div class="history-item">
                <span style="color:{color};font-weight:600;font-size:0.88rem;">{item['prediction']}</span>
                <span style="color:#94a3b8;font-size:0.78rem;margin-left:6px;">({item['confidence']}%)</span>
                <br>
                <span style="color:#64748b;font-size:0.74rem;">{src_label} — {item.get('label','')}</span>
                <span style="color:#334155;font-size:0.72rem;float:right;">{item['time']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, col_clr = st.columns(2)
        with col_clr:
            if st.button("Clear", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with col_dl:
            json_data = json.dumps(st.session_state.history, indent=2)
            st.download_button(
                "Export JSON",
                data=json_data,
                file_name="loan_history.json",
                mime="application/json",
                use_container_width=True,
            )

    st.divider()
    st.markdown('<p style="color:#1e293b;font-size:0.72rem;text-align:center;">Powered by scikit-learn + Streamlit</p>', unsafe_allow_html=True)


st.markdown("""
<div class="header-card">
    <p class="header-title">NLP Loan Approval Predictor</p>
    <p class="header-sub">AI-powered credit decision engine — submit a customer note or upload a document for instant analysis.</p>
</div>
""", unsafe_allow_html=True)

try:
    model, vectorizer = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Could not load model: {e}. Run `python train.py` first.")
    st.stop()

tab_text, tab_bulk = st.tabs(["Single Prediction", "Bulk CSV Upload"])

with tab_text:
    customer_note = st.text_area(
        "Customer Note",
        height=180,
        placeholder="e.g. Customer has a stable income of 80,000/month, excellent CIBIL score of 780, no outstanding debts or defaults...",
        key="text_input",
    )

    word_count = len(customer_note.split()) if customer_note.strip() else 0
    char_count = len(customer_note)
    col_wc, col_cc = st.columns([1, 1])
    col_wc.caption(f"{word_count} words")
    col_cc.caption(f"{char_count} / 2000 chars")

    if st.button("Run Prediction", key="btn_text"):
        if not customer_note.strip():
            st.warning("Please enter a customer note before submitting.")
        else:
            with st.spinner("Analyzing text with NLP model..."):
                time.sleep(0.4)
                prediction, confidence = predict(customer_note, model, vectorizer)

            show_result(prediction, confidence)

            st.session_state.history.append({
                "prediction": prediction,
                "confidence": int(confidence * 100),
                "source": "text",
                "label": "Manual text",
                "time": datetime.datetime.now().strftime("%H:%M"),
                "preview": customer_note[:200],
            })

with tab_bulk:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Upload a CSV file containing customer notes to predict eligibility for all rows at once.")
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"], label_visibility="collapsed")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        text_col = None
        for col in df.columns:
            if col.lower() in ["notes", "text", "description", "details", "remark", "remarks"]:
                text_col = col
                break
                
        if not text_col and len(df.columns) > 0:
            text_col = df.columns[-1]
            
        st.markdown("<br>", unsafe_allow_html=True)
        selected_col = st.selectbox("Select the column containing customer notes:", df.columns, index=list(df.columns).index(text_col) if text_col in df.columns else 0)
        
        if st.button("Run Bulk Prediction", key="btn_bulk"):
            with st.spinner("Running predictions on all rows. Please wait..."):
                results = []
                for idx, row in df.iterrows():
                    text_val = str(row[selected_col])
                    if not text_val or text_val.lower() == 'nan':
                        results.append("Not Eligible")
                        continue
                        
                    pred, _ = predict(text_val, model, vectorizer)
                    final_pred = "Eligible" if pred == "Approved" else "Not Eligible"
                    results.append(final_pred)
                    
                df["Status"] = results
                
            st.success(f"Analysis complete for {len(df)} records!")
            
            def highlight_status(val):
                if val == 'Eligible':
                    return 'color: #34d399; font-weight: 700'
                return 'color: #f87171; font-weight: 700'
                
            st.dataframe(df.style.map(highlight_status, subset=['Status']), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Output CSV",
                data=csv,
                file_name="loan_predictions_output.csv",
                mime="text/csv",
            )
