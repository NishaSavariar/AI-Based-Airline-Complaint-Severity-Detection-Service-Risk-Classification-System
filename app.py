import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
from keras.models import load_model
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from keras.preprocessing.sequence import pad_sequences

# Load data
df = pd.read_csv("data/cleaned_airline_reviews.csv")

# Load ML models
tfidf = joblib.load("models/tfidf.pkl")
severity_model = joblib.load("models/severity_lr.pkl")
risk_model = joblib.load("models/risk_rf.pkl")

# Load DL models
dl_severity_model = load_model("models/dl_severity_model.keras")
dl_risk_model = load_model("models/dl_risk_model.keras")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer_dl = pickle.load(f)

# Load Transformer
tokenizer_tr = DistilBertTokenizer.from_pretrained("models/distilbert_severity")
model_tr = DistilBertForSequenceClassification.from_pretrained("models/distilbert_severity")

st.set_page_config(page_title="Airline Complaint Severity System", layout="wide")

st.title("AI-Based Airline Complaint Severity & Risk Classification")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Introduction", "EDA Dashboard", "Prediction"]
)

# Page 1 – Introduction
if page == "Introduction":
    st.header("Project Overview")
    
    st.write("""
    This project analyzes airline customer reviews and predicts:
    - Complaint Severity (Low, Medium, High, Critical)
    - Service Risk (Risk / No Risk)
    
    Models Used:
    - Machine Learning (TF-IDF + Logistic Regression)
    - Deep Learning (BiLSTM)
    - Transformer (DistilBERT)
    """)

    st.subheader("Dataset Sample")
    st.dataframe(df.head())


# Page 2 – EDA Dashboard
elif page == "EDA Dashboard":
    st.header("Exploratory Data Analysis")

    st.subheader("Severity Distribution")
    st.bar_chart(df["severity_label"].value_counts())

    st.subheader("Risk Distribution")
    st.bar_chart(df["risk_flag"].value_counts())

    st.subheader("Review Length Distribution")
    df["review_length"] = df["full_review_text"].apply(lambda x: len(str(x).split()))
    st.bar_chart(df["review_length"])

# Page 3 – Prediction
elif page == "Prediction":
    st.header("Severity & Risk Prediction")

    # Single Review Prediction
    st.subheader("Single Review Prediction")

    review_text = st.text_area("Enter Airline Review")

    if st.button("Predict"):
        # ML Prediction
        text_tfidf = tfidf.transform([review_text])
        severity_pred_ml = severity_model.predict(text_tfidf)[0]
        risk_pred_ml = risk_model.predict(text_tfidf)[0]

        # DL Prediction
        seq = tokenizer_dl.texts_to_sequences([review_text])
        pad = pad_sequences(seq, maxlen=200)

        severity_dl = np.argmax(dl_severity_model.predict(pad))
        risk_dl = (dl_risk_model.predict(pad) > 0.5).astype(int)[0][0]

        # Transformer Prediction
        inputs = tokenizer_tr(review_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model_tr(**inputs)
        severity_tr = torch.argmax(outputs.logits).item()

        severity_map = {0:"Critical",1:"High",2:"Low",3:"Medium"}

        st.subheader("Final Prediction")

        # Use Transformer for Severity
        severity_final = severity_tr

        # Use Random Forest for Risk
        risk_final = risk_pred_ml

        severity_map = {0:"Critical",1:"High",2:"Low",3:"Medium"}

        st.write("Severity:", severity_map.get(severity_final))
        st.write("Risk:", "Risk" if risk_final==1 else "No Risk")

        st.write("Model Used:")
        st.write("Severity Model: DistilBERT Transformer")
        st.write("Risk Model: Random Forest")

    
    # CSV Bulk Prediction
    st.subheader("Bulk Prediction (Upload CSV)")
    
    uploaded_file = st.file_uploader("Upload CSV file with review column", type=["csv"])

    if uploaded_file is not None:
        bulk_df = pd.read_csv(uploaded_file)

        if "review" in bulk_df.columns:
            bulk_df["review"] = bulk_df["review"].fillna("").astype(str)
            texts = bulk_df["review"]

            # ML Predictions
            tfidf_text = tfidf.transform(texts)
            bulk_df["Severity_ML"] = severity_model.predict(tfidf_text)
            bulk_df["Risk_ML"] = risk_model.predict(tfidf_text)

            # Map labels
            severity_map = {0:"Critical", 1:"High", 2:"Low", 3:"Medium"}
            risk_map = {1:"Risk", 0:"No Risk"}

            bulk_df["Severity_Label"] = bulk_df["Severity_ML"].map(severity_map)
            bulk_df["Risk_Label"] = bulk_df["Risk_ML"].map(risk_map)

            st.write("Bulk Prediction Results")
            st.dataframe(bulk_df)

            csv = bulk_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                csv,
                "bulk_predictions.csv",
                "text/csv"
            )
            
        else:
            st.error("CSV must contain a column named 'review'")