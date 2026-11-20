import re
import joblib
import streamlit as st

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

st.title("SMS Spam Detector")
st.write("Paste a message below to check if it is spam.")

message = st.text_area("Message", height=150)

if st.button("Check"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        vec = vectorizer.transform([clean_text(message)])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]
        if pred == 1:
            st.error(f"Spam (confidence: {prob:.1%})")
        else:
            st.success(f"Not spam (spam probability: {prob:.1%})")