import re
import joblib
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


SPAM_EXAMPLE = "WINNER!! You have been selected to receive a £900 prize. Call 09061701461 to claim now"
HAM_EXAMPLE = "Hey, are we meeting at the library at 5 today?"

if "message" not in st.session_state:
    st.session_state.message = ""


def set_message(text):
    st.session_state.message = text


# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses **TF-IDF** and a **Naive Bayes** classifier "
        "trained on the SMS Spam Collection dataset."
    )
    st.subheader("Model performance")
    st.metric("Accuracy", "98.2%")
    st.metric("Precision", "96.7%")
    st.metric("Recall", "88.6%")
    st.metric("F1 score", "0.924")
    st.caption("Trained on SMS messages only, so results on emails may be less reliable.")

# ---------- Header ----------
st.title("📩 SMS Spam Detector")
st.write("Paste a message below to check whether it is spam.")
st.divider()

# ---------- Main layout ----------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Check a message")

    b1, b2 = st.columns(2)
    b1.button("Try a spam example", on_click=set_message, args=(SPAM_EXAMPLE,), use_container_width=True)
    b2.button("Try a normal example", on_click=set_message, args=(HAM_EXAMPLE,), use_container_width=True)

    message = st.text_area("Message", key="message", height=200, placeholder="Type or paste a message here...")

    check = st.button("🔍 Check message", type="primary", use_container_width=True)

    if check:
        if message.strip() == "":
            st.warning("Please enter a message.")
        else:
            vec = vectorizer.transform([clean_text(message)])
            pred = model.predict(vec)[0]
            prob = float(model.predict_proba(vec)[0][1])

            st.divider()
            if pred == 1:
                st.error("🚨 This looks like SPAM")
            else:
                st.success("✅ This looks like a normal message (Not Spam)")

            m1, m2 = st.columns(2)
            m1.metric("Spam probability", f"{prob:.1%}")
            m2.metric("Not-spam probability", f"{1 - prob:.1%}")
            st.progress(prob)

with right:
    st.subheader("How it works")
    st.markdown(
        """
        1. **Clean:** the text is lowercased and symbols are removed.
        2. **Convert:** TF-IDF turns the words into numbers.
        3. **Classify:** Naive Bayes gives a spam probability.
        4. **Decide:** above 50% is flagged as spam.
        """
    )
    st.subheader("Common spam signs")
    st.markdown(
        """
        - Words like *free*, *win*, *claim*, *prize*
        - Phone numbers and short codes
        - Urgent deadlines
        - Money amounts
        """
    )
    st.info("Tip: use the example buttons to see how the model reacts.")