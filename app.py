# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# 1. Load model and vectorizer
# -----------------------------
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# 2. Streamlit page config
# -----------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection System")
st.markdown("""
Enter any news headline or article, or select one of the **latest 20 news titles** from the dataset to check if it's **True or Fake**.
""")

# -----------------------------
# 3. Function to load latest titles
# -----------------------------
def load_latest_titles():
    csv_path = "main_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "title" in df.columns:
            return df["title"].dropna().astype(str).tail(20).tolist()
    return []

# -----------------------------
# 4. Session state for titles
# -----------------------------
if "latest_titles" not in st.session_state:
    st.session_state.latest_titles = load_latest_titles()

# -----------------------------
# 5. Refresh button
# -----------------------------
if st.button("🔄 Refresh News List"):
    st.session_state.latest_titles = load_latest_titles()
    st.success("News list refreshed!")

# -----------------------------
# 6. Dropdown for latest news
# -----------------------------
selected_news = st.selectbox(
    "Select one of the latest 20 news titles:",
    options=[""] + st.session_state.latest_titles
)

# -----------------------------
# 7. User input (manual entry)
# -----------------------------
news_input = st.text_area(
    "Or type/paste your own news here:",
    value="",  # always empty
    height=150
)

# Final input decision
final_news = news_input if news_input.strip() else selected_news

# -----------------------------
# 8. Prediction
# -----------------------------
if st.button("Check News"):
    if final_news.strip() == "":
        st.warning("Please enter or select a news to check!")
    else:
        news_vector = tfidf.transform([final_news])
        prediction = model.predict(news_vector)[0]
        probability = model.predict_proba(news_vector)[0]

        if prediction == 1:
            st.success(f"✅ This news is likely **True** (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.error(f"❌ This news is likely **Fake** (Confidence: {probability[0]*100:.2f}%)")
