import pandas as pd
import re
import nltk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download stopwords quietly
with st.spinner("Downloading NLTK stopwords..."):
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords   

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spam (2).csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "message"]
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        return df
    except FileNotFoundError:
        st.error("Dataset file 'spam (2).csv' not found. Please upload it to the app directory.")
        st.stop()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df = load_data()
df["message"] = df["message"].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("nb", MultinomialNB(alpha=1.0))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.title("📩 Spam Detection with Naïve Bayes")
st.write("Enter a message to check if it's spam or not.")

user_input = st.text_area("Type your message here...")

if st.button("Check Spam"):
    if user_input:
        processed_input = preprocess_text(user_input)
        prediction = model.predict([processed_input])[0]
        if prediction == 1:
            st.error("🚨 This is a spam message!")
        else:
            st.success("✅ This is not spam.")
    else:
        st.warning("Please enter a message.")

st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"*Accuracy:* {accuracy:.2f}")
st.sidebar.write(f"*Precision:* {precision:.2f}")
st.sidebar.write(f"*Recall:* {recall:.2f}")
st.sidebar.write(f"*F1-Score:* {f1:.2f}")
