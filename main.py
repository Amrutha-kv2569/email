import pandas as pd
import re
import nltk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK stopwords if not already present
for resource in ['stopwords']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

from nltk.corpus import stopwords

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("spam.csv", encoding="latin-1")
        # Rename and select only relevant columns
        df = df.rename(columns={"v1": "label", "v2": "message"})
        df = df[["label", "message"]]
        df["label"] = df["label"].map({"ham": 0, "spam": 1})  # Convert labels to 0 and 1
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load data
df = load_data()
df["message"] = df["message"].apply(preprocess_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Build the model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("nb", MultinomialNB(alpha=1.0))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Streamlit App UI
st.title("📩 Spam Message Detector with Naive Bayes")
st.write("Enter a message and check if it's spam or not using ML!")

user_input = st.text_area("✉️ Type your message here:")

if st.button("🚀 Check Spam"):
    if user_input.strip() != "":
        processed = preprocess_text(user_input)
        prediction = model.predict([processed])[0]
        if prediction == 1:
            st.error("🚨 This is a spam message!")
        else:
            st.success("✅ This is not spam.")
    else:
        st.warning("⚠️ Please enter a message.")

# Sidebar - Model Performance
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"**Accuracy:** {accuracy:.2f}")
st.sidebar.write(f"**Precision:** {precision:.2f}")
st.sidebar.write(f"**Recall:** {recall:.2f}")
st.sidebar.write(f"**F1 Score:** {f1:.2f}")
