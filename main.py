import pandas as pd
import re
import nltk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download stopwords if not available
nltk.download('stopwords')
from nltk.corpus import stopwords   

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("spam(2).csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})  # Convert labels to 0 and 1
    return df

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Load and preprocess data
df = load_data()
df["message"] = df["message"].apply(preprocess_text)
    
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

# Build Pipeline with TF-IDF and Naïve Bayes (Laplace Smoothing)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),  # Use unigrams and bigrams
    ("nb", MultinomialNB(alpha=1.0))  # Laplace smoothing
])

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Streamlit UI
st.title("📩 Spam Detection with Naïve Bayes")
st.write("Enter a message to check if it's spam or not.")

# Input text
user_input = st.text_area("Type your message here...")

# Predict if spam or not
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

# Display model performance
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"*Accuracy:* {accuracy:.2f}")
st.sidebar.write(f"*Precision:* {precision:.2f}")
st.sidebar.write(f"*Recall:* {recall:.2f}")
st.sidebar.write(f"*F1-Score:* {f1:.2f}")