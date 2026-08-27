import streamlit as st
import joblib
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Feedback Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------

model = joblib.load("sentiment_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

stop_words = set(stopwords.words("english"))

stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💬 Customer Feedback Sentiment Analyzer")

st.markdown(
    """
    This application uses **Natural Language Processing (NLP)** and
    **Machine Learning** to classify customer reviews as:

    **Positive • Neutral • Negative**
    """
)

st.divider()

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("Analyze a Customer Review")

review = st.text_area(
    "Enter customer review",
    height=150,
    placeholder="Example: I love this dress. The quality is excellent and it fits perfectly."
)

analyze_button = st.button(
    "Analyze Sentiment",
    type="primary"
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if analyze_button:

    if review.strip() == "":
        st.warning("Please enter a customer review before analyzing.")

    else:

        cleaned_review = clean_text(review)

        review_tfidf = tfidf.transform([cleaned_review])

        prediction = model.predict(review_tfidf)[0]

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success("😊 Positive Sentiment")

        elif prediction == "Neutral":
            st.info("😐 Neutral Sentiment")

        else:
            st.error("☹️ Negative Sentiment")

        with st.expander("View NLP preprocessing"):

            st.write("**Original review:**")
            st.write(review)

            st.write("**Cleaned review:**")
            st.code(cleaned_review)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model",
        value="Linear SVM"
    )

with col2:
    st.metric(
        label="Accuracy",
        value="80.76%"
    )

with col3:
    st.metric(
        label="Macro F1",
        value="0.61"
    )

st.caption(
    "The final model uses class-balanced Linear SVM with TF-IDF "
    "features to reduce bias toward the majority Positive class."
)

# --------------------------------------------------
# PROJECT INSIGHTS
# --------------------------------------------------

st.divider()

st.subheader("Key Customer Feedback Insights")

st.markdown(
    """
    Based on the analysis of negative customer reviews:

    - **Style & Appearance** was one of the most frequently detected issue themes.
    - **Fit & Size** appeared frequently, especially for Bottoms.
    - **Quality & Material** was another major source of dissatisfaction.
    - The **Trend** department had the highest observed negative review rate,
      although it had a relatively small number of reviews.
    """
)

# --------------------------------------------------
# PROJECT WORKFLOW
# --------------------------------------------------

st.divider()

st.subheader("How the System Works")

st.markdown(
    """
    **Customer Review**  
    ↓  
    **Text Cleaning & Lemmatization**  
    ↓  
    **TF-IDF Vectorization**  
    ↓  
    **Balanced Linear SVM**  
    ↓  
    **Sentiment Prediction**
    """
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Portfolio Project • NLP • Machine Learning • Customer Feedback Analytics"
)
