import streamlit as st
import joblib
import re
import spacy
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd

# APP CONFIGURATION
st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screening Tool")
st.markdown("Upload a resume (TXT file) to analyze its content and predict the best-fit job category.")

# LOAD ARTIFACTS AND RESOURCES (CACHED)
@st.cache_resource
def load_artifacts():
    """Loads the trained model and vectorizer."""
    try:
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        model = joblib.load('resume_classifier_model.joblib')
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

@st.cache_resource
def load_nlp_resources():
    """Loads NLP resources for preprocessing."""
    try:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        nlp = spacy.load('en_core_web_sm')
        return stop_words, nlp
    except Exception as e:
        st.error(f"Error loading NLP resources: {e}")
        st.stop()

# Load models and NLP tools
vectorizer, model = load_artifacts()
stop_words, nlp = load_nlp_resources()

# Dynamically get job categories from the trained model
JOB_CATEGORIES = model.classes_


# PREPROCESSING & VISUALIZATION FUNCTIONS

def preprocess_text(text):
    """Same cleaning function used during training."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    doc = nlp(text)
    tokens = [token.lemma_.strip() for token in doc if token.lemma_.strip() not in stop_words and len(token.lemma_.strip()) > 1]
    return ' '.join(tokens)

def plot_match_scores(proba_dict, selected_job):
    """Creates a styled bar plot of match scores."""
    df = pd.DataFrame.from_dict(proba_dict, orient='index', columns=['Match Score'])
    df = df.sort_values('Match Score', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df.index, df['Match Score'], color='#1f77b4', alpha=0.7)
    
    # Highlight the selected job category
    if selected_job in df.index:
        idx = list(df.index).index(selected_job)
        bars[idx].set_color('#ff7f0e')
    
    # Style the plot
    ax.set_ylabel('Match Score', fontsize=12)
    ax.set_title('Resume Match Scores by Job Category', fontsize=14, pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    return fig

# MAIN APPLICATION INTERFACE

# --- Sidebar for user input ---
with st.sidebar:
    st.subheader("🔍 Hiring Filters")
    selected_job = st.selectbox("Select the job you're hiring for:", JOB_CATEGORIES)
    st.markdown("---")
    st.info("This tool uses a trained machine learning model to classify resumes into one of several professional categories.")

# --- Main panel for file upload and results ---
uploaded_file = st.file_uploader("Choose a resume file (TXT format)", type=['txt'], accept_multiple_files=False)

if uploaded_file is not None:
    with st.spinner("Analyzing resume..."):
        try:
            # 1. Read and preprocess the text
            raw_text = uploaded_file.read().decode('utf-8')
            cleaned_text = preprocess_text(raw_text)
            
            if not cleaned_text.strip():
                st.warning("The file is empty or contains no processable text. Please try another file.")
                st.stop()

            # 2. Vectorize and Predict
            vector = vectorizer.transform([cleaned_text])
            prediction = model.predict(vector)[0]
            
            # Check if the model supports probability prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(vector)[0]
                proba_dict = dict(zip(JOB_CATEGORIES, probabilities))
            else:
                # Fallback for models without predict_proba (like default LinearSVC)
                st.warning("This model does not provide probability scores. Displaying a binary match.")
                proba_dict = {cat: 1.0 if cat == prediction else 0.0 for cat in JOB_CATEGORIES}

            # 3. Display Results
            st.header("Analysis Results")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("🔮 Best Match")
                st.success(f"**{prediction}**")
                
                confidence = proba_dict.get(prediction, 0)
                st.metric("Confidence Score", f"{confidence:.1%}")

                st.subheader("🎯 Hiring Match")
                match_score = proba_dict.get(selected_job, 0)
                st.metric(f"Fit for '{selected_job}'", f"{match_score:.1%}")
                st.progress(match_score)

            with col2:
                st.subheader("📊 Category-wise Score")
                fig = plot_match_scores(proba_dict, selected_job)
                st.pyplot(fig)

            # --- Display top matches and raw text ---
            st.markdown("---")
            sorted_matches = sorted(proba_dict.items(), key=lambda item: item[1], reverse=True)
            top_matches = sorted_matches[:3]

            st.subheader("🏆 Top 3 Matches")
            for i, (category, score) in enumerate(top_matches, 1):
                st.markdown(f"**{i}. {category}:** `{score:.1%}` confidence")
            
            with st.expander("Show Uploaded Resume Text"):
                st.text_area("", raw_text, height=300)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")