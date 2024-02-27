import nltk
import re
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean the resume text
def clean_resume(text):
    clean_text = re.sub(r'http\S+', '', text)                   # Remove URLs
    clean_text = re.sub(r'\S*@\S*\s?', '', clean_text)          # Remove email addresses
    clean_text = re.sub(r'¬', '', clean_text)                   # Remove special characters
    clean_text = re.sub(r'\d{10}', '', clean_text)              # Remove phone numbers
    clean_text = re.sub(r'\b[^\s]{1,2}\b', '', clean_text)      # Remove short words
    clean_text = re.sub(r'\s+', ' ', clean_text)                # Remove extra whitespaces
    return clean_text.strip()

# Load the trained model and TF-IDF vectorizer
clf = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# Mapping of category IDs to category names
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Streamlit web application
st.title("RESUME SCREENING APPLICATION")
upload_file = st.file_uploader("Upload your resume below", type=['pdf', 'docx', 'txt'])

if upload_file is not None:
    try:
        resume_bytes = upload_file.read()
        resume_text = resume_bytes.decode('utf-8')
    except UnicodeDecodeError:
        resume_text = resume_bytes.decode('latin-1')

    cleaned_resume = clean_resume(resume_text)
    input_features = tfidf_vectorizer.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    category_name = category_mapping.get(prediction_id, "Unknown")
    st.write("Predicted Category:", category_name)
