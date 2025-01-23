import streamlit as st
import PyPDF2
from openai.embeddings_utils import cosine_similarity, get_embedding  # If using OpenAI
import openai

# (1) Set up your OpenAI API key if you’re using it
openai.api_key = "YOUR_OPENAI_API_KEY"

st.title("CV Matcher Demo")

# -------------------------------
# UPLOAD CV
# -------------------------------
uploaded_cv = st.file_uploader("Upload a CV (PDF only for now)", type=["pdf"])
# Or handle Word if you installed python-docx

# -------------------------------
# JOB DESCRIPTION INPUT
# -------------------------------
job_description = st.text_area("Paste a job description", value="Example: Looking for a data analyst...")

if uploaded_cv and job_description:
    # (A) Extract text from PDF
    reader = PyPDF2.PdfReader(uploaded_cv)
    cv_text = ""
    for page in reader.pages:
        cv_text += page.extract_text()

    # (B) Compute embeddings (if going the embeddings route)
    cv_embedding = get_embedding(cv_text, engine="text-embedding-ada-002")
    jd_embedding = get_embedding(job_description, engine="text-embedding-ada-002")

    # (C) Similarity score
    similarity_score = cosine_similarity(cv_embedding, jd_embedding)

    st.markdown(f"**Match Score**: {similarity_score:.2f} (higher is better)")

    # Optionally, give a short explanation
    # You could integrate an LLM to highlight key matching points
    # For this minimal demo, just show raw text or keywords found
    st.markdown("---")
    st.write("**Extracted CV Text (Preview):**")
    st.write(cv_text[:500] + "...")
else:
    st.write("Please upload a CV and paste a job description.")
