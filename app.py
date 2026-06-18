import os
import streamlit as st
import fitz
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""

    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    for page in pdf:
        text += page.get_text()

    return text


# Analyze Resume using Groq
def analyze_resume(resume_text):

    prompt = f"""
    Analyze the following resume and provide:

    1. Candidate Summary
    2. Technical Skills
    3. Strengths
    4. Weaknesses
    5. Missing Skills
    6. ATS Score out of 100
    7. Suggestions to Improve

    Resume:

    {resume_text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


# Streamlit UI
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄"
)

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume PDF and get AI feedback.")

uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

if uploaded_file is not None:

    with st.spinner("Reading Resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    st.success("Resume uploaded successfully!")

    if st.button("Analyze Resume"):

        with st.spinner("Analyzing using Groq AI..."):
            result = analyze_resume(resume_text)

        st.subheader("Analysis Result")
        st.write(result)