import streamlit as st
import requests
import os
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")


st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("🤖 AI Resume Analyzer")
st.write("Upload your resume and check how well it matches a job description.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description (optional)")

if st.button("Analyze Resume"):

    if resume_file:

        files = {"resume": resume_file}

        data = {}
        if job_description.strip():
            data["job_description"] = job_description

        response = requests.post(
         API_URL,
        files=files,
        data=data
        )


        result = response.json()

        st.subheader("📊 Match Result")

        st.metric("Overall Match", f"{result['overall_match_percent']}%")
        st.success(result["match_label"])

        st.subheader("✅ Found Skills")
        st.write(result["found_skills"])

        st.subheader("❌ Missing Skills")
        st.write(result["missing_skills"])

        if result["suggestions"]:
            st.subheader("💡 Suggestions")
            for tip in result["suggestions"]:
                st.write("- ", tip)

    else:
        st.error("Please upload a resume.")
