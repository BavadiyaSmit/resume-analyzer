import streamlit as st
import requests
import os
import time

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("🤖 AI Resume Analyzer")
st.write("Upload your resume and check how well it matches a job description.")

st.caption(f"Backend: {API_URL}")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description (optional)")

if st.button("Analyze Resume"):

    if not resume_file:
        st.error("Please upload a resume.")
        st.stop()

    # Prepare multipart upload correctly
    pdf_bytes = resume_file.getvalue()
    files = {
        "resume": ("resume.pdf", pdf_bytes, "application/pdf")
    }

    data = {}
    if job_description.strip():
        data["job_description"] = job_description

    try:
        response = requests.post(API_URL, files=files, data=data, timeout=120)

        if response.status_code != 200:
            st.error(f"Backend error: {response.status_code}")
            st.text(response.text)
            st.stop()

        result = response.json()

    except Exception as e:
        st.error("Could not connect to backend.")
        st.exception(e)
        st.stop()

    # =============================
    # DISPLAY RESULTS
    # =============================

    st.subheader("📊 Final Match Result")

    st.metric("Final Match Score", f"{result['final_match_percent']}%")
    st.success(result["final_match_label"])

    # Show semantic info
    if result["semantic_available"]:
        st.info(f"Semantic Match: {result['semantic_match_percent']}%")
    else:
        st.warning("Semantic model unavailable. Using skill-based scoring.")

    st.info(f"Skill Match: {result['skill_match_percent']}%")

    st.subheader("✅ Found Skills")
    st.write(result["found_skills"])

    st.subheader("❌ Missing Skills")
    st.write(result["missing_skills"])

    if result["suggestions"]:
        st.subheader("💡 Suggestions")
        for tip in result["suggestions"]:
            st.write("•", tip)
