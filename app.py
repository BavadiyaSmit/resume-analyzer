from fastapi import FastAPI, UploadFile, File, Form
from io import BytesIO
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Resume Analyzer Prototype")

# Multilingual model (works better for German + English)
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return model


SKILLS = [
    "python", "java", "javascript", "typescript", "c", "c++", "sql",
    "html", "css", "react", "node", "fastapi", "flask", "django",
    "git", "github", "bitbucket",
    "docker", "kubernetes", "aws", "gcp",
    "machine learning", "ml", "deep learning", "dl", "pytorch", "tensorflow",
    "nlp", "computer vision", "opencv",
    "rest", "rest api", "json", "xml",
    "postgresql", "mysql"
]

SECTION_PATTERNS = {
    "skills": [
        r"it-kenntnisse",
        r"programmiersprachen",
        r"skills",
        r"webtechnologien",
        r"datenbanken",
        r"versionsverwaltung",
        r"entwicklungsumgebung",
        r"kenntnisse",
    ],
    "experience": [
        r"praktische erfahrung",
        r"hochschul-projekte",
        r"projekte",
        r"experience",
        r"projects",
    ],
    "education": [
        r"hochschul-ausbildung",
        r"ausbildung",
        r"bildung",
        r"education",
        r"studium",
    ],
}


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def clean_text(t: str) -> str:
    # Normalize for better matching (fixes machine-learning vs machine learning)
    t = t.lower()
    t = t.replace("-", " ")
    t = t.replace("/", " ")
    return " ".join(t.split())


def similarity(a: str, b: str) -> float:
    a = clean_text(a)
    b = clean_text(b)
    m = get_model()
    emb = m.encode([a, b])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def extract_skills(text: str) -> set[str]:
    t = clean_text(text)
    found = set()
    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, t):
            found.add(skill)
    return found


def split_into_sections(text: str) -> dict:
    t = clean_text(text)

    # Find heading positions
    matches = []
    for sec, patterns in SECTION_PATTERNS.items():
        for p in patterns:
            m = re.search(p, t)
            if m:
                matches.append((m.start(), sec))
                break

    # Fallback: no headings found
    if not matches:
        return {"skills": t, "experience": t, "education": t}

    matches.sort()

    sections = {"skills": "", "experience": "", "education": ""}
    for i, (start, sec) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(t)
        sections[sec] += t[start:end].strip() + "\n"

    # If a section is still empty, fallback to full text
    for k in sections:
        if not sections[k].strip():
            sections[k] = t

    return sections


@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs to test the API"}

def label_from_percent(p: float) -> str:
    if p >= 75:
        return "Strong Match"
    if p >= 55:
        return "Good Match"
    if p >= 35:
        return "Possible Fit"
    return "Needs Improvement"


@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(None)
):
    resume_bytes = await resume.read()
    resume_text = extract_text_from_pdf(resume_bytes)

    if not resume_text:
        return {"error": "Could not extract text from this PDF (might be scanned). Try a text-based PDF."}

    # If no job description is provided, read from sample_job.txt
    if not job_description:
        with open("sample_job.txt", "r", encoding="utf-8") as f:
          job_description = f.read()

    # Overall semantic match
    sim = similarity(resume_text, job_description)

    # Skills extraction (keyword-based)
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    missing = sorted(list(job_skills - resume_skills))
    found = sorted(list(job_skills & resume_skills))

    tips = []
    if missing:
        tips.append(
            "Consider adding these skills/projects to match the job: "
            + ", ".join(missing[:8])
        )

    # Section-wise scoring (based on headings)
    resume_sections = split_into_sections(resume_text)
    job_sections = split_into_sections(job_description)

    skills_sim = similarity(resume_sections["skills"], job_sections["skills"])
    exp_sim = similarity(resume_sections["experience"], job_sections["experience"])
    edu_sim = similarity(resume_sections["education"], job_sections["education"])

    overall_sections = (skills_sim + exp_sim + edu_sim) / 3.0
    overall_percent = overall_sections * 100
    label = label_from_percent(overall_percent)

    suggestions = []
    if label in ["Needs Improvement", "Possible Fit"]:
     suggestions.append("Add 1–2 projects that directly match the job requirements.")
    if missing:
     suggestions.append("Try to include these keywords in your CV if you truly have them: " + ", ".join(missing[:6]))


    return {
        "similarity_score": round(sim, 4),
        "match_percent": round(sim * 100, 2),

        "overall_match_percent": round(overall_sections * 100, 2),
        "skills_match_percent": round(skills_sim * 100, 2),
        "experience_match_percent": round(exp_sim * 100, 2),
        "education_match_percent": round(edu_sim * 100, 2),

        "found_skills": found,
        "missing_skills": missing,
        "tips": tips,
        "match_label": label,
        "suggestions": suggestions,

        "resume_chars": len(resume_text),
        "job_chars": len(job_description),
    }
