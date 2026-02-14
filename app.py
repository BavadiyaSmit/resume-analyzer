from fastapi import FastAPI, UploadFile, File, Form
from io import BytesIO
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI(title="Resume Analyzer Prototype")

# ==============================
# MODEL (Lazy Loaded)
# ==============================

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L3-v2"
        )
    return _model


# ==============================
# CONFIG
# ==============================

SKILLS = [
    "python", "java", "javascript", "typescript", "c", "c++", "sql",
    "html", "css", "react", "node", "fastapi", "flask", "django",
    "git", "github", "bitbucket",
    "docker", "kubernetes", "aws", "gcp",
    "machine learning", "ml", "deep learning", "dl",
    "pytorch", "tensorflow", "nlp", "computer vision", "opencv",
    "rest", "rest api", "json", "xml",
    "postgresql", "mysql"
]


SECTION_PATTERNS = {
    "skills": [
        r"skills", r"it-kenntnisse", r"programmiersprachen",
        r"webtechnologien", r"datenbanken", r"kenntnisse"
    ],
    "experience": [
        r"experience", r"projects", r"projekte",
        r"praktische erfahrung"
    ],
    "education": [
        r"education", r"studium", r"bildung", r"ausbildung"
    ],
}


# ==============================
# UTIL FUNCTIONS
# ==============================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def clean_text(t: str) -> str:
    t = (t or "").lower()
    t = t.replace("-", " ")
    t = t.replace("/", " ")
    return " ".join(t.split())


def limit_text(t: str, max_chars: int) -> str:
    return clean_text(t)[:max_chars]


def similarity(a: str, b: str) -> tuple[float, str | None]:
    """
    Returns:
        (similarity_float, error_message_or_None)
    Never crashes the API.
    """
    try:
        m = get_model()
        a = clean_text(a)
        b = clean_text(b)
        emb = m.encode([a, b], batch_size=2, show_progress_bar=False)
        sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        return sim, None
    except Exception as e:
        return 0.0, str(e)


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

    matches = []
    for sec, patterns in SECTION_PATTERNS.items():
        for p in patterns:
            m = re.search(p, t)
            if m:
                matches.append((m.start(), sec))
                break

    if not matches:
        return {"skills": t, "experience": t, "education": t}

    matches.sort()
    sections = {"skills": "", "experience": "", "education": ""}

    for i, (start, sec) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(t)
        sections[sec] += t[start:end].strip() + "\n"

    for k in sections:
        if not sections[k].strip():
            sections[k] = t

    return sections


def label_from_percent(p: float) -> str:
    if p >= 75:
        return "Strong Match"
    if p >= 55:
        return "Good Match"
    if p >= 35:
        return "Possible Fit"
    return "Needs Improvement"


# ==============================
# ROUTES
# ==============================

@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs to test the API"}


@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(None)
):
    resume_bytes = await resume.read()
    resume_text = extract_text_from_pdf(resume_bytes)
    resume_text = limit_text(resume_text, 8000)

    if not resume_text:
        return {"error": "Could not extract text from this PDF."}

    if not job_description:
        with open("sample_job.txt", "r", encoding="utf-8") as f:
            job_description = f.read()

    job_description = limit_text(job_description, 4000)

    # --------------------------
    # Skill Matching
    # --------------------------

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    missing = sorted(list(job_skills - resume_skills))
    found = sorted(list(job_skills & resume_skills))

    skill_ratio = (
        len(found) / len(job_skills) if len(job_skills) > 0 else 0.0
    )
    skill_match_percent = round(skill_ratio * 100, 2)

    # --------------------------
    # Semantic Similarity
    # --------------------------

    sim, sim_err = similarity(resume_text, job_description)
    semantic_available = sim_err is None
    semantic_percent = round(sim * 100, 2)

    # Section scores
    resume_sections = split_into_sections(resume_text)
    job_sections = split_into_sections(job_description)

    skills_sim, _ = similarity(resume_sections["skills"], job_sections["skills"])
    exp_sim, _ = similarity(resume_sections["experience"], job_sections["experience"])
    edu_sim, _ = similarity(resume_sections["education"], job_sections["education"])

    overall_sections = (skills_sim + exp_sim + edu_sim) / 3.0
    overall_match_percent = round(overall_sections * 100, 2)

    # --------------------------
    # Final Scoring (Robust)
    # --------------------------

    final_match_percent = (
        semantic_percent if semantic_available else skill_match_percent
    )

    final_match_label = label_from_percent(final_match_percent)

    # Suggestions
    suggestions = []

    if final_match_label in ["Needs Improvement", "Possible Fit"]:
        suggestions.append(
            "Add 1–2 projects that directly match the job requirements."
        )

    if missing:
        suggestions.append(
            "Try to include these keywords if you truly have them: "
            + ", ".join(missing[:6])
        )

    # --------------------------
    # RESPONSE
    # --------------------------

    return {
        "similarity_score": round(sim, 4),
        "semantic_available": semantic_available,
        "semantic_error": sim_err if not semantic_available else None,

        "skill_match_percent": skill_match_percent,
        "semantic_match_percent": semantic_percent,
        "final_match_percent": final_match_percent,
        "final_match_label": final_match_label,

        "overall_match_percent": overall_match_percent,
        "skills_match_percent": round(skills_sim * 100, 2),
        "experience_match_percent": round(exp_sim * 100, 2),
        "education_match_percent": round(edu_sim * 100, 2),

        "found_skills": found,
        "missing_skills": missing,
        "suggestions": suggestions,

        "resume_chars": len(resume_text),
        "job_chars": len(job_description),
    }
