import re
from io import BytesIO
from typing import List, Set

import pandas as pd
from PyPDF2 import PdfReader
import docx


def _normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9+#.\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    document = docx.Document(bio)
    lines = [p.text for p in document.paragraphs if p.text]
    return "\n".join(lines)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_uploaded_file(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    if file_name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    if file_name.endswith(".txt"):
        return extract_text_from_txt(file_bytes)

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def build_skill_aliases() -> dict:
    return {
        "python": ["python", "py", "python3"],
        "machine learning": ["machine learning", "ml"],
        "artificial intelligence": ["artificial intelligence", "ai"],
        "deep learning": ["deep learning", "dl"],
        "large language models": ["large language models", "llm", "llms"],
        "natural language processing": ["natural language processing", "nlp"],
        "computer vision": ["computer vision", "cv"],
        "pytorch": ["pytorch", "torch"],
        "tensorflow": ["tensorflow", "tf", "tensor flow"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "numpy": ["numpy", "np"],
        "pandas": ["pandas", "pd"],
        "statistics": ["statistics", "stats", "stat"],
        "sql": ["sql", "postgres", "postgresql", "mysql"],
        "apis": ["api", "apis", "rest api", "rest apis"],
        "git": ["git", "github", "git/github"],
        "docker": ["docker"],
        "mlops": ["mlops", "ml ops"],
        "kubernetes": ["kubernetes", "k8s"],
        "ci/cd": ["ci/cd", "cicd", "ci cd"],
        "cloud": ["cloud", "aws", "gcp", "azure"],
        "excel": ["excel"],
        "bi tools": ["bi tools", "power bi", "tableau"],
        "dashboarding": ["dashboarding", "dashboards", "dashboard"],
        "etl": ["etl"],
        "data warehousing": ["data warehousing", "data warehouse"],
        "apache spark": ["apache spark", "spark", "pyspark"],
        "airflow": ["airflow", "apache airflow"],
        "html": ["html"],
        "css": ["css"],
        "javascript": ["javascript", "js"],
        "react": ["react", "reactjs"],
        "ui/ux": ["ui/ux", "ui", "ux"],
        "java": ["java"],
        "node.js": ["node.js", "nodejs", "node"],
        "networking": ["networking", "networks"],
        "linux": ["linux"],
        "security fundamentals": ["security fundamentals", "cybersecurity", "cyber security"],
        "siem": ["siem", "splunk"],
        "incident response": ["incident response"],
        "testing": ["testing", "software testing"],
        "automation testing": ["automation testing", "test automation"],
        "selenium": ["selenium"],
        "api testing": ["api testing", "postman"],
        "business analysis": ["business analysis"],
        "requirements gathering": ["requirements gathering", "requirements elicitation"],
        "process modeling": ["process modeling", "bpmn"],
        "documentation": ["documentation", "technical writing"],
        "stakeholder communication": ["stakeholder communication"],
        "mathematics": ["mathematics", "math"],
        "research": ["research"],
        "experimentation": ["experimentation", "a/b testing", "ab testing"],
        "vector databases": ["vector databases", "vector db", "pinecone", "faiss", "chroma"],
        "rag": ["rag", "retrieval augmented generation"],
        "data structures": ["data structures"],
        "algorithms": ["algorithms"],
        "system design": ["system design"],
        "feature engineering": ["feature engineering"],
        "data visualization": ["data visualization", "matplotlib", "seaborn", "plotly"],
    }


def extract_skills_from_text(text: str, role_df: pd.DataFrame) -> List[str]:
    normalized_text = f" {_normalize_text(text)} "
    aliases = build_skill_aliases()

    dataset_skills: Set[str] = {
        str(skill).strip().lower() for skill in role_df["skill"].dropna().unique().tolist()
    }

    found = set()

    for skill in dataset_skills:
        candidate_terms = aliases.get(skill, [skill])

        for term in candidate_terms:
            term_norm = _normalize_text(term)
            pattern = rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])"
            if re.search(pattern, normalized_text):
                found.add(skill)
                break

    return sorted(found)