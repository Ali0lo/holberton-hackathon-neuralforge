import re
from io import BytesIO
from typing import List, Set, Tuple

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

from roleforge.llm_helper import llm_extract_cv_skills


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


def extract_text_from_docx(file) -> str:
    doc = Document(file)
    text = []

    for para in doc.paragraphs:
        if para.text:
            text.append(para.text)

    return "\n".join(text)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_uploaded_file(file):
    file_type = file.name.split(".")[-1].lower()

    if file_type == "pdf":
        file_bytes = file.read()
        return extract_text_from_pdf(file_bytes)

    elif file_type == "docx":
        return extract_text_from_docx(file)

    elif file_type == "txt":
        return file.read().decode("utf-8", errors="ignore")

    else:
        raise ValueError("Unsupported file type")


def build_skill_aliases() -> dict:
    return {
        "python": ["python", "py", "python3"],
        "machine learning": [
            "machine learning",
            "ml",
            "supervised learning",
            "unsupervised learning",
            "reinforcement learning",
            "classical machine learning",
            "predictive modeling",
            "model training",
            "model evaluation",
        ],
        "artificial intelligence": ["artificial intelligence", "ai"],
        "deep learning": [
            "deep learning",
            "dl",
            "neural networks",
            "neural network",
            "cnn",
            "rnn",
        ],
        "large language models": [
            "large language models",
            "llm",
            "llms",
            "transformers",
            "transformer models",
            "prompt engineering",
        ],
        "natural language processing": ["natural language processing", "nlp"],
        "computer vision": ["computer vision", "cv"],
        "pytorch": ["pytorch", "torch"],
        "tensorflow": ["tensorflow", "tf", "tensor flow"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "numpy": ["numpy", "np"],
        "pandas": ["pandas", "pd"],
        "statistics": ["statistics", "stats", "stat"],
        "sql": ["sql", "postgres", "postgresql", "mysql", "databases"],
        "apis": ["api", "apis", "rest api", "rest apis"],
        "git": ["git", "github", "git/github"],
        "docker": ["docker"],
        "mlops": ["mlops", "ml ops"],
        "kubernetes": ["kubernetes", "k8s"],
        "ci/cd": ["ci/cd", "cicd", "ci cd"],
        "cloud": ["cloud", "aws", "gcp", "azure", "cloud computing", "terraform"],
        "excel": ["excel"],
        "bi tools": ["bi tools", "power bi", "tableau"],
        "dashboarding": ["dashboarding", "dashboards", "dashboard"],
        "etl": ["etl", "data pipelines"],
        "data warehousing": ["data warehousing", "data warehouse"],
        "apache spark": ["apache spark", "spark", "pyspark"],
        "airflow": ["airflow", "apache airflow"],
        "html": ["html"],
        "css": ["css"],
        "javascript": ["javascript", "js", "web development"],
        "react": ["react", "reactjs"],
        "ui/ux": ["ui/ux", "ui", "ux"],
        "java": ["java"],
        "node.js": ["node.js", "nodejs", "node"],
        "networking": ["networking", "networks"],
        "linux": ["linux"],
        "security fundamentals": [
            "security fundamentals",
            "cybersecurity",
            "cyber security",
            "network security",
            "cloud security",
            "firewalls",
            "security architecture",
            "penetration testing",
            "web security",
            "owasp",
        ],
        "siem": ["siem", "splunk", "log analysis"],
        "incident response": [
            "incident response",
            "threat analysis",
            "threat detection",
        ],
        "testing": ["testing", "software testing"],
        "automation testing": ["automation testing", "test automation"],
        "selenium": ["selenium"],
        "api testing": ["api testing", "postman"],
        "business analysis": ["business analysis"],
        "requirements gathering": [
            "requirements gathering",
            "requirements elicitation",
        ],
        "process modeling": ["process modeling", "bpmn"],
        "documentation": ["documentation", "technical writing"],
        "stakeholder communication": ["stakeholder communication"],
        "mathematics": ["mathematics", "math"],
        "research": ["research"],
        "experimentation": ["experimentation", "a/b testing", "ab testing"],
        "vector databases": [
            "vector databases",
            "vector db",
            "pinecone",
            "faiss",
            "chroma",
        ],
        "rag": ["rag", "retrieval augmented generation"],
        "data structures": ["data structures"],
        "algorithms": ["algorithms"],
        "system design": ["system design"],
        "feature engineering": ["feature engineering"],
        "data visualization": ["data visualization", "matplotlib", "seaborn", "plotly"],
        "monitoring": ["monitoring"],
    }


def extract_skill_matches_from_text(text: str, role_df: pd.DataFrame) -> List[Tuple[str, str]]:
    normalized_text = f" {_normalize_text(text)} "
    aliases = build_skill_aliases()

    dataset_skills: Set[str] = {
        str(skill).strip().lower() for skill in role_df["skill"].dropna().unique().tolist()
    }

    found = []
    seen = set()

    for skill in dataset_skills:
        candidate_terms = aliases.get(skill, [skill])

        for term in candidate_terms:
            term_norm = _normalize_text(term)
            pattern = rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])"
            if re.search(pattern, normalized_text):
                key = (skill, term)
                if key not in seen:
                    found.append((skill, term))
                    seen.add(key)
                break

    return sorted(found, key=lambda x: x[0])


def extract_skills_from_text(
    text: str,
    role_df: pd.DataFrame,
    target_role: str = "",
    use_llm: bool = False,
) -> List[str]:
    matches = extract_skill_matches_from_text(text, role_df)
    regex_skills = sorted({skill for skill, _ in matches})

    if use_llm:
        allowed_skills = sorted(
            {str(skill).strip() for skill in role_df["skill"].dropna().unique().tolist()}
        )
        llm_skills = llm_extract_cv_skills(
            cv_text=text,
            target_role=target_role,
            allowed_skills=allowed_skills,
        )
        return sorted(set(regex_skills + llm_skills))

    return regex_skills