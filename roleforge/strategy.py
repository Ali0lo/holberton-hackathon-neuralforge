from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SKILL_ALIASES = {
    "py": "python",
    "python3": "python",
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "machine_learning": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "deep-learning": "deep learning",
    "deep_learning": "deep learning",
    "llm": "large language models",
    "llms": "large language models",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "torch": "pytorch",
    "tf": "tensorflow",
    "tensor flow": "tensorflow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "np": "numpy",
    "pd": "pandas",
    "stats": "statistics",
    "stat": "statistics",
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "api": "apis",
    "rest api": "apis",
    "rest apis": "apis",
    "github": "git",
    "git/github": "git",
    "power bi": "bi tools",
    "tableau": "dashboarding",
}


def normalize_skill(skill: str) -> str:
    s = str(skill).strip().lower()
    s = s.replace("&", "and")
    s = " ".join(s.split())
    return SKILL_ALIASES.get(s, s)


def canonicalize_display_skill(skill: str) -> str:
    mapping = {
        "python": "Python",
        "machine learning": "Machine Learning",
        "artificial intelligence": "Artificial Intelligence",
        "deep learning": "Deep Learning",
        "large language models": "Large Language Models",
        "natural language processing": "Natural Language Processing",
        "computer vision": "Computer Vision",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "scikit-learn": "Scikit-learn",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "statistics": "Statistics",
        "sql": "SQL",
        "apis": "APIs",
        "git": "Git",
        "docker": "Docker",
        "mlops": "MLOps",
        "kubernetes": "Kubernetes",
        "ci/cd": "CI/CD",
        "cloud": "Cloud",
        "excel": "Excel",
        "bi tools": "BI Tools",
        "dashboarding": "Dashboarding",
        "etl": "ETL",
        "data warehousing": "Data Warehousing",
        "apache spark": "Apache Spark",
        "airflow": "Airflow",
        "html": "HTML",
        "css": "CSS",
        "javascript": "JavaScript",
        "react": "React",
        "ui/ux": "UI/UX",
        "java": "Java",
        "node.js": "Node.js",
        "networking": "Networking",
        "linux": "Linux",
        "security fundamentals": "Security Fundamentals",
        "siem": "SIEM",
        "incident response": "Incident Response",
        "testing": "Testing",
        "automation testing": "Automation Testing",
        "selenium": "Selenium",
        "api testing": "API Testing",
        "business analysis": "Business Analysis",
        "requirements gathering": "Requirements Gathering",
        "process modeling": "Process Modeling",
        "documentation": "Documentation",
        "stakeholder communication": "Stakeholder Communication",
        "mathematics": "Mathematics",
        "research": "Research",
        "experimentation": "Experimentation",
        "vector databases": "Vector Databases",
        "rag": "RAG",
        "data structures": "Data Structures",
        "algorithms": "Algorithms",
        "system design": "System Design",
        "feature engineering": "Feature Engineering",
        "data visualization": "Data Visualization",
    }
    return mapping.get(normalize_skill(skill), skill)


ROLE_BUCKETS = {
    "AI Engineer": {
        "foundation": ["python", "git", "apis"],
        "core": ["machine learning", "deep learning", "large language models", "natural language processing"],
        "advanced": ["pytorch", "tensorflow", "vector databases", "rag", "docker"],
    },
    "ML Engineer": {
        "foundation": ["python", "numpy", "pandas", "sql", "git"],
        "core": ["machine learning", "scikit-learn", "statistics"],
        "advanced": ["deep learning", "pytorch", "tensorflow", "mlops", "docker"],
    },
    "Deep Learning Engineer": {
        "foundation": ["python", "numpy", "git"],
        "core": ["deep learning", "pytorch", "tensorflow"],
        "advanced": ["computer vision", "natural language processing", "docker"],
    },
    "Data Scientist": {
        "foundation": ["python", "pandas", "numpy", "sql"],
        "core": ["statistics", "machine learning", "data visualization"],
        "advanced": ["scikit-learn", "feature engineering", "experimentation"],
    },
    "Data Analyst": {
        "foundation": ["sql", "excel"],
        "core": ["python", "pandas", "data visualization"],
        "advanced": ["statistics", "bi tools", "dashboarding"],
    },
    "Data Engineer": {
        "foundation": ["python", "sql", "git"],
        "core": ["etl", "data warehousing", "apache spark"],
        "advanced": ["airflow", "cloud", "docker"],
    },
    "MLOps Engineer": {
        "foundation": ["python", "git", "apis"],
        "core": ["mlops", "docker", "cloud"],
        "advanced": ["kubernetes", "ci/cd", "monitoring"],
    },
    "Software Engineer": {
        "foundation": ["python", "git", "sql"],
        "core": ["data structures", "algorithms", "apis"],
        "advanced": ["docker", "system design", "javascript"],
    },
    "Frontend Developer": {
        "foundation": ["html", "css", "javascript"],
        "core": ["react", "apis"],
        "advanced": ["git", "ui/ux"],
    },
    "Backend Developer": {
        "foundation": ["python", "sql", "git"],
        "core": ["apis", "system design"],
        "advanced": ["docker", "java", "node.js"],
    },
    "Cybersecurity Engineer": {
        "foundation": ["linux", "networking"],
        "core": ["security fundamentals", "incident response", "siem"],
        "advanced": ["python", "cloud", "git"],
    },
    "QA Engineer": {
        "foundation": ["testing", "git"],
        "core": ["automation testing", "selenium", "api testing"],
        "advanced": ["python", "ci/cd"],
    },
    "Systems Analyst": {
        "foundation": ["business analysis", "requirements gathering"],
        "core": ["process modeling", "documentation"],
        "advanced": ["stakeholder communication", "sql"],
    },
    "AI Research Scientist": {
        "foundation": ["python", "mathematics"],
        "core": ["machine learning", "deep learning", "research"],
        "advanced": ["pytorch", "experimentation", "large language models"],
    },
}


@dataclass
class StrategyResult:
    readiness_score: float
    reality_verdict: str
    fastest_role: str
    confidence: str
    estimated_months_to_ready: int
    matched_skills: List[str]
    bottlenecks: List[Tuple[str, float]]
    compressed_path: str
    what_if_projections: Dict[int, float]
    roadmap: Dict[str, List[str]]


def _get_role_df(role_df: pd.DataFrame, target_role: str) -> pd.DataFrame:
    filtered = role_df[
        role_df["role"].astype(str).str.strip().str.lower()
        == str(target_role).strip().lower()
    ].copy()

    if filtered.empty:
        raise ValueError(f"Target role '{target_role}' not found.")

    return filtered


def _bucket_based_readiness(
    role_df: pd.DataFrame,
    user_skills: List[str],
    target_role: str,
) -> Tuple[float, List[str], List[Tuple[str, float]]]:
    target_df = _get_role_df(role_df, target_role).copy()
    target_df["skill_norm"] = target_df["skill"].apply(normalize_skill)

    normalized_user_skills = {normalize_skill(skill) for skill in user_skills}

    role_weights = {normalize_skill(row["skill"]): float(row["weight"]) for _, row in target_df.iterrows()}
    role_display = {normalize_skill(row["skill"]): str(row["skill"]) for _, row in target_df.iterrows()}

    if target_role in ROLE_BUCKETS:
        buckets = ROLE_BUCKETS[target_role]
        bucket_weights = {"foundation": 0.35, "core": 0.45, "advanced": 0.20}
        score = 0.0

        for bucket_name, bucket_skills in buckets.items():
            present = [s for s in bucket_skills if s in normalized_user_skills]
            bucket_score = len(present) / max(1, len(bucket_skills))
            score += bucket_score * bucket_weights[bucket_name] * 100.0

        readiness_score = min(100.0, score)
    else:
        total_weight = float(target_df["weight"].sum())
        matched_weight = float(
            target_df[target_df["skill_norm"].isin(normalized_user_skills)]["weight"].sum()
        )
        readiness_score = 0.0 if total_weight <= 0 else (matched_weight / total_weight) * 100.0

    matched_norm = [s for s in role_weights.keys() if s in normalized_user_skills]
    matched_skills = [canonicalize_display_skill(role_display.get(s, s)) for s in matched_norm]

    missing = []
    for skill_norm, weight in role_weights.items():
        if skill_norm not in normalized_user_skills:
            missing.append((canonicalize_display_skill(role_display.get(skill_norm, skill_norm)), weight))

    missing.sort(key=lambda x: x[1], reverse=True)

    return round(readiness_score, 1), matched_skills, missing


def _build_skill_space(role_df: pd.DataFrame) -> Dict[str, int]:
    skills = sorted(role_df["skill"].astype(str).apply(normalize_skill).dropna().unique().tolist())
    return {skill: idx for idx, skill in enumerate(skills)}


def _build_user_vector(user_skills: List[str], skill_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(skill_index), dtype=float)
    for skill in user_skills:
        skill_norm = normalize_skill(skill)
        if skill_norm in skill_index:
            vec[skill_index[skill_norm]] = 1.0
    return vec


def _build_role_vector(role_df: pd.DataFrame, role_name: str, skill_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(skill_index), dtype=float)
    df = _get_role_df(role_df, role_name).copy()
    for _, row in df.iterrows():
        skill_norm = normalize_skill(row["skill"])
        if skill_norm in skill_index:
            vec[skill_index[skill_norm]] = float(row["weight"])
    return vec


def _cosine_similarity(user_vec: np.ndarray, role_vec: np.ndarray) -> float:
    user_norm = np.linalg.norm(user_vec)
    role_norm = np.linalg.norm(role_vec)
    if user_norm == 0 or role_norm == 0:
        return 0.0
    return float(np.dot(user_vec, role_vec) / (user_norm * role_norm))


def _find_fastest_role(role_df: pd.DataFrame, user_skills: List[str]) -> Tuple[str, float]:
    skill_index = _build_skill_space(role_df)
    user_vec = _build_user_vector(user_skills, skill_index)

    best_role = None
    best_score = -1.0

    for role in sorted(role_df["role"].astype(str).str.strip().unique().tolist()):
        role_vec = _build_role_vector(role_df, role, skill_index)
        sim = _cosine_similarity(user_vec, role_vec) * 100.0
        if sim > best_score:
            best_score = sim
            best_role = role

    if best_role is None:
        raise ValueError("Could not determine fastest role.")

    return best_role, round(best_score, 1)


def _estimate_months_to_ready(readiness_score: float, hours_per_week: int) -> int:
    if readiness_score >= 85:
        return 1

    gap = max(0.0, 85.0 - readiness_score)

    if hours_per_week <= 5:
        gain = 5.0
    elif hours_per_week <= 10:
        gain = 8.0
    elif hours_per_week <= 15:
        gain = 11.0
    elif hours_per_week <= 20:
        gain = 14.0
    elif hours_per_week <= 30:
        gain = 18.0
    else:
        gain = 22.0

    return max(1, int(np.ceil(gap / gain)))


def _reality_verdict(readiness_score: float, estimated_months: int) -> str:
    if readiness_score >= 75 and estimated_months <= 3:
        return "Feasible"
    if readiness_score >= 45 and estimated_months <= 6:
        return "Stretch"
    return "Unrealistic"


def _confidence_label(num_user_skills: int, num_role_skills: int) -> str:
    if num_user_skills >= 6 and num_role_skills >= 6:
        return "High"
    if num_user_skills >= 3:
        return "Medium"
    return "Low"


def _what_if_projection(readiness_score: float) -> Dict[int, float]:
    return {
        5: round(min(100.0, readiness_score + 10.0), 1),
        10: round(min(100.0, readiness_score + 20.0), 1),
        15: round(min(100.0, readiness_score + 30.0), 1),
    }


def _build_compressed_path(target_role: str, fastest_role: str) -> str:
    if fastest_role.lower() == target_role.lower():
        return target_role
    return f"{fastest_role} -> {target_role}"


def _build_roadmap(bottlenecks: List[Tuple[str, float]]) -> Dict[str, List[str]]:
    top_skills = [skill for skill, _ in bottlenecks[:3]]

    roadmap = {
        "Week 1": [],
        "Week 2": [],
        "Week 3": [],
        "Week 4": [],
    }

    if top_skills:
        roadmap["Week 1"] += [
            f"Focus on {top_skills[0]}",
            f"Complete one practical exercise in {top_skills[0]}",
        ]
    else:
        roadmap["Week 1"].append("Review fundamentals")

    if len(top_skills) >= 2:
        roadmap["Week 2"] += [
            f"Focus on {top_skills[1]}",
            f"Build one mini project using {top_skills[1]}",
        ]
    else:
        roadmap["Week 2"].append("Strengthen current skills")

    if len(top_skills) >= 3:
        roadmap["Week 3"].append(f"Focus on {top_skills[2]}")
    else:
        roadmap["Week 3"].append("Integrate two learned skills together")

    roadmap["Week 3"].append("Start a portfolio-ready project")
    roadmap["Week 4"] += [
        "Finish the project",
        "Polish GitHub README",
        "Summarize remaining gaps",
    ]

    return roadmap


def build_strategy(
    role_df: pd.DataFrame,
    user_skills: List[str],
    target_role: str,
    hours_per_week: int,
) -> StrategyResult:
    required_columns = {"role", "skill", "weight"}
    missing_columns = required_columns - set(role_df.columns)
    if missing_columns:
        raise ValueError(f"role dataframe is missing columns: {sorted(missing_columns)}")

    role_df = role_df.copy()
    role_df["role"] = role_df["role"].astype(str).str.strip()
    role_df["skill"] = role_df["skill"].astype(str).str.strip()
    role_df["weight"] = pd.to_numeric(role_df["weight"], errors="coerce").fillna(0.0)

    readiness_score, matched_skills, bottlenecks = _bucket_based_readiness(
        role_df=role_df,
        user_skills=user_skills,
        target_role=target_role,
    )

    estimated_months = _estimate_months_to_ready(readiness_score, hours_per_week)
    verdict = _reality_verdict(readiness_score, estimated_months)
    fastest_role, _ = _find_fastest_role(role_df, user_skills)
    compressed_path = _build_compressed_path(target_role, fastest_role)

    target_skill_count = len(_get_role_df(role_df, target_role))
    confidence = _confidence_label(len(user_skills), target_skill_count)

    return StrategyResult(
        readiness_score=readiness_score,
        reality_verdict=verdict,
        fastest_role=fastest_role,
        confidence=confidence,
        estimated_months_to_ready=estimated_months,
        matched_skills=matched_skills,
        bottlenecks=bottlenecks,
        compressed_path=compressed_path,
        what_if_projections=_what_if_projection(readiness_score),
        roadmap=_build_roadmap(bottlenecks),
    )