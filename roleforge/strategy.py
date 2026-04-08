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
    "dl": "deep learning",
    "deep-learning": "deep learning",
    "deep_learning": "deep learning",
    "stats": "statistics",
    "stat": "statistics",
    "sql db": "sql",
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "viz": "data visualization",
    "visualization": "data visualization",
    "data viz": "data visualization",
    "dataviz": "data visualization",
    "api": "apis",
    "rest api": "apis",
    "rest apis": "apis",
    "git/github": "git",
    "github": "git",
    "llm": "large language models",
    "llms": "large language models",
    "nlp": "natural language processing",
    "cv": "computer vision",
}


def normalize_skill(skill: str) -> str:
    s = str(skill).strip().lower()
    s = s.replace("&", "and")
    s = " ".join(s.split())
    return SKILL_ALIASES.get(s, s)


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
        role_df["role"].astype(str).str.strip().str.lower() == str(target_role).strip().lower()
    ].copy()
    if filtered.empty:
        raise ValueError(f"Target role '{target_role}' not found.")
    return filtered


def _calculate_weighted_readiness(
    role_df: pd.DataFrame,
    user_skills: List[str],
    target_role: str,
) -> Tuple[float, List[str], List[Tuple[str, float]]]:
    target_df = _get_role_df(role_df, target_role).copy()

    normalized_user_skills = {normalize_skill(skill) for skill in user_skills}
    target_df["skill_norm"] = target_df["skill"].apply(normalize_skill)

    total_weight = float(target_df["weight"].sum())
    if total_weight <= 0:
        return 0.0, [], []

    matched = target_df[target_df["skill_norm"].isin(normalized_user_skills)].copy()
    missing = target_df[~target_df["skill_norm"].isin(normalized_user_skills)].copy()

    readiness_score = float(matched["weight"].sum()) / total_weight * 100.0
    matched_skills = matched["skill"].astype(str).tolist()

    missing = missing.sort_values(by="weight", ascending=False)
    bottlenecks = [(str(row["skill"]), float(row["weight"])) for _, row in missing.iterrows()]

    return readiness_score, matched_skills, bottlenecks


def _cosine_similarity(user_vec: np.ndarray, role_vec: np.ndarray) -> float:
    user_norm = np.linalg.norm(user_vec)
    role_norm = np.linalg.norm(role_vec)
    if user_norm == 0 or role_norm == 0:
        return 0.0
    return float(np.dot(user_vec, role_vec) / (user_norm * role_norm))


def _build_skill_space(role_df: pd.DataFrame) -> Dict[str, int]:
    skills = role_df["skill"].astype(str).apply(normalize_skill).dropna().unique().tolist()
    skills = sorted(skills)
    return {skill: idx for idx, skill in enumerate(skills)}


def _build_user_vector(user_skills: List[str], skill_index: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(skill_index), dtype=float)
    for skill in user_skills:
        norm = normalize_skill(skill)
        if norm in skill_index:
            vec[skill_index[norm]] = 1.0
    return vec


def _build_role_vector(
    role_df: pd.DataFrame,
    role_name: str,
    skill_index: Dict[str, int],
) -> np.ndarray:
    vec = np.zeros(len(skill_index), dtype=float)
    df = _get_role_df(role_df, role_name).copy()
    df["skill_norm"] = df["skill"].apply(normalize_skill)

    for _, row in df.iterrows():
        skill = row["skill_norm"]
        weight = float(row["weight"])
        if skill in skill_index:
            vec[skill_index[skill]] = weight

    return vec


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

    estimated = int(np.ceil(gap / gain))
    return max(1, estimated)


def _reality_verdict(readiness_score: float, estimated_months: int) -> str:
    if readiness_score >= 75 and estimated_months <= 3:
        return "Feasible"
    if readiness_score >= 50 and estimated_months <= 6:
        return "Stretch"
    return "Unrealistic"


def _confidence_label(num_user_skills: int, num_role_skills: int) -> str:
    if num_user_skills >= 6 and num_role_skills >= 5:
        return "High"
    if num_user_skills >= 3:
        return "Medium"
    return "Low"


def _what_if_projection(readiness_score: float) -> Dict[int, float]:
    return {
        5: round(min(100.0, readiness_score + 12.0), 1),
        10: round(min(100.0, readiness_score + 22.0), 1),
        15: round(min(100.0, readiness_score + 30.0), 1),
    }


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


def _build_compressed_path(target_role: str, fastest_role: str) -> str:
    if fastest_role.lower() == target_role.lower():
        return target_role
    return f"{fastest_role} → {target_role}"


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
            f"Complete one hands-on task in {top_skills[0]}",
        ]
    else:
        roadmap["Week 1"].append("Review fundamentals")

    if len(top_skills) >= 2:
        roadmap["Week 2"] += [
            f"Focus on {top_skills[1]}",
            f"Build one mini exercise using {top_skills[1]}",
        ]
    else:
        roadmap["Week 2"].append("Strengthen current skills")

    if len(top_skills) >= 3:
        roadmap["Week 3"].append(f"Focus on {top_skills[2]}")
    else:
        roadmap["Week 3"].append("Combine learned skills in one task")

    roadmap["Week 3"].append("Start a small portfolio project")
    roadmap["Week 4"] += [
        "Finish the project",
        "Polish GitHub README",
        "Write down remaining gaps",
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

    readiness_score, matched_skills, bottlenecks = _calculate_weighted_readiness(
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
        readiness_score=round(readiness_score, 1),
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