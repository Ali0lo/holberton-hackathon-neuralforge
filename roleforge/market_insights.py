from typing import Dict, List, Tuple

import pandas as pd


def load_market_skills(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"role", "skill", "count", "share"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"top_role_skills.csv is missing columns: {sorted(missing)}")

    df["role"] = df["role"].astype(str).str.strip().str.lower()
    df["skill"] = df["skill"].astype(str).str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df["share"] = pd.to_numeric(df["share"], errors="coerce").fillna(0.0)
    return df


def _normalize_role(role: str) -> str:
    return str(role).strip().lower()


def get_market_skill_insights(
    market_df: pd.DataFrame,
    target_role: str,
    matched_skills: List[str],
    bottlenecks: List[Tuple[str, float]],
    top_n: int = 8,
) -> Dict[str, List[dict]]:
    role = _normalize_role(target_role)
    role_df = market_df[market_df["role"] == role].copy()

    if role_df.empty:
        return {
            "top_market_skills": [],
            "market_missing_skills": [],
            "market_aligned_skills": [],
        }

    role_df = role_df.sort_values(["count", "share"], ascending=[False, False]).head(top_n)

    matched_lower = {str(s).strip().lower() for s in matched_skills}
    bottleneck_lower = {str(skill).strip().lower() for skill, _ in bottlenecks}

    top_market_skills = []
    market_missing_skills = []
    market_aligned_skills = []

    for _, row in role_df.iterrows():
        item = {
            "skill": row["skill"],
            "count": int(row["count"]),
            "share": float(row["share"]),
        }
        top_market_skills.append(item)

        skill_lower = str(row["skill"]).strip().lower()
        if skill_lower in bottleneck_lower:
            market_missing_skills.append(item)
        if skill_lower in matched_lower:
            market_aligned_skills.append(item)

    return {
        "top_market_skills": top_market_skills,
        "market_missing_skills": market_missing_skills,
        "market_aligned_skills": market_aligned_skills,
    }