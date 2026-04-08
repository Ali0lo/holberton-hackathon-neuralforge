from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import pandas as pd

from roleforge.strategy import normalize_skill


@dataclass
class CourseRecommendation:
    skill: str
    course_title: str
    provider: str
    url: str
    level: str
    duration_hours: float
    price_type: str
    quality_score: float


def load_course_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {
        "skill",
        "course_title",
        "provider",
        "url",
        "level",
        "duration_hours",
        "price_type",
        "quality_score",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"course_catalog.csv is missing required columns: {sorted(missing)}")

    for col in ["skill", "course_title", "provider", "url", "level", "price_type"]:
        df[col] = df[col].astype(str).str.strip()

    df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce").fillna(0)
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0)
    df["skill_norm"] = df["skill"].apply(normalize_skill)

    return df


def _extract_skill_name(item: Union[str, Tuple, List]) -> str:
    if isinstance(item, (tuple, list)) and len(item) > 0:
        return str(item[0]).strip()
    return str(item).strip()


def recommend_courses(
    bottlenecks: Sequence[Union[str, Tuple, List]],
    course_df: pd.DataFrame,
    max_courses: int = 3,
) -> List[CourseRecommendation]:
    recommendations: List[CourseRecommendation] = []
    used_titles = set()

    for item in bottlenecks:
        skill = _extract_skill_name(item)
        skill_norm = normalize_skill(skill)

        matches = course_df[course_df["skill_norm"] == skill_norm].copy()
        if matches.empty:
            continue

        matches = matches.sort_values(by=["quality_score", "duration_hours"], ascending=[False, True])
        best = matches.iloc[0]
        title = str(best["course_title"]).strip()

        if title in used_titles:
            continue

        used_titles.add(title)
        recommendations.append(
            CourseRecommendation(
                skill=str(best["skill"]).strip(),
                course_title=title,
                provider=str(best["provider"]).strip(),
                url=str(best["url"]).strip(),
                level=str(best["level"]).strip(),
                duration_hours=float(best["duration_hours"]),
                price_type=str(best["price_type"]).strip(),
                quality_score=float(best["quality_score"]),
            )
        )

        if len(recommendations) >= max_courses:
            break

    return recommendations