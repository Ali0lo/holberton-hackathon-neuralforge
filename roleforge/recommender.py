from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, Optional
import os

import pandas as pd
import requests

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


TRUSTED_DOMAINS = [
    "coursera.org",
    "deeplearning.ai",
    "fast.ai",
    "developers.google.com",
    "tensorflow.org",
    "pytorch.org",
    "scikit-learn.org",
    "freecodecamp.org",
    "khanacademy.org",
    "learn.microsoft.com",
    "docs.llamaindex.ai",
    "pinecone.io",
    "docker.com",
    "kubernetes.io",
]


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


def _search_courses_brave(skill: str, level: str = "beginner") -> List[CourseRecommendation]:
    """
    Optional live search using Brave Search API.
    Requires BRAVE_SEARCH_API_KEY in environment.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return []

    query = f"best {level} {skill} course site:{' OR site:'.join(TRUSTED_DOMAINS[:6])}"

    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            params={
                "q": query,
                "count": 5,
            },
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return []

    results = []
    web_results = data.get("web", {}).get("results", [])

    for item in web_results:
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        provider = url.split("/")[2] if "://" in url else "Web"
        if not url or not title:
            continue

        if not any(domain in url for domain in TRUSTED_DOMAINS):
            continue

        results.append(
            CourseRecommendation(
                skill=skill,
                course_title=title,
                provider=provider,
                url=url,
                level=level,
                duration_hours=0.0,
                price_type="unknown",
                quality_score=0.80,
            )
        )

    return results


def recommend_courses(
    bottlenecks: Sequence[Union[str, Tuple, List]],
    course_df: pd.DataFrame,
    max_courses: int = 3,
    use_live_search: bool = False,
) -> List[CourseRecommendation]:
    recommendations: List[CourseRecommendation] = []
    used_titles = set()

    for item in bottlenecks:
        skill = _extract_skill_name(item)
        skill_norm = normalize_skill(skill)

        # optional live search first
        if use_live_search:
            live_results = _search_courses_brave(skill)
            for rec in live_results:
                if rec.course_title not in used_titles:
                    used_titles.add(rec.course_title)
                    recommendations.append(rec)
                    break

            if len(recommendations) >= max_courses:
                break

        # fallback to CSV
        matches = course_df[course_df["skill_norm"] == skill_norm].copy()
        if matches.empty:
            continue

        matches = matches.sort_values(
            by=["quality_score", "duration_hours"],
            ascending=[False, True],
        )

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