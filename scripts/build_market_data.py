import re
from pathlib import Path

import pandas as pd


INPUT_CSV = Path('/Users/eldanizarifzada/Downloads/JobsDatasetProcessed.csv')
OUTPUT_DIR = Path("data")


ROLE_MAP = {
    "artificial intelligence": "ai engineer",
    "data science": "data scientist",
    "data analyst": "data analyst",
    "machine learning": "machine learning engineer",
    "software engineering": "software engineer",
    "software engineer": "software engineer",
    "backend": "backend engineer",
    "frontend": "frontend engineer",
    "full stack": "full stack engineer",
    "devops": "devops engineer",
    "cloud": "cloud engineer",
    "cybersecurity": "cybersecurity engineer",
    "cyber security": "cybersecurity engineer",
    "soc": "soc analyst",
    "penetration testing": "penetration tester",
    "penetration tester": "penetration tester",
    "data engineering": "data engineer",
    "mlops": "mlops engineer",
}


SKILL_MAP = {
    "python programming": "python",
    "python": "python",
    "sql databases": "sql",
    "sql": "sql",
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "artificial intelligence": "artificial intelligence",
    "large language models": "large language models",
    "llms": "large language models",
    "llm": "large language models",
    "natural language processing": "natural language processing",
    "nlp": "natural language processing",
    "computer vision": "computer vision",
    "cv": "computer vision",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "numpy": "numpy",
    "pandas": "pandas",
    "statistics": "statistics",
    "data analysis": "data analysis",
    "data visualization": "data visualization",
    "power bi": "bi tools",
    "tableau": "dashboarding",
    "excel": "excel",
    "etl": "etl",
    "data pipelines": "etl",
    "apache spark": "apache spark",
    "spark": "apache spark",
    "airflow": "airflow",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "ci/cd": "ci/cd",
    "mlops": "mlops",
    "cloud": "cloud",
    "aws": "cloud",
    "azure": "cloud",
    "gcp": "cloud",
    "terraform": "cloud",
    "linux": "linux",
    "network security": "security fundamentals",
    "cybersecurity": "security fundamentals",
    "cyber security": "security fundamentals",
    "web security": "security fundamentals",
    "owasp": "security fundamentals",
    "penetration testing": "security fundamentals",
    "siem": "siem",
    "splunk": "siem",
    "incident response": "incident response",
    "git": "git",
    "github": "git",
    "apis": "apis",
    "api development": "apis",
    "rest api": "apis",
    "javascript": "javascript",
    "react": "react",
    "html": "html",
    "css": "css",
    "system design": "system design",
    "data structures": "data structures",
    "algorithms": "algorithms",
}


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"\s+", " ", text)
    return text


def map_role(query: str) -> str:
    q = normalize_text(query)
    for key, mapped in ROLE_MAP.items():
        if key in q:
            return mapped
    return q


def map_skill(skill: str) -> str:
    s = normalize_text(skill)
    return SKILL_MAP.get(s, s)


def demand_label(count: int, q75: float, q40: float) -> str:
    if count >= q75:
        return "High"
    if count >= q40:
        return "Medium"
    return "Low"


def split_skills(skill_text: str) -> list[str]:
    if pd.isna(skill_text):
        return []

    parts = [p.strip() for p in str(skill_text).split(",")]
    parts = [p for p in parts if p and p.lower() != "nan"]

    cleaned = []
    seen = set()
    for p in parts:
        mapped = map_skill(p)
        if mapped and mapped not in seen:
            cleaned.append(mapped)
            seen.add(mapped)
    return cleaned


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    required = {"Query", "IT Skills"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work["role"] = work["Query"].apply(map_role)

    # -------------------------
    # 1) role_demand.csv
    # -------------------------
    role_counts = (
        work.groupby("role")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
    )

    q75 = role_counts["job_count"].quantile(0.75)
    q40 = role_counts["job_count"].quantile(0.40)

    role_counts["demand_label"] = role_counts["job_count"].apply(
        lambda x: demand_label(int(x), q75, q40)
    )

    role_counts.to_csv(OUTPUT_DIR / "role_demand.csv", index=False)

    # -------------------------
    # 2) top_role_skills.csv
    # -------------------------
    work["skill_list"] = work["IT Skills"].apply(split_skills)

    exploded = work[["role", "skill_list"]].explode("skill_list").rename(
        columns={"skill_list": "skill"}
    )
    exploded = exploded.dropna(subset=["skill"])
    exploded = exploded[exploded["skill"].astype(str).str.len() > 1]

    skill_counts = (
        exploded.groupby(["role", "skill"])
        .size()
        .reset_index(name="count")
        .sort_values(["role", "count"], ascending=[True, False])
    )

    role_totals = (
        exploded.groupby("role")
        .size()
        .reset_index(name="role_skill_mentions")
    )

    skill_counts = skill_counts.merge(role_totals, on="role", how="left")
    skill_counts["share"] = (
        skill_counts["count"] / skill_counts["role_skill_mentions"]
    ).round(4)

    skill_counts = skill_counts.drop(columns=["role_skill_mentions"])

    # keep top 15 skills per role
    skill_counts = (
        skill_counts.groupby("role", group_keys=False)
        .head(15)
        .reset_index(drop=True)
    )

    skill_counts.to_csv(OUTPUT_DIR / "top_role_skills.csv", index=False)

    print("Done.")
    print(f"Saved: {OUTPUT_DIR / 'role_demand.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'top_role_skills.csv'}")


if __name__ == "__main__":
    main()