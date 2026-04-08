from pathlib import Path
from typing import Dict, List

import pandas as pd


DATA_DIR = Path("data")
SKILLS_FILE = DATA_DIR / "Skills.xlsx"
OCC_FILE = DATA_DIR / "Occupation Data.xlsx"
OUTPUT_FILE = DATA_DIR / "role_skill_weights.csv"


# Official O*NET titles -> cleaner aliases for demo/app usage
ROLE_ALIAS_MAP: Dict[str, str] = {
    "Business Intelligence Analysts": "Data Analyst",
    "Data Scientists": "Data Scientist",
    "Software Developers": "Software Engineer",
    "Software Quality Assurance Analysts and Testers": "QA Engineer",
    "Web Developers": "Frontend Developer",
    "Web and Digital Interface Designers": "Frontend Developer",
    "Computer Programmers": "Software Engineer",
    "Computer Systems Analysts": "Systems Analyst",
    "Information Security Analysts": "Cybersecurity Engineer",
    "Database Administrators": "Data Engineer",
    "Database Architects": "Data Engineer",
    "Network and Computer Systems Administrators": "Systems Engineer",
    "Computer and Information Research Scientists": "AI / ML Research Scientist",
}


# Synthetic hackathon-friendly roles built from one or more O*NET roles
SYNTHETIC_ROLE_SOURCES: Dict[str, List[str]] = {
    "AI Engineer": [
        "Computer and Information Research Scientists",
        "Software Developers",
        "Data Scientists",
    ],
    "ML Engineer": [
        "Data Scientists",
        "Computer and Information Research Scientists",
        "Software Developers",
    ],
    "Deep Learning Engineer": [
        "Data Scientists",
        "Computer and Information Research Scientists",
    ],
    "Backend Developer": [
        "Software Developers",
        "Computer Programmers",
    ],
    "Frontend Developer": [
        "Web Developers",
        "Web and Digital Interface Designers",
        "Software Developers",
    ],
    "Cybersecurity Engineer": [
        "Information Security Analysts",
    ],
    "Data Engineer": [
        "Database Architects",
        "Database Administrators",
        "Data Scientists",
    ],
    "Data Analyst": [
        "Business Intelligence Analysts",
        "Data Scientists",
    ],
    "Software Engineer": [
        "Software Developers",
        "Computer Programmers",
        "Software Quality Assurance Analysts and Testers",
    ],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_input_columns(skills_df: pd.DataFrame, occ_df: pd.DataFrame) -> None:
    required_skills_cols = {
        "O*NET-SOC Code",
        "Title",
        "Element Name",
        "Scale Name",
        "Data Value",
    }
    required_occ_cols = {
        "O*NET-SOC Code",
        "Title",
        "Description",
    }

    missing_skills = required_skills_cols - set(skills_df.columns)
    missing_occ = required_occ_cols - set(occ_df.columns)

    if missing_skills:
        raise ValueError(f"Missing columns in Skills.xlsx: {sorted(missing_skills)}")

    if missing_occ:
        raise ValueError(f"Missing columns in Occupation Data.xlsx: {sorted(missing_occ)}")


def build_base_dataset(skills_df: pd.DataFrame, occ_df: pd.DataFrame) -> pd.DataFrame:
    # Keep only importance rows for a cleaner role-skill weight table
    skills_df = skills_df[
        skills_df["Scale Name"].astype(str).str.strip().str.lower() == "importance"
    ].copy()

    skills_df = skills_df[
        ["O*NET-SOC Code", "Title", "Element Name", "Data Value"]
    ].copy()
    skills_df.columns = ["onet_soc_code", "role", "skill", "weight"]

    occ_df = occ_df[
        ["O*NET-SOC Code", "Title", "Description"]
    ].copy()
    occ_df.columns = ["onet_soc_code", "role_occ", "description"]

    merged = skills_df.merge(occ_df, on="onet_soc_code", how="left")

    merged["role"] = merged["role"].fillna(merged["role_occ"])
    merged["role"] = merged["role"].astype(str).str.strip()
    merged["skill"] = merged["skill"].astype(str).str.strip()
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce")

    merged = merged.dropna(subset=["role", "skill", "weight"])

    base = (
        merged.groupby(["role", "skill"], as_index=False)["weight"]
        .mean()
        .sort_values(["role", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return base


def build_alias_roles(base_df: pd.DataFrame) -> pd.DataFrame:
    alias_rows = []

    for original_role, alias_role in ROLE_ALIAS_MAP.items():
        subset = base_df[base_df["role"] == original_role].copy()
        if subset.empty:
            continue

        subset["role"] = alias_role
        alias_rows.append(subset)

    if not alias_rows:
        return pd.DataFrame(columns=base_df.columns)

    alias_df = pd.concat(alias_rows, ignore_index=True)
    alias_df = (
        alias_df.groupby(["role", "skill"], as_index=False)["weight"]
        .mean()
        .sort_values(["role", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return alias_df


def build_synthetic_roles(base_df: pd.DataFrame) -> pd.DataFrame:
    synthetic_rows = []

    for synthetic_role, source_roles in SYNTHETIC_ROLE_SOURCES.items():
        subset = base_df[base_df["role"].isin(source_roles)].copy()
        if subset.empty:
            continue

        subset["role"] = synthetic_role
        synthetic_rows.append(subset)

    if not synthetic_rows:
        return pd.DataFrame(columns=base_df.columns)

    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    synthetic_df = (
        synthetic_df.groupby(["role", "skill"], as_index=False)["weight"]
        .mean()
        .sort_values(["role", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return synthetic_df


def finalize_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    alias_df = build_alias_roles(base_df)
    synthetic_df = build_synthetic_roles(base_df)

    final_df = pd.concat([base_df, alias_df, synthetic_df], ignore_index=True)

    final_df = (
        final_df.groupby(["role", "skill"], as_index=False)["weight"]
        .mean()
        .sort_values(["role", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return final_df


def main() -> None:
    if not SKILLS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SKILLS_FILE}")

    if not OCC_FILE.exists():
        raise FileNotFoundError(f"Missing file: {OCC_FILE}")

    skills_df = pd.read_excel(SKILLS_FILE, sheet_name="Skills")
    occ_df = pd.read_excel(OCC_FILE, sheet_name="Occupation Data")

    skills_df = normalize_columns(skills_df)
    occ_df = normalize_columns(occ_df)

    validate_input_columns(skills_df, occ_df)

    base_df = build_base_dataset(skills_df, occ_df)
    final_df = finalize_dataset(base_df)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(final_df)}")
    print(f"Roles: {final_df['role'].nunique()}")
    print(f"Skills: {final_df['skill'].nunique()}")
    print("\nSample roles:")
    print(final_df["role"].drop_duplicates().head(40).tolist())


if __name__ == "__main__":
    main()