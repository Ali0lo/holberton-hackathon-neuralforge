from pathlib import Path
import re

import pandas as pd


DATA_DIR = Path("data")
SKILLS_FILE = DATA_DIR / "Skills.xlsx"
OCC_FILE = DATA_DIR / "Occupation Data.xlsx"
OUTPUT_FILE = DATA_DIR / "role_skill_weights.csv"


TARGET_ROLES = {
    "data scientist",
    "machine learning engineer",
    "ml engineer",
    "ai engineer",
    "artificial intelligence engineer",
    "data analyst",
    "business intelligence analyst",
    "backend developer",
    "backend engineer",
    "software developer",
    "software engineer",
    "mlops engineer",
    "data engineer",
    "research scientist",
    "computer vision engineer",
    "nlp engineer",
}


def clean_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_role(name: str) -> str:
    return str(name).strip()


def normalize_scale_name(name: str) -> str:
    return str(name).strip().lower()


def load_first_usable_sheet(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if not df.empty:
            df.columns = [clean_col(c) for c in df.columns]
            return df
    raise ValueError(f"No usable sheet found in {path}")


def load_best_skills_sheet(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)

    best_df = None
    best_score = -1

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        if df.empty:
            continue

        cleaned = df.copy()
        cleaned.columns = [clean_col(c) for c in cleaned.columns]
        cols = set(cleaned.columns)

        score = 0
        for token in ["onet_soc_code", "element_name", "scale_name", "data_value"]:
            if token in cols:
                score += 1

        if score > best_score:
            best_score = score
            best_df = cleaned

    if best_df is None:
        raise ValueError("Could not find a usable sheet in Skills.xlsx")

    return best_df


def detect_occ_columns(df: pd.DataFrame):
    cols = set(df.columns)

    code_col = None
    title_col = None

    for c in df.columns:
        if c in {"onet_soc_code", "o_net_soc_code", "soc_code"}:
            code_col = c
        if c in {"title", "occupation_title"}:
            title_col = c

    if code_col is None:
        for c in df.columns:
            if "code" in c:
                code_col = c
                break

    if title_col is None:
        for c in df.columns:
            if "title" in c or "occupation" in c:
                title_col = c
                break

    if code_col is None or title_col is None:
        raise ValueError(f"Could not detect occupation columns from: {list(df.columns)}")

    return code_col, title_col


def detect_skill_columns(df: pd.DataFrame):
    code_col = None
    skill_col = None
    scale_col = None
    value_col = None

    for c in df.columns:
        if c in {"onet_soc_code", "o_net_soc_code", "soc_code"}:
            code_col = c
        elif c in {"element_name", "skill", "skill_name"}:
            skill_col = c
        elif c == "scale_name":
            scale_col = c
        elif c == "data_value":
            value_col = c

    if code_col is None:
        for c in df.columns:
            if "code" in c:
                code_col = c
                break

    if skill_col is None:
        for c in df.columns:
            if "element" in c or "skill" in c:
                skill_col = c
                break

    if scale_col is None:
        for c in df.columns:
            if "scale" in c:
                scale_col = c
                break

    if value_col is None:
        for c in df.columns:
            if "value" in c:
                value_col = c
                break

    if None in {code_col, skill_col, scale_col, value_col}:
        raise ValueError(f"Could not detect skill columns from: {list(df.columns)}")

    return code_col, skill_col, scale_col, value_col


def main():
    if not SKILLS_FILE.exists():
        raise FileNotFoundError(f"Missing {SKILLS_FILE}")
    if not OCC_FILE.exists():
        raise FileNotFoundError(f"Missing {OCC_FILE}")

    occ_df = load_first_usable_sheet(OCC_FILE)
    skill_df = load_best_skills_sheet(SKILLS_FILE)

    occ_code_col, occ_title_col = detect_occ_columns(occ_df)
    skill_code_col, skill_name_col, scale_col, value_col = detect_skill_columns(skill_df)

    occ_df = occ_df[[occ_code_col, occ_title_col]].dropna().copy()
    occ_df.columns = ["onet_soc_code", "role"]
    occ_df["role"] = occ_df["role"].apply(normalize_role)

    skill_df = skill_df[[skill_code_col, skill_name_col, scale_col, value_col]].dropna().copy()
    skill_df.columns = ["onet_soc_code", "skill", "scale_name", "data_value"]
    skill_df["scale_name"] = skill_df["scale_name"].apply(normalize_scale_name)
    skill_df["data_value"] = pd.to_numeric(skill_df["data_value"], errors="coerce")
    skill_df = skill_df.dropna(subset=["data_value"])

    merged = skill_df.merge(occ_df, on="onet_soc_code", how="left").dropna(subset=["role"])

    # Prefer Importance if available
    importance = merged[merged["scale_name"].str.contains("importance", na=False)].copy()

    if importance.empty:
        raise ValueError("No Importance rows found in Skills.xlsx")

    # Filter to stronger demo roles first
    importance["role_lower"] = importance["role"].str.lower()
    filtered = importance[
        importance["role_lower"].apply(lambda x: any(target in x for target in TARGET_ROLES))
    ].copy()

    if filtered.empty:
        filtered = importance.copy()

    result = filtered[["role", "skill", "data_value"]].copy()
    result.columns = ["role", "skill", "weight"]

    # Aggregate duplicates
    result = (
        result.groupby(["role", "skill"], as_index=False)["weight"]
        .mean()
        .sort_values(["role", "weight"], ascending=[True, False])
    )

    result.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE} with {len(result)} rows")
    print(f"Roles: {result['role'].nunique()}")
    print(f"Skills: {result['skill'].nunique()}")


if __name__ == "__main__":
    main()