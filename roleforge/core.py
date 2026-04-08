import pandas as pd

# =============================
# READINESS CALCULATION
# =============================
def calculate_readiness(df, user_skills, target_role):
    role_df = df[df["role"] == target_role]

    total_weight = role_df["weight"].sum()
    matched_weight = role_df[role_df["skill"].isin(user_skills)]["weight"].sum()

    if total_weight == 0:
        return 0.0

    return (matched_weight / total_weight) * 100


# =============================
# MISSING SKILLS
# =============================
def get_missing_skills(df, user_skills, target_role):
    role_df = df[df["role"] == target_role]

    missing = role_df[~role_df["skill"].isin(user_skills)]

    # sort by importance
    missing = missing.sort_values(by="weight", ascending=False)

    return list(zip(missing["skill"], missing["weight"]))


# =============================
# PROJECTION ENGINE
# =============================
def simulate_progress(current_readiness, hours_per_week):
    # simple heuristic model
    if hours_per_week <= 5:
        rate = 4
    elif hours_per_week <= 10:
        rate = 7
    elif hours_per_week <= 20:
        rate = 10
    else:
        rate = 13

    projections = {
        "3 months": min(current_readiness + rate * 3, 100),
        "6 months": min(current_readiness + rate * 6, 100),
    }

    return projections


# =============================
# ALTERNATIVE ROLE
# =============================
def suggest_alternative_role(df, user_skills, target_role):
    roles = df["role"].unique()

    best_role = target_role
    best_score = 0

    for role in roles:
        score = calculate_readiness(df, user_skills, role)
        if score > best_score:
            best_score = score
            best_role = role

    return best_role, best_score


# =============================
# ROADMAP GENERATOR
# =============================
def generate_roadmap(missing_skills):
    roadmap = {}

    top_skills = [skill for skill, _ in missing_skills[:4]]

    roadmap["Week 1"] = [f"Learn {top_skills[0]}" if len(top_skills) > 0 else "Review basics"]
    roadmap["Week 2"] = [f"Practice {top_skills[1]}" if len(top_skills) > 1 else "Practice skills"]
    roadmap["Week 3"] = ["Build a small project"]
    roadmap["Week 4"] = ["Polish project + create GitHub portfolio"]

    return roadmap