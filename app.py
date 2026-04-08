import streamlit as st

st.set_page_config(
    page_title="RoleForge",
    page_icon="🚀",
    layout="wide",
)

import pandas as pd
import plotly.express as px

from roleforge.recommender import load_course_catalog, recommend_courses
from roleforge.strategy import build_strategy


@st.cache_data
def load_role_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {"role", "skill", "weight"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"role_skill_weights.csv is missing required columns: {missing_columns}")

    df["role"] = df["role"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    return df


@st.cache_data
def load_courses(path: str) -> pd.DataFrame:
    return load_course_catalog(path)


def parse_user_skills(raw_text: str) -> list[str]:
    return [s.strip() for s in raw_text.split(",") if s.strip()]


st.title("🚀 RoleForge — Career Reality Simulator")
st.caption("Constraint-aware career planning. Honest outcomes.")

try:
    role_df = load_role_data("data/role_skill_weights.csv")
    course_df = load_courses("data/course_catalog.csv")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

roles = sorted(role_df["role"].dropna().unique().tolist())
if not roles:
    st.error("No roles found in data/role_skill_weights.csv")
    st.stop()

st.sidebar.header("Input")
target_role = st.sidebar.selectbox("Target role", roles)
user_skills_text = st.sidebar.text_area(
    "Current skills (comma-separated)",
    placeholder="python, sql, pandas, ml",
    height=140,
)
hours_per_week = st.sidebar.slider("Hours per week", 1, 40, 8, 1)
user_estimated_months = st.sidebar.slider("How many months do you think it will take?", 1, 24, 3, 1)

run_simulation = st.sidebar.button("Run simulation", use_container_width=True)

if not run_simulation:
    st.info("Enter your skills, choose a target role, and run the simulation.")
    st.stop()

user_skills = parse_user_skills(user_skills_text)
if not user_skills:
    st.warning("Please enter at least one skill.")
    st.stop()

try:
    strategy = build_strategy(
        role_df=role_df,
        user_skills=user_skills,
        target_role=target_role,
        hours_per_week=hours_per_week,
    )
except Exception as e:
    st.error(f"Strategy engine failed: {e}")
    st.stop()

try:
    recommended_courses = recommend_courses(strategy.bottlenecks, course_df, max_courses=3)
except Exception:
    recommended_courses = []

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readiness", f"{strategy.readiness_score:.1f}%")
c2.metric("Reality Verdict", strategy.reality_verdict)
c3.metric("Fastest Role", strategy.fastest_role)
c4.metric("Confidence", strategy.confidence)

st.subheader("🧠 Reality Gap")
gap = strategy.estimated_months_to_ready - user_estimated_months
if gap > 0:
    st.error(
        f"You estimated **{user_estimated_months} months**, but the simulator suggests "
        f"**{strategy.estimated_months_to_ready} months**. Reality gap: **+{gap} months**."
    )
elif gap < 0:
    st.success(
        f"You estimated **{user_estimated_months} months**, and the simulator suggests "
        f"**{strategy.estimated_months_to_ready} months**."
    )
else:
    st.info(f"Your estimate matches the simulator: **{strategy.estimated_months_to_ready} months**.")

left, right = st.columns(2)

with left:
    st.subheader("✅ Matched Skills")
    if strategy.matched_skills:
        for skill in strategy.matched_skills:
            st.write(f"- {skill}")
    else:
        st.write("No matched skills found.")

with right:
    st.subheader("🚧 Top Bottlenecks")
    if strategy.bottlenecks:
        for skill, weight in strategy.bottlenecks[:5]:
            st.write(f"- **{skill}** — weight {weight:.2f}")
    else:
        st.write("No bottlenecks detected.")

st.subheader("🔁 Strategy Recommendation")
if strategy.fastest_role.lower() != target_role.lower():
    st.info(f"Fastest realistic role: **{strategy.fastest_role}**")
    st.write(f"**Compressed path:** {strategy.compressed_path}")
else:
    st.success("Your target role is already the best current path.")

st.subheader("📈 What-if Simulation")
proj_df = pd.DataFrame(
    [{"Hours per week": h, "Projected readiness": s} for h, s in strategy.what_if_projections.items()]
).sort_values("Hours per week")

fig = px.line(
    proj_df,
    x="Hours per week",
    y="Projected readiness",
    markers=True,
    title="Projected Readiness by Weekly Effort",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("📚 Recommended Courses")
if recommended_courses:
    for course in recommended_courses:
        st.markdown(
            f"- **{course.skill}** -> [{course.course_title}]({course.url})  \n"
            f"  {course.provider} | {course.level} | {course.duration_hours}h | {course.price_type}"
        )
else:
    st.write("No course recommendations found.")

st.subheader("🗓️ 4-Week Roadmap")
for week, tasks in strategy.roadmap.items():
    st.markdown(f"**{week}**")
    for task in tasks:
        st.write(f"- {task}")

with st.expander("Technical summary"):
    st.write("User skills:", user_skills)
    st.write("Target role:", target_role)
    st.write("Hours/week:", hours_per_week)
    st.write("Loaded roles:", roles[:50])
    st.write("Strategy:", strategy)