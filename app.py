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
from roleforge.llm_helper import generate_roleforge_explanation, generate_cv_overview
from roleforge.cv_parser import (
    extract_text_from_uploaded_file,
    extract_skill_matches_from_text,
)


@st.cache_data
def load_role_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = {"role", "skill", "weight"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"role_skill_weights.csv is missing required columns: {missing_columns}"
        )

    df["role"] = df["role"].astype(str).str.strip()
    df["skill"] = df["skill"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    return df


@st.cache_data
def load_courses(path: str) -> pd.DataFrame:
    return load_course_catalog(path)


def parse_user_skills(raw_text: str) -> list[str]:
    return [s.strip() for s in raw_text.split(",") if s.strip()]


def build_compact_roadmap(base_roadmap: dict, total_weeks: int) -> list[tuple[str, list[str]]]:
    """
    Expands the 4-phase roadmap into N weeks, then compresses consecutive
    identical phases into intervals like 'Week 1-4'.
    """
    if not base_roadmap:
        return []

    phase_tasks = list(base_roadmap.values())
    num_phases = len(phase_tasks)
    total_weeks = max(1, int(total_weeks))

    expanded = []
    for week_idx in range(total_weeks):
        phase_idx = min((week_idx * num_phases) // total_weeks, num_phases - 1)
        expanded.append(phase_tasks[phase_idx])

    compact = []
    start_week = 1
    current_tasks = expanded[0]

    for i in range(1, total_weeks):
        if expanded[i] != current_tasks:
            end_week = i
            label = f"Week {start_week}" if start_week == end_week else f"Week {start_week}-{end_week}"
            compact.append((label, current_tasks))
            start_week = i + 1
            current_tasks = expanded[i]

    end_week = total_weeks
    label = f"Week {start_week}" if start_week == end_week else f"Week {start_week}-{end_week}"
    compact.append((label, current_tasks))

    return compact


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
input_mode = st.sidebar.radio("Skill input mode", ["Manual input", "Upload CV"])

cv_text = ""
extracted_skill_matches = []
user_skills = []

if input_mode == "Manual input":
    user_skills_text = st.sidebar.text_area(
        "Current skills (comma-separated)",
        placeholder="python, ml, pytorch, pandas",
        height=140,
    )
    user_skills = parse_user_skills(user_skills_text)

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CV",
        type=["pdf", "docx", "txt"],
        help="Upload a PDF, DOCX, or TXT CV.",
    )
    extra_skills_text = st.sidebar.text_area(
        "Optional extra skills",
        placeholder="rag, vector databases",
        height=100,
    )

    if uploaded_file is not None:
        try:
            cv_text = extract_text_from_uploaded_file(uploaded_file)
            extracted_skill_matches = extract_skill_matches_from_text(cv_text, role_df)
            extracted_skills = [skill for skill, _ in extracted_skill_matches]
            extra_skills = parse_user_skills(extra_skills_text)
            user_skills = sorted(set(extracted_skills + extra_skills))
        except Exception as e:
            st.error(f"Failed to read CV: {e}")
            st.stop()

hours_per_week = st.sidebar.slider("Hours per week", 1, 40, 8, 1)
user_estimated_months = st.sidebar.slider(
    "How many months do you think it will take?", 1, 24, 3, 1
)
use_live_course_search = st.sidebar.checkbox("Use live course search", value=False)
use_local_llm = st.sidebar.checkbox("Use local LLM explanation", value=False)

run_simulation = st.sidebar.button("Run simulation", use_container_width=True)

if not run_simulation:
    st.info("Enter your skills, choose a target role, and run the simulation.")
    st.stop()

if not user_skills:
    if input_mode == "Upload CV":
        st.warning("Please upload a CV with extractable text or add some extra skills.")
    else:
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
    recommended_courses = recommend_courses(
        strategy.bottlenecks,
        course_df,
        max_courses=3,
        use_live_search=use_live_course_search,
    )
except Exception:
    recommended_courses = []

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readiness", f"{strategy.readiness_score:.1f}%")
c2.metric("Reality Verdict", strategy.reality_verdict)
c3.metric("Fastest Role", strategy.fastest_role)
c4.metric("Confidence", strategy.confidence)
if strategy.reality_verdict == "Highly Ready":
    st.success("You already have strong alignment with this role. Focus on polishing projects and portfolio proof.")
elif strategy.reality_verdict == "Feasible":
    st.success("This target looks realistic within your selected timeframe if you stay consistent.")
elif strategy.reality_verdict == "Stretch":
    st.warning("This role is possible, but it will require focused effort and closing several important gaps.")
else:
    st.error("This goal is unrealistic for the selected timeframe at your current readiness level.")

if input_mode == "Upload CV" and cv_text:
    st.subheader("📄 CV Overview")

    if use_local_llm:
        cv_overview = generate_cv_overview(
            cv_text=cv_text,
            extracted_skills=user_skills,
            target_role=target_role,
        )
        if cv_overview:
            st.write(cv_overview)

    if extracted_skill_matches:
        st.markdown("**Detected Skills from CV**")
        detected_only = []
        seen = set()

    for skill, _ in extracted_skill_matches:
        if skill not in seen:
            detected_only.append(skill)
            seen.add(skill)

    for skill in detected_only:
        st.write(f"- {skill}")      

    with st.expander("Preview extracted CV text"):
        st.text_area("CV text", cv_text[:5000], height=220)

if use_local_llm:
    explanation = generate_roleforge_explanation(
        target_role=target_role,
        user_skills=user_skills,
        readiness_score=strategy.readiness_score,
        bottlenecks=strategy.bottlenecks,
        fastest_role=strategy.fastest_role,
    )
    if explanation:
        st.subheader("🧠 AI Explanation")
        st.write(explanation)

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
    st.info(
        f"Your estimate matches the simulator: **{strategy.estimated_months_to_ready} months**."
    )

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

st.subheader("📈 Learning Curve")
curve_df = pd.DataFrame(strategy.projection_series)
fig_curve = px.line(
    curve_df,
    x="Month",
    y="Projected readiness",
    markers=True,
    title="Projected Readiness Over Time",
)
st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("📊 Skill Gap Visualization")

if strategy.bottlenecks:
    bottleneck_df = pd.DataFrame(
        strategy.bottlenecks[:8],
        columns=["Skill", "Weight"],
    ).sort_values("Weight", ascending=True)

    fig_gap = px.bar(
        bottleneck_df,
        x="Weight",
        y="Skill",
        orientation="h",
        title="Top Missing Skills by Importance",
    )
    st.plotly_chart(fig_gap, use_container_width=True)
else:
    st.success("No major skill gaps detected.")

st.subheader("📚 Recommended Courses")
if recommended_courses:
    for course in recommended_courses:
        st.markdown(
            f"- **{course.skill}** -> [{course.course_title}]({course.url})  \n"
            f"  {course.provider} | {course.level} | {course.duration_hours}h | {course.price_type}"
        )
else:
    st.write("No course recommendations found.")

st.subheader("🗓️ Roadmap")

total_weeks = user_estimated_months * 4
compact_roadmap = build_compact_roadmap(strategy.roadmap, total_weeks)

st.caption(f"Generated for **{total_weeks} weeks** based on your selected timeframe.")

for interval, tasks in compact_roadmap:
    st.markdown(f"**{interval}**")
    for task in tasks:
        st.write(f"- {task}")

with st.expander("Technical summary"):
    st.write("User skills:", user_skills)
    st.write("Target role:", target_role)
    st.write("Hours/week:", hours_per_week)
    st.write("Loaded roles:", roles)
    st.write("Input mode:", input_mode)
    st.write("CV skill matches:", extracted_skill_matches)
    st.write("Strategy:", strategy)