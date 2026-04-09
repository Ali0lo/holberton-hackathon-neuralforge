import os
import streamlit as st

st.set_page_config(
    page_title="RoleForge",
    page_icon="🚀",
    layout="wide",
)

import pandas as pd
import plotly.express as px

from roleforge.recommender import load_course_catalog, recommend_courses
from roleforge.strategy import build_strategy, normalize_skill
from roleforge.llm_helper import (
    generate_roleforge_explanation,
    llm_map_user_skills,
    stream_roleforge_explanation,
    stream_cv_overview,
)
from roleforge.cv_parser import (
    extract_text_from_uploaded_file,
    extract_skill_matches_from_text,
    extract_skills_from_text,
)
from roleforge.report_export import build_roleforge_report_pdf
from roleforge.project_generator import generate_project_recommendation
from roleforge.resume_analyzer import analyze_resume_gaps
from roleforge.market_insights import load_market_skills, get_market_skill_insights


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


def format_role_name(role: str) -> str:
    special_words = {
        "ai": "AI",
        "ml": "ML",
        "mlops": "MLOps",
        "soc": "SOC",
        "llm": "LLM",
        "nlp": "NLP",
        "cv": "CV",
        "qa": "QA",
        "ui": "UI",
        "ux": "UX",
        "ci/cd": "CI/CD",
    }

    words = str(role).split()
    formatted = []

    for word in words:
        lw = word.lower()
        if lw in special_words:
            formatted.append(special_words[lw])
        else:
            formatted.append(word[0].upper() + word[1:].lower() if word else word)

    return " ".join(formatted)


def build_compact_roadmap(
    base_roadmap: dict, total_weeks: int
) -> list[tuple[str, list[str]]]:
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
            label = (
                f"Week {start_week}"
                if start_week == end_week
                else f"Week {start_week}-{end_week}"
            )
            compact.append((label, current_tasks))
            start_week = i + 1
            current_tasks = expanded[i]

    end_week = total_weeks
    label = (
        f"Week {start_week}"
        if start_week == end_week
        else f"Week {start_week}-{end_week}"
    )
    compact.append((label, current_tasks))

    return compact


def gap_priority_label(weight: float) -> str:
    if weight >= 4.5:
        return "Critical"
    if weight >= 3.5:
        return "Important"
    return "Helpful"


def add_simulated_skill(user_skills: list[str], new_skill: str) -> list[str]:
    cleaned = new_skill.strip()
    if not cleaned:
        return user_skills

    existing_norm = {normalize_skill(skill) for skill in user_skills}
    if normalize_skill(cleaned) in existing_norm:
        return user_skills

    return sorted(user_skills + [cleaned])


st.title("🚀 RoleForge — Career Reality Simulator")
st.caption("Constraint-aware career planning. Honest outcomes.")

try:
    role_df = load_role_data("data/role_skill_weights.csv")
    course_df = load_courses("data/course_catalog.csv")
    market_df = load_market_skills("data/top_role_skills.csv")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

raw_roles = sorted(role_df["role"].dropna().unique().tolist())
if not raw_roles:
    st.error("No roles found in data/role_skill_weights.csv")
    st.stop()

display_roles = [format_role_name(r) for r in raw_roles]
role_map = dict(zip(display_roles, raw_roles))

allowed_skills = sorted(
    {str(skill).strip() for skill in role_df["skill"].dropna().unique().tolist()}
)

with st.expander("LLM Debug"):
    st.write("OPENAI key loaded (env):", bool(os.getenv("OPENAI_API_KEY")))
    st.write("OPENAI model (env):", os.getenv("OPENAI_MODEL"))
    try:
        st.write("OPENAI key in secrets:", "OPENAI_API_KEY" in st.secrets)
        st.write("OPENAI model in secrets:", st.secrets.get("OPENAI_MODEL", "missing"))
    except Exception:
        st.write("Secrets not accessible")

st.sidebar.header("Input")
target_role_display = st.sidebar.selectbox("Target role", display_roles, key="target_role_v2")
target_role = role_map[target_role_display]
input_mode = st.sidebar.radio("Skill input mode", ["Manual input", "Upload CV"])

cv_text = ""
extracted_skill_matches = []
user_skills = []

hours_per_week = st.sidebar.slider("Hours per week", 1, 40, 8, 1)
user_estimated_months = st.sidebar.slider(
    "How many months do you think it will take?", 1, 24, 3, 1
)
use_llm = st.sidebar.checkbox("Use AI explanation", value=True)
use_llm_skill_mapping = st.sidebar.checkbox("Use LLM skill mapping", value=False)

st.sidebar.caption("📎 Max CV upload size: 5MB")

if input_mode == "Manual input":
    user_skills_text = st.sidebar.text_area(
        "Current skills (comma-separated)",
        placeholder="python, ai, supervised learning, transformers, pandas",
        height=140,
    )
    raw_user_skills = parse_user_skills(user_skills_text)

    if use_llm_skill_mapping and raw_user_skills:
        llm_skills = llm_map_user_skills(
            raw_skills=raw_user_skills,
            target_role=target_role,
            allowed_skills=allowed_skills,
        )
        user_skills = sorted(set(raw_user_skills + llm_skills))
    else:
        user_skills = raw_user_skills

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
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File too large. Max size is 5MB.")
            st.stop()

        try:
            cv_text = extract_text_from_uploaded_file(uploaded_file)
            extracted_skill_matches = extract_skill_matches_from_text(cv_text, role_df)

            extracted_skills = extract_skills_from_text(
                cv_text,
                role_df,
                target_role=target_role,
                use_llm=use_llm_skill_mapping,
            )

            extra_skills = parse_user_skills(extra_skills_text)

            if use_llm_skill_mapping and extra_skills:
                extra_llm = llm_map_user_skills(
                    raw_skills=extra_skills,
                    target_role=target_role,
                    allowed_skills=allowed_skills,
                )
                extra_skills = sorted(set(extra_skills + extra_llm))

            user_skills = sorted(set(extracted_skills + extra_skills))
        except Exception as e:
            st.error(f"Failed to read CV: {e}")
            st.stop()

if "simulation_ran" not in st.session_state:
    st.session_state.simulation_ran = False
if "baseline_user_skills" not in st.session_state:
    st.session_state.baseline_user_skills = []
if "baseline_target_role" not in st.session_state:
    st.session_state.baseline_target_role = None
if "baseline_hours_per_week" not in st.session_state:
    st.session_state.baseline_hours_per_week = None
if "baseline_user_estimated_months" not in st.session_state:
    st.session_state.baseline_user_estimated_months = None
if "baseline_cv_text" not in st.session_state:
    st.session_state.baseline_cv_text = ""
if "baseline_extracted_skill_matches" not in st.session_state:
    st.session_state.baseline_extracted_skill_matches = []
if "baseline_input_mode" not in st.session_state:
    st.session_state.baseline_input_mode = input_mode

run_simulation = st.sidebar.button("Run simulation", use_container_width=True)

if run_simulation:
    if not user_skills:
        if input_mode == "Upload CV":
            st.warning("Please upload a CV with extractable text or add some extra skills.")
        else:
            st.warning("Please enter at least one skill.")
        st.stop()

    st.session_state.simulation_ran = True
    st.session_state.baseline_user_skills = user_skills.copy()
    st.session_state.baseline_target_role = target_role
    st.session_state.baseline_hours_per_week = hours_per_week
    st.session_state.baseline_user_estimated_months = user_estimated_months
    st.session_state.baseline_cv_text = cv_text
    st.session_state.baseline_extracted_skill_matches = extracted_skill_matches
    st.session_state.baseline_input_mode = input_mode

if not st.session_state.simulation_ran:
    st.info("Enter your skills or upload your CV, then click Run simulation.")
    st.stop()

baseline_user_skills = st.session_state.baseline_user_skills
baseline_target_role = st.session_state.baseline_target_role
baseline_hours_per_week = st.session_state.baseline_hours_per_week
baseline_user_estimated_months = st.session_state.baseline_user_estimated_months
baseline_cv_text = st.session_state.baseline_cv_text
baseline_extracted_skill_matches = st.session_state.baseline_extracted_skill_matches
baseline_input_mode = st.session_state.baseline_input_mode

strategy = build_strategy(
    role_df=role_df,
    user_skills=baseline_user_skills,
    target_role=baseline_target_role,
    hours_per_week=baseline_hours_per_week,
)

recommended_courses = recommend_courses(
    strategy.bottlenecks,
    course_df,
    max_courses=3,
    use_live_search=False,
    use_llm_rerank=False,
    target_role=baseline_target_role,
)

project_recommendation = generate_project_recommendation(
    target_role=baseline_target_role,
    matched_skills=strategy.matched_skills,
    bottlenecks=strategy.bottlenecks,
)

resume_gap_report = analyze_resume_gaps(
    target_role=baseline_target_role,
    matched_skills=strategy.matched_skills,
    bottlenecks=strategy.bottlenecks,
)

market_insights = get_market_skill_insights(
    market_df=market_df,
    target_role=baseline_target_role,
    matched_skills=strategy.matched_skills,
    bottlenecks=strategy.bottlenecks,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Readiness", f"{strategy.readiness_score:.1f}%")
c2.metric("Reality Verdict", strategy.reality_verdict)
c3.metric("Fastest Role", format_role_name(strategy.fastest_role))
c4.metric("Confidence", strategy.confidence)

if strategy.fastest_role.lower() != baseline_target_role.lower():
    st.info(
        f"Best role for you right now: **{format_role_name(strategy.fastest_role)}**. "
        f"That path looks more realistic than **{format_role_name(baseline_target_role)}** at your current skill level."
    )

if strategy.reality_verdict == "Highly Ready":
    st.success(
        "You already have strong alignment with this role. Focus on polishing projects and portfolio proof."
    )
elif strategy.reality_verdict == "Feasible":
    st.success(
        "This target looks realistic within your selected timeframe if you stay consistent."
    )
elif strategy.reality_verdict == "Stretch":
    st.warning(
        "This role is possible, but it will require focused effort and closing several important gaps."
    )
else:
    st.error(
        "This goal is unrealistic for the selected timeframe at your current readiness level."
    )

st.subheader("🎯 Suggested Roles Based on Your Profile")
if strategy.alternative_roles:
    top_roles = strategy.alternative_roles[:3]
    cols = st.columns(len(top_roles))
    for col, (role, score) in zip(cols, top_roles):
        with col:
            st.metric(format_role_name(role), f"{score:.1f}%")
else:
    st.write("No role suggestions available.")

if baseline_input_mode == "Upload CV" and baseline_cv_text:
    st.subheader("📄 CV Overview")

    if use_llm:
        st.write_stream(
            stream_cv_overview(
                cv_text=baseline_cv_text,
                extracted_skills=baseline_user_skills,
                target_role=baseline_target_role,
            )
        )

    if baseline_extracted_skill_matches:
        st.markdown("**Detected Skills from CV**")
        detected_only = []
        seen = set()

        for skill, _ in baseline_extracted_skill_matches:
            if skill not in seen:
                detected_only.append(skill)
                seen.add(skill)

        for skill in detected_only:
            st.write(f"- {skill}")

    with st.expander("Preview extracted CV text"):
        st.text_area("CV text", baseline_cv_text[:5000], height=220)

st.subheader("🧠 AI Explanation")
if use_llm:
    st.write_stream(
        stream_roleforge_explanation(
            target_role=baseline_target_role,
            user_skills=baseline_user_skills,
            readiness_score=strategy.readiness_score,
            bottlenecks=strategy.bottlenecks,
            fastest_role=strategy.fastest_role,
        )
    )
else:
    explanation = generate_roleforge_explanation(
        target_role=baseline_target_role,
        user_skills=baseline_user_skills,
        readiness_score=strategy.readiness_score,
        bottlenecks=strategy.bottlenecks,
        fastest_role=strategy.fastest_role,
    )
    if explanation:
        st.write(explanation)

st.subheader("🧠 Reality Gap")
gap = strategy.estimated_months_to_ready - baseline_user_estimated_months
if gap > 0:
    st.error(
        f"You estimated **{baseline_user_estimated_months} months**, but the simulator suggests "
        f"**{strategy.estimated_months_to_ready} months**. Reality gap: **+{gap} months**."
    )
elif gap < 0:
    st.success(
        f"You estimated **{baseline_user_estimated_months} months**, and the simulator suggests "
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
            priority = gap_priority_label(weight)
            st.write(f"- **{skill}** — weight {weight:.2f} | priority: **{priority}**")
    else:
        st.write("No bottlenecks detected.")

st.subheader("📡 Public Market Skill Signals")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top skills in public IT jobs for this path**")
    if market_insights["top_market_skills"]:
        for item in market_insights["top_market_skills"][:6]:
            st.write(
                f"- {item['skill']} ({item['count']} mentions, share {item['share']:.2%})"
            )
    else:
        st.write("No market skill data available for this role yet.")

with col2:
    st.markdown("**Missing but market-relevant skills**")
    if market_insights["market_missing_skills"]:
        for item in market_insights["market_missing_skills"][:6]:
            st.write(
                f"- {item['skill']} ({item['count']} mentions)"
            )
    else:
        st.write("No overlapping market-missing skills found.")

st.markdown("**Skills you already have that align with public IT job data**")
if market_insights["market_aligned_skills"]:
    for item in market_insights["market_aligned_skills"][:6]:
        st.write(f"- {item['skill']} ({item['count']} mentions)")
else:
    st.write("No aligned market skills found yet.")

st.subheader("📝 Resume Gap Fixer")
st.markdown("**Missing keywords to strengthen in your CV / portfolio**")
for keyword in resume_gap_report["missing_keywords"]:
    st.write(f"- {keyword}")

st.markdown("**Suggested bullet point themes**")
for bullet in resume_gap_report["bullet_point_ideas"]:
    st.write(f"- {bullet}")

st.subheader("🔁 What-if Simulator")

suggested_skills = []
existing_norm = {normalize_skill(skill) for skill in baseline_user_skills}

for skill, _weight in strategy.bottlenecks:
    if normalize_skill(skill) not in existing_norm:
        suggested_skills.append(skill)

suggested_skills = suggested_skills[:10]

if "selected_what_if_skills" not in st.session_state:
    st.session_state.selected_what_if_skills = []

col_a, col_b = st.columns([3, 1])

with col_a:
    selected_what_if_skills = st.multiselect(
        "Try adding one or more high-impact missing skills",
        options=suggested_skills,
        default=st.session_state.selected_what_if_skills,
        key="selected_what_if_skills",
    )

with col_b:
    st.write("")
    st.write("")
    simulate_clicked = st.button(
        "Simulate",
        use_container_width=True,
        disabled=not bool(suggested_skills),
    )

if simulate_clicked and selected_what_if_skills:
    simulated_skills = baseline_user_skills.copy()

    for skill in selected_what_if_skills:
        simulated_skills = add_simulated_skill(simulated_skills, skill)

    simulated_strategy = build_strategy(
        role_df=role_df,
        user_skills=simulated_skills,
        target_role=baseline_target_role,
        hours_per_week=baseline_hours_per_week,
    )

    st.session_state["what_if_result"] = {
        "skills": selected_what_if_skills,
        "readiness": simulated_strategy.readiness_score,
        "verdict": simulated_strategy.reality_verdict,
        "fastest_role": simulated_strategy.fastest_role,
        "months_to_ready": simulated_strategy.estimated_months_to_ready,
    }

if not selected_what_if_skills and "what_if_result" in st.session_state:
    del st.session_state["what_if_result"]

if "what_if_result" in st.session_state:
    result = st.session_state["what_if_result"]

    readiness_gain = result["readiness"] - strategy.readiness_score
    months_saved = strategy.estimated_months_to_ready - result["months_to_ready"]

    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric(
        "New Readiness",
        f"{result['readiness']:.1f}%",
        f"{readiness_gain:+.1f}%",
    )
    wc2.metric("New Verdict", result["verdict"])
    wc3.metric("New Fastest Role", format_role_name(result["fastest_role"]))
    wc4.metric("Months Saved", f"{max(months_saved, 0)}")

    added_skills_text = ", ".join(result["skills"])
    st.info(
        f"If you add **{added_skills_text}**, your readiness changes from "
        f"**{strategy.readiness_score:.1f}%** to **{result['readiness']:.1f}%**."
    )

st.subheader("🔁 Strategy Recommendation")
if strategy.fastest_role.lower() != baseline_target_role.lower():
    st.info(f"Fastest realistic role: **{format_role_name(strategy.fastest_role)}**")
    st.write(
        f"**Compressed path:** {format_role_name(strategy.fastest_role)} -> {format_role_name(baseline_target_role)}"
    )
else:
    st.success("Your target role is already the best current path.")

st.subheader("🧭 Closest Alternative Roles")
if strategy.alternative_roles:
    for role, score in strategy.alternative_roles:
        st.write(f"- **{format_role_name(role)}** — similarity score: {score:.1f}%")
else:
    st.write("No alternative roles found.")

st.subheader("🛠️ Suggested Portfolio Project")
st.markdown(f"**{project_recommendation['title']}**")
st.write(project_recommendation["summary"])

st.markdown("**What you will build**")
for item in project_recommendation["deliverables"]:
    st.write(f"- {item}")

st.markdown("**Why this project matters**")
for item in project_recommendation["why_it_matters"]:
    st.write(f"- {item}")

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

st.subheader("📊 Skill Importance Breakdown")
if strategy.bottlenecks:
    importance_df = pd.DataFrame(
        strategy.bottlenecks[:8],
        columns=["Skill", "Weight"],
    ).sort_values("Weight", ascending=False)

    fig_importance = px.pie(
        importance_df,
        names="Skill",
        values="Weight",
        title="Where Your Gap Comes From",
    )
    st.plotly_chart(fig_importance, use_container_width=True)

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

st.subheader("📚 Recommended Courses (Curated)")
if recommended_courses:
    for course in recommended_courses:
        st.markdown(
            f"- **{course.skill}** -> [{course.course_title}]({course.url})  \n"
            f"  {course.provider} | {course.level} | {course.duration_hours}h | {course.price_type}"
        )
else:
    st.write("No course recommendations found.")

st.subheader("🗓️ Roadmap")
total_weeks = baseline_user_estimated_months * 4
compact_roadmap = build_compact_roadmap(strategy.roadmap, total_weeks)

st.caption(f"Generated for **{total_weeks} weeks** based on your selected timeframe.")

for interval, tasks in compact_roadmap:
    st.markdown(f"**{interval}**")
    for task in tasks:
        st.write(f"- {task}")

st.subheader("📄 Export Report")
pdf_bytes = build_roleforge_report_pdf(
    target_role=format_role_name(baseline_target_role),
    user_skills=baseline_user_skills,
    readiness_score=strategy.readiness_score,
    reality_verdict=strategy.reality_verdict,
    fastest_role=format_role_name(strategy.fastest_role),
    confidence=strategy.confidence,
    estimated_months_to_ready=strategy.estimated_months_to_ready,
    matched_skills=strategy.matched_skills,
    bottlenecks=strategy.bottlenecks,
    roadmap=compact_roadmap,
)

st.download_button(
    label="Download PDF Report",
    data=pdf_bytes,
    file_name="roleforge_report.pdf",
    mime="application/pdf",
    use_container_width=True,
)

with st.expander("Technical summary"):
    st.write("User skills:", baseline_user_skills)
    st.write("Target role:", baseline_target_role)
    st.write("Hours/week:", baseline_hours_per_week)
    st.write("Loaded roles:", raw_roles)
    st.write("Input mode:", baseline_input_mode)
    st.write("CV skill matches:", baseline_extracted_skill_matches)
    st.write("LLM skill mapping enabled:", use_llm_skill_mapping)
    st.write("Alternative roles:", strategy.alternative_roles)
    st.write("Market insights:", market_insights)
    st.write("Strategy:", strategy)