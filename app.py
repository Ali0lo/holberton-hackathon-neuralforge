from __future__ import annotations

import streamlit as st
from roleforge import RoleForgeEngine


@st.cache_resource
def load_engine() -> RoleForgeEngine:
    return RoleForgeEngine.from_csv("data/role_skill_weights.csv")


engine = load_engine()

st.set_page_config(page_title="RoleForge Lite", page_icon="🚀", layout="wide")
st.title("🚀 RoleForge Lite")
st.caption("Career Reality Simulator for a hackathon MVP")

with st.sidebar:
    st.header("Inputs")
    current_skills_raw = st.text_area(
        "Current skills",
        placeholder="Python, SQL, Pandas, APIs",
        height=140,
    )
    target_role = st.selectbox("Target role", options=engine.roles, index=0)
    hours_per_week = st.slider("Hours per week", min_value=1, max_value=25, value=8)
    run = st.button("Run simulation", type="primary")

if run:
    current_skills = [s.strip() for s in current_skills_raw.split(",") if s.strip()]
    if not current_skills:
        st.warning("Enter at least one skill.")
    else:
        result = engine.simulate(current_skills, target_role, hours_per_week)

        c1, c2, c3 = st.columns(3)
        c1.metric("Readiness score", f"{result.readiness_score}%")
        c2.metric("3-month projection", f"{result.projection_3_months}%")
        c3.metric("6-month projection", f"{result.projection_6_months}%")

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Skill gap analysis")
            st.write("**Matched skills:**", ", ".join(result.matched_skills) if result.matched_skills else "None")
            st.write("**Missing skills:**", ", ".join(result.missing_skills) if result.missing_skills else "None")
            st.dataframe(result.prioritized_gaps, use_container_width=True)

        with right:
            st.subheader("Reality check")
            st.info(result.reality_check)
            if result.alternative_role:
                st.success(
                    f"Closest short-term role: {result.alternative_role} ({result.alternative_role_readiness}%)"
                )

        st.subheader("4-week roadmap")
        for item in result.roadmap:
            st.markdown(f"**Week {item['week']} — {item['focus']}**  ")
            st.write(item["goal"])
else:
    st.write("Enter skills, choose a role, and run the simulation.")
