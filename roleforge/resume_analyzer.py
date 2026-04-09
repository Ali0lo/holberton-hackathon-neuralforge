from typing import Dict, List, Tuple


def analyze_resume_gaps(
    target_role: str,
    matched_skills: List[str],
    bottlenecks: List[Tuple[str, float]],
) -> Dict[str, List[str]]:
    missing_keywords = [skill for skill, _ in bottlenecks[:6]]
    matched = matched_skills[:3]

    bullet_point_ideas = []

    for skill in missing_keywords[:3]:
        bullet_point_ideas.append(
            f"Add a project bullet showing hands-on work with {skill.lower()}."
        )

    if matched:
        bullet_point_ideas.append(
            f"Quantify impact using your existing strengths in {', '.join([m.lower() for m in matched])}."
        )

    bullet_point_ideas.append(
        f"Tailor your CV summary toward {target_role.lower()} with role-relevant keywords."
    )

    return {
        "missing_keywords": missing_keywords,
        "bullet_point_ideas": bullet_point_ideas,
    }