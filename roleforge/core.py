from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    target_role: str
    readiness_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    prioritized_gaps: list[dict]
    projection_3_months: float
    projection_6_months: float
    reality_check: str
    alternative_role: str | None
    alternative_role_readiness: float | None
    roadmap: list[dict]


class RoleForgeEngine:
    """Hackathon-friendly simulation engine.

    This engine intentionally starts simple:
    - weighted skill overlap for readiness
    - bounded time-based progress simulation
    - adjacent-role recommendation by weighted overlap

    It is designed to be explainable during a demo.
    """

    def __init__(self, role_skill_weights: pd.DataFrame):
        required_columns = {"role", "skill", "weight"}
        missing = required_columns - set(role_skill_weights.columns)
        if missing:
            raise ValueError(f"role_skill_weights missing columns: {sorted(missing)}")

        df = role_skill_weights.copy()
        df["role"] = df["role"].astype(str).str.strip()
        df["skill"] = df["skill"].astype(str).str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="raise")

        self.role_skill_weights = df
        self.roles = sorted(df["role"].unique().tolist())

    @classmethod
    def from_csv(cls, path: str | Path) -> "RoleForgeEngine":
        return cls(pd.read_csv(path))

    def simulate(
        self,
        current_skills: Iterable[str],
        target_role: str,
        hours_per_week: float,
    ) -> SimulationResult:
        normalized_skills = _normalize_skills(current_skills)
        target_df = self._get_role_df(target_role)

        readiness_payload = self._compute_readiness(target_df, normalized_skills)
        current_readiness = readiness_payload["readiness_score"]
        prioritized_gaps = readiness_payload["prioritized_gaps"]
        missing_weight = readiness_payload["missing_weight"]

        projection_3 = self._project_readiness(current_readiness, missing_weight, hours_per_week, months=3)
        projection_6 = self._project_readiness(current_readiness, missing_weight, hours_per_week, months=6)
        alt_role, alt_score = self._recommend_alternative_role(normalized_skills, exclude_role=target_role)
        reality_check = self._build_reality_check(target_role, current_readiness, projection_3, hours_per_week, alt_role, alt_score)
        roadmap = self._build_roadmap(prioritized_gaps, hours_per_week)

        return SimulationResult(
            target_role=target_role,
            readiness_score=round(current_readiness, 1),
            matched_skills=readiness_payload["matched_skills"],
            missing_skills=readiness_payload["missing_skills"],
            prioritized_gaps=prioritized_gaps,
            projection_3_months=round(projection_3, 1),
            projection_6_months=round(projection_6, 1),
            reality_check=reality_check,
            alternative_role=alt_role,
            alternative_role_readiness=round(alt_score, 1) if alt_score is not None else None,
            roadmap=roadmap,
        )

    def _get_role_df(self, role: str) -> pd.DataFrame:
        df = self.role_skill_weights[self.role_skill_weights["role"] == role].copy()
        if df.empty:
            raise ValueError(f"Unknown role: {role}. Available roles: {', '.join(self.roles)}")
        return df.sort_values("weight", ascending=False).reset_index(drop=True)

    def _compute_readiness(self, role_df: pd.DataFrame, current_skills: set[str]) -> dict:
        total_weight = role_df["weight"].sum()
        role_df = role_df.copy()
        role_df["is_matched"] = role_df["skill"].apply(lambda s: s.lower() in current_skills)

        matched_df = role_df[role_df["is_matched"]]
        missing_df = role_df[~role_df["is_matched"]].sort_values("weight", ascending=False)
        matched_weight = matched_df["weight"].sum()
        missing_weight = max(total_weight - matched_weight, 0.0)

        readiness_score = 100.0 * matched_weight / total_weight if total_weight else 0.0

        prioritized_gaps = [
            {
                "skill": row["skill"],
                "weight": round(float(row["weight"]), 3),
                "priority": idx + 1,
            }
            for idx, (_, row) in enumerate(missing_df.iterrows())
        ]

        return {
            "readiness_score": readiness_score,
            "matched_skills": matched_df["skill"].tolist(),
            "missing_skills": missing_df["skill"].tolist(),
            "prioritized_gaps": prioritized_gaps,
            "missing_weight": missing_weight,
        }

    def _project_readiness(
        self,
        current_readiness: float,
        missing_weight: float,
        hours_per_week: float,
        months: int,
    ) -> float:
        hours = max(float(hours_per_week), 0.0)

        # Bounded progress curve for hackathon realism.
        # Gains increase with time and effort, but with diminishing returns.
        monthly_capture = min(0.22, 0.035 + 0.012 * hours)
        coverage_gain = missing_weight * monthly_capture * months
        readiness_gain = min(100.0 - current_readiness, coverage_gain * 100.0)

        # Soft cap: early-stage candidates should not look "job-ready" too fast.
        soft_cap = 92.0 if hours >= 15 else 85.0 if hours >= 10 else 78.0
        return min(current_readiness + readiness_gain, soft_cap)

    def _recommend_alternative_role(self, current_skills: set[str], exclude_role: str) -> tuple[str | None, float | None]:
        best_role = None
        best_score = -1.0

        for role in self.roles:
            if role == exclude_role:
                continue
            role_df = self._get_role_df(role)
            score = self._compute_readiness(role_df, current_skills)["readiness_score"]
            if score > best_score:
                best_role = role
                best_score = score

        if best_role is None:
            return None, None
        return best_role, best_score

    def _build_reality_check(
        self,
        target_role: str,
        current_readiness: float,
        projection_3: float,
        hours_per_week: float,
        alternative_role: str | None,
        alternative_score: float | None,
    ) -> str:
        if projection_3 >= 75:
            return (
                f"At {hours_per_week:g}h/week, {target_role} looks reasonably achievable for an MVP-level transition, "
                f"but you still need portfolio proof and interview readiness."
            )

        if current_readiness < 35 and alternative_role and alternative_score is not None:
            return (
                f"At {hours_per_week:g}h/week, {target_role} is unlikely to be job-ready in 3 months from your current baseline. "
                f"The closest short-term pivot is {alternative_role} ({alternative_score:.1f}% readiness today)."
            )

        return (
            f"At {hours_per_week:g}h/week, reaching full job-readiness for {target_role} in 3 months is unlikely. "
            f"Your realistic near-term goal is to reduce the top skill gaps and build one strong proof-of-work project."
        )

    def _build_roadmap(self, prioritized_gaps: list[dict], hours_per_week: float) -> list[dict]:
        gaps = prioritized_gaps[:4]
        if not gaps:
            return [
                {"week": 1, "focus": "Portfolio project", "goal": "Build a proof-of-work project aligned to the target role."},
                {"week": 2, "focus": "Interview prep", "goal": "Practice role-relevant questions and review fundamentals."},
                {"week": 3, "focus": "Portfolio polish", "goal": "Write project README, metrics, and decision trade-offs."},
                {"week": 4, "focus": "Applications", "goal": "Tailor CV and begin targeted applications."},
            ]

        pace = "aggressive" if hours_per_week >= 12 else "balanced" if hours_per_week >= 7 else "light"
        roadmap = []
        for week_idx in range(4):
            if week_idx < len(gaps):
                skill = gaps[week_idx]["skill"]
                roadmap.append(
                    {
                        "week": week_idx + 1,
                        "focus": skill,
                        "goal": f"{pace.title()} study block on {skill}, then apply it in a mini exercise.",
                    }
                )
            elif week_idx == 2:
                roadmap.append(
                    {
                        "week": week_idx + 1,
                        "focus": "Project integration",
                        "goal": "Combine newly learned skills in one scoped portfolio project.",
                    }
                )
            else:
                roadmap.append(
                    {
                        "week": week_idx + 1,
                        "focus": "Portfolio polish",
                        "goal": "Document the project, assumptions, metrics, and trade-offs.",
                    }
                )
        return roadmap


def _normalize_skills(skills: Iterable[str]) -> set[str]:
    cleaned: set[str] = set()
    for skill in skills:
        token = str(skill).strip().lower()
        if token:
            cleaned.add(token)
    return cleaned
