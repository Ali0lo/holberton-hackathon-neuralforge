import os
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def generate_roleforge_explanation(
    target_role: str,
    user_skills: list[str],
    readiness_score: float,
    bottlenecks: list[tuple[str, float]],
    fastest_role: str,
) -> str:
    prompt = f"""
You are an honest career advisor.
Write a short explanation for a user.

Target role: {target_role}
User skills: {", ".join(user_skills)}
Readiness score: {readiness_score}
Top bottlenecks: {", ".join([b[0] for b in bottlenecks[:3]])}
Fastest realistic role: {fastest_role}

Requirements:
- be honest, direct, and useful
- max 120 words
- no hype
- mention one practical next step
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception:
        return ""


def generate_cv_overview(
    cv_text: str,
    extracted_skills: list[str],
    target_role: str,
) -> str:
    prompt = f"""
You are an honest career assistant.
Write a short CV overview for a user applying toward {target_role}.

Extracted skills: {", ".join(extracted_skills)}
CV text:
{cv_text[:5000]}

Requirements:
- max 140 words
- summarize the candidate profile briefly
- mention 2 strengths
- mention 1 gap relative to the target role
- be direct and practical
- do not invent experience that is not in the CV
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception:
        return ""