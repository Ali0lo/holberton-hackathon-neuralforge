import json
import os
import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def _ollama_generate(prompt: str, timeout: int = 25) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception:
        return ""


def _ollama_stream(prompt: str, timeout: int = 60):
    """
    Yield text chunks from Ollama streaming response.
    """
    try:
        with requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
            },
            timeout=timeout,
            stream=True,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                    if data.get("done", False):
                        break
                except Exception:
                    continue
    except Exception:
        return


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
    return _ollama_generate(prompt, timeout=20)


def stream_roleforge_explanation(
    target_role: str,
    user_skills: list[str],
    readiness_score: float,
    bottlenecks: list[tuple[str, float]],
    fastest_role: str,
):
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
    yield from _ollama_stream(prompt, timeout=40)


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
    return _ollama_generate(prompt, timeout=25)


def stream_cv_overview(
    cv_text: str,
    extracted_skills: list[str],
    target_role: str,
):
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
    yield from _ollama_stream(prompt, timeout=50)


def llm_map_user_skills(
    raw_skills: list[str],
    target_role: str,
    allowed_skills: list[str],
) -> list[str]:
    """
    Map noisy/raw user skills into the allowed canonical skill vocabulary.
    Example:
      ai -> artificial intelligence / machine learning / large language models
      supervised learning -> machine learning
    """
    if not raw_skills or not allowed_skills:
        return []

    prompt = f"""
You are a skill normalizer for a career simulator.

Target role: {target_role}

Raw user skills:
{json.dumps(raw_skills, ensure_ascii=False)}

Allowed canonical skills:
{json.dumps(allowed_skills, ensure_ascii=False)}

Task:
Return ONLY a JSON array of canonical skills from the allowed list that best match the raw user skills.
Rules:
- only use items from the allowed canonical skills list
- map synonyms when appropriate
- include skills only if they are genuinely supported by the raw input
- do not invent unrelated skills
- output JSON only
"""

    text = _ollama_generate(prompt, timeout=25)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            allowed_lower = {s.lower(): s for s in allowed_skills}
            cleaned = []
            seen = set()
            for item in parsed:
                key = str(item).strip().lower()
                if key in allowed_lower and key not in seen:
                    cleaned.append(allowed_lower[key])
                    seen.add(key)
            return cleaned
    except Exception:
        return []

    return []


def llm_extract_cv_skills(
    cv_text: str,
    target_role: str,
    allowed_skills: list[str],
) -> list[str]:
    """
    Extract canonical skills from CV text, restricted to the allowed vocabulary.
    """
    if not cv_text or not allowed_skills:
        return []

    prompt = f"""
You are a resume skill extractor for a career simulator.

Target role: {target_role}

Allowed canonical skills:
{json.dumps(allowed_skills, ensure_ascii=False)}

CV text:
{cv_text[:7000]}

Task:
Return ONLY a JSON array of skills from the allowed canonical skills list that are clearly supported by the CV text.
Rules:
- only use items from the allowed canonical skills list
- do not infer skills without evidence
- do not output anything except JSON
"""

    text = _ollama_generate(prompt, timeout=30)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            allowed_lower = {s.lower(): s for s in allowed_skills}
            cleaned = []
            seen = set()
            for item in parsed:
                key = str(item).strip().lower()
                if key in allowed_lower and key not in seen:
                    cleaned.append(allowed_lower[key])
                    seen.add(key)
            return cleaned
    except Exception:
        return []

    return []


def llm_rerank_courses(
    target_role: str,
    bottlenecks: list[tuple[str, float]],
    course_candidates: list[dict],
) -> list[dict]:
    """
    Rerank fetched course candidates for better relevance.
    Returns reordered list of the SAME items only.
    """
    if not course_candidates:
        return []

    simplified = []
    for idx, item in enumerate(course_candidates):
        simplified.append(
            {
                "id": idx,
                "skill": item.get("skill", ""),
                "course_title": item.get("course_title", ""),
                "provider": item.get("provider", ""),
                "url": item.get("url", ""),
                "level": item.get("level", ""),
                "duration_hours": item.get("duration_hours", 0.0),
                "price_type": item.get("price_type", "unknown"),
                "quality_score": item.get("quality_score", 0.0),
            }
        )

    prompt = f"""
You are ranking courses for a user.

Target role: {target_role}
Top bottlenecks: {", ".join([b[0] for b in bottlenecks[:5]])}

Candidates:
{json.dumps(simplified, ensure_ascii=False)}

Task:
Return ONLY a JSON array of candidate ids in best-to-worst order.
Rules:
- prioritize relevance to bottlenecks and target role
- prefer practical, reputable, beginner/intermediate-friendly content
- output only JSON
"""

    text = _ollama_generate(prompt, timeout=25)
    if not text:
        return course_candidates

    try:
        ranked_ids = json.loads(text)
        if isinstance(ranked_ids, list):
            by_id = {i: item for i, item in enumerate(course_candidates)}
            reranked = []
            seen = set()
            for rid in ranked_ids:
                if isinstance(rid, int) and rid in by_id and rid not in seen:
                    reranked.append(by_id[rid])
                    seen.add(rid)
            for i, item in enumerate(course_candidates):
                if i not in seen:
                    reranked.append(item)
            return reranked
    except Exception:
        return course_candidates

    return course_candidates