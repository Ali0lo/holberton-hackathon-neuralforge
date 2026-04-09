from typing import Dict, List, Tuple


def _top_missing_skills(bottlenecks: List[Tuple[str, float]], n: int = 3) -> List[str]:
    return [skill for skill, _ in bottlenecks[:n]]


def generate_project_recommendation(
    target_role: str,
    matched_skills: List[str],
    bottlenecks: List[Tuple[str, float]],
) -> Dict[str, object]:
    role = str(target_role).strip().lower()
    missing = _top_missing_skills(bottlenecks, n=3)

    if role in {"ai engineer", "machine learning engineer", "data scientist"}:
        return {
            "title": "End-to-End Intelligent Prediction App",
            "summary": (
                "Build a practical ML application that takes raw data, trains a model, "
                "serves predictions through a simple interface, and explains the results."
            ),
            "deliverables": [
                "Data cleaning and feature engineering pipeline",
                "Model training notebook or script",
                "Evaluation report with metrics and error analysis",
                "Simple Streamlit app for predictions",
                f"Include focus on: {', '.join(missing) if missing else 'core ML workflow'}",
            ],
            "why_it_matters": [
                "Shows you can move beyond theory into working systems",
                "Demonstrates readiness for applied ML or AI tasks",
                "Creates a portfolio piece recruiters can understand quickly",
            ],
        }

    if role in {"data analyst", "data engineer"}:
        return {
            "title": "Business Intelligence Pipeline and Dashboard",
            "summary": (
                "Build a small analytics system that ingests raw data, transforms it, "
                "and presents useful insights in a dashboard."
            ),
            "deliverables": [
                "SQL-based analysis or data transformation layer",
                "Clean reporting dataset or ETL pipeline",
                "Interactive dashboard with key metrics",
                "Short business insights summary",
                f"Include focus on: {', '.join(missing) if missing else 'analysis and reporting'}",
            ],
            "why_it_matters": [
                "Demonstrates both technical execution and decision support",
                "Makes your work easy to explain to non-technical people",
                "Fits well for analyst and data engineering interviews",
            ],
        }

    if role in {
        "cybersecurity analyst",
        "cybersecurity engineer",
        "soc analyst",
        "penetration tester",
    }:
        return {
            "title": "Security Monitoring or Assessment Lab",
            "summary": (
                "Build a small security project that simulates detection, monitoring, "
                "or vulnerability assessment in a controlled environment."
            ),
            "deliverables": [
                "Documented lab environment or attack/defense scenario",
                "Detection rules, findings, or assessment notes",
                "Short incident or vulnerability report",
                "Architecture diagram or workflow summary",
                f"Include focus on: {', '.join(missing) if missing else 'security operations fundamentals'}",
            ],
            "why_it_matters": [
                "Shows practical security thinking instead of only certification knowledge",
                "Makes your profile more credible for SOC and security roles",
                "Creates interview material you can talk through confidently",
            ],
        }

    if role in {"backend engineer", "frontend engineer", "full stack engineer", "software engineer"}:
        return {
            "title": "Production-Style Web Application",
            "summary": (
                "Build a full-featured application with a clean UI, backend logic, "
                "data storage, and deployment-ready structure."
            ),
            "deliverables": [
                "Frontend interface",
                "Backend API or service layer",
                "Database integration",
                "Authentication or role-based features",
                f"Include focus on: {', '.join(missing) if missing else 'core engineering workflow'}",
            ],
            "why_it_matters": [
                "Demonstrates real product-building ability",
                "Shows architecture, implementation, and polish together",
                "Gives you a strong portfolio piece for software roles",
            ],
        }

    if role in {"devops engineer", "cloud engineer", "mlops engineer"}:
        return {
            "title": "Deployment and Automation Project",
            "summary": (
                "Build a project that packages an application, automates delivery, "
                "and shows reproducible deployment or monitoring."
            ),
            "deliverables": [
                "Dockerized application or service",
                "CI/CD workflow",
                "Deployment documentation",
                "Monitoring or logging setup",
                f"Include focus on: {', '.join(missing) if missing else 'automation and reliability'}",
            ],
            "why_it_matters": [
                "Shows operational thinking, not just coding",
                "Makes your skills easier to validate during a demo",
                "Fits strongly with DevOps, cloud, and MLOps expectations",
            ],
        }

    return {
        "title": "Applied Portfolio Project",
        "summary": (
            "Build a practical project aligned with your target role that combines "
            "at least one current strength and one missing skill."
        ),
        "deliverables": [
            "Working implementation",
            "Documentation and README",
            "Short reflection on decisions and trade-offs",
            f"Include focus on: {', '.join(missing) if missing else 'your top skill gaps'}",
        ],
        "why_it_matters": [
            "Turns skill gaps into visible progress",
            "Creates proof of work instead of only claims",
            "Improves interview storytelling and portfolio quality",
        ],
    }