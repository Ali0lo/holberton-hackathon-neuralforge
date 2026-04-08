from roleforge import RoleForgeEngine


def main() -> None:
    engine = RoleForgeEngine.from_csv("data/role_skill_weights.csv")

    user_skills = ["Python", "SQL", "Pandas", "APIs"]
    target_role = "AI Engineer"
    hours_per_week = 8

    result = engine.simulate(user_skills, target_role, hours_per_week)

    print("=== RoleForge Demo ===")
    print(f"Target role: {result.target_role}")
    print(f"Readiness: {result.readiness_score}%")
    print(f"Matched skills: {', '.join(result.matched_skills) if result.matched_skills else 'None'}")
    print(f"Missing skills: {', '.join(result.missing_skills) if result.missing_skills else 'None'}")
    print(f"3-month projection: {result.projection_3_months}%")
    print(f"6-month projection: {result.projection_6_months}%")
    print(f"Reality check: {result.reality_check}")
    print(f"Alternative role: {result.alternative_role} ({result.alternative_role_readiness}%)")
    print("Roadmap:")
    for item in result.roadmap:
        print(f"  Week {item['week']}: {item['focus']} -> {item['goal']}")


if __name__ == "__main__":
    main()
