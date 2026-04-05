from __future__ import annotations

from app.loaders.role_loader import RoleLoader


def test_role_loader_supports_txt_json_yaml_and_skips_invalid(tmp_path) -> None:
    roles_dir = tmp_path / "roles"
    roles_dir.mkdir()

    (roles_dir / "doctor.yaml").write_text(
        "\n".join(
            [
                "name: doctor",
                "description: Medical explainer",
                "system_prompt: Provide careful health information.",
                "constraints:",
                "  - Stay grounded",
            ]
        ),
        encoding="utf-8",
    )
    (roles_dir / "lawyer.json").write_text(
        '{"description":"Legal explainer","system_prompt":"Use the provided legal context."}',
        encoding="utf-8",
    )
    (roles_dir / "coding_assistant.txt").write_text(
        "You are a grounded coding assistant.",
        encoding="utf-8",
    )
    (roles_dir / "broken.yaml").write_text("[]", encoding="utf-8")

    loader = RoleLoader(roles_dir)
    roles, errors = loader.load_roles()

    assert set(roles) == {"doctor", "lawyer", "coding_assistant"}
    assert roles["coding_assistant"].system_prompt == "You are a grounded coding assistant."
    assert roles["lawyer"].name == "lawyer"
    assert errors
    assert "broken.yaml" in errors[0]
