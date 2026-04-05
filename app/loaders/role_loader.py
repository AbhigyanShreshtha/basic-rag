from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.core.models import RoleProfile


class RoleLoader:
    SUPPORTED_SUFFIXES = {".txt", ".json", ".yaml", ".yml"}

    def __init__(self, roles_dir: Path) -> None:
        self.roles_dir = roles_dir

    def load_roles(self) -> tuple[dict[str, RoleProfile], list[str]]:
        roles: dict[str, RoleProfile] = {}
        errors: list[str] = []

        if not self.roles_dir.exists():
            return roles, errors

        for path in sorted(self.roles_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
                continue
            try:
                role = self.load_role_file(path)
            except (OSError, json.JSONDecodeError, yaml.YAMLError, ValidationError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            roles[role.name] = role

        return roles, errors

    def load_role_file(self, path: Path) -> RoleProfile:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return RoleProfile(name=path.stem, system_prompt=path.read_text(encoding="utf-8"))

        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text) if suffix == ".json" else yaml.safe_load(raw_text)
        if not isinstance(payload, dict):
            raise ValueError("role file must contain an object")
        if "name" not in payload:
            payload["name"] = path.stem
        return RoleProfile.model_validate(payload)
