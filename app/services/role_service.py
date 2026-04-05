from __future__ import annotations

import logging

from app.core.models import RoleProfile
from app.loaders.role_loader import RoleLoader


class RoleService:
    def __init__(self, role_loader: RoleLoader) -> None:
        self.role_loader = role_loader
        self.logger = logging.getLogger(__name__)
        self._roles: dict[str, RoleProfile] = {}

    def load_roles(self) -> list[RoleProfile]:
        roles, errors = self.role_loader.load_roles()
        for error in errors:
            self.logger.warning("Skipping invalid role file", extra={"role_error": error})
        self._roles = roles
        return self.list_roles()

    def list_roles(self) -> list[RoleProfile]:
        return sorted(self._roles.values(), key=lambda role: role.name.lower())

    def get_role(self, role_name: str | None) -> RoleProfile | None:
        if not role_name:
            return None
        return self._roles.get(role_name)
