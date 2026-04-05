"""Custom frontmatter schema definitions per tenant."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Maps type name strings to Python types for validation
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
}

SUPPORTED_TYPES = frozenset(_TYPE_MAP.keys())


class SchemaManager:
    """Manage custom frontmatter field schemas per tenant.

    Schema definitions are stored in
    ``{tenant_dir}/.smartrag/custom_schema.json``.
    """

    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _schema_path(self, tenant_id: str) -> str:
        return os.path.join(
            self._base_dir, "tenants", tenant_id, ".smartrag", "custom_schema.json"
        )

    def _load(self, tenant_id: str) -> dict[str, dict[str, Any]]:
        """Load schema definitions. Returns {field_name: field_def}."""
        path = self._schema_path(tenant_id)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.exception("Failed to load custom schema for tenant %s", tenant_id)
            return {}

    def _save(self, tenant_id: str, schema: dict[str, dict[str, Any]]) -> None:
        path = self._schema_path(tenant_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def define_field(
        self,
        tenant_id: str,
        field_name: str,
        field_type: str,
        required: bool = False,
        default: Any = None,
        description: str = "",
    ) -> dict[str, Any]:
        """Define or update a custom frontmatter field.

        Parameters
        ----------
        field_name:
            Name of the frontmatter field. Must be a valid identifier.
        field_type:
            One of: "string", "int", "float", "bool", "list".
        required:
            Whether the field must be present during validation.
        default:
            Default value used when the field is absent and not required.
        description:
            Human-readable description of the field's purpose.

        Raises
        ------
        ValueError
            If field_type is unsupported or field_name is invalid.
        """
        if field_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported field type '{field_type}'. "
                f"Must be one of: {', '.join(sorted(SUPPORTED_TYPES))}"
            )

        if not field_name or not field_name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Invalid field name '{field_name}'. "
                "Use only alphanumeric characters, hyphens, and underscores."
            )

        # Validate default matches declared type if provided
        if default is not None:
            expected_type = _TYPE_MAP[field_type]
            if not isinstance(default, expected_type):
                raise ValueError(
                    f"Default value type mismatch: expected {field_type}, "
                    f"got {type(default).__name__}"
                )

        field_def: dict[str, Any] = {
            "field_type": field_type,
            "required": required,
            "default": default,
            "description": description,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        schema = self._load(tenant_id)
        schema[field_name] = field_def
        self._save(tenant_id, schema)
        logger.info(
            "Defined schema field '%s' (type=%s) for tenant %s",
            field_name,
            field_type,
            tenant_id,
        )
        return {"field_name": field_name, **field_def}

    def list_fields(self, tenant_id: str) -> list[dict[str, Any]]:
        """Return all custom schema field definitions for a tenant."""
        schema = self._load(tenant_id)
        return [
            {"field_name": name, **definition}
            for name, definition in schema.items()
        ]

    def remove_field(self, tenant_id: str, field_name: str) -> bool:
        """Remove a custom schema field. Returns True if it existed."""
        schema = self._load(tenant_id)
        if field_name not in schema:
            return False
        del schema[field_name]
        self._save(tenant_id, schema)
        logger.info("Removed schema field '%s' for tenant %s", field_name, tenant_id)
        return True

    def validate_frontmatter(
        self, tenant_id: str, frontmatter: dict[str, Any]
    ) -> None:
        """Validate frontmatter against the tenant's custom schema.

        Checks that all required fields are present and all present
        fields match their declared types.

        Raises
        ------
        ValueError
            With a detailed message listing all validation errors.
        """
        schema = self._load(tenant_id)
        if not schema:
            return  # No schema defined — everything is valid

        errors: list[str] = []

        for field_name, definition in schema.items():
            required = definition.get("required", False)
            field_type = definition.get("field_type", "string")
            expected_type = _TYPE_MAP.get(field_type)

            if field_name not in frontmatter:
                if required:
                    errors.append(f"Missing required field '{field_name}'")
                continue

            value = frontmatter[field_name]
            if expected_type and not isinstance(value, expected_type):
                # Allow int where float is expected
                if field_type == "float" and isinstance(value, int):
                    continue
                errors.append(
                    f"Field '{field_name}' expected type '{field_type}', "
                    f"got '{type(value).__name__}'"
                )

        if errors:
            raise ValueError(
                f"Frontmatter validation failed: {'; '.join(errors)}"
            )
