"""Helpers for conservative Venetian dataset normalization."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from constants import TASK_PREFIX


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "venetian_normalization.json"
TEXT_FIELDS = ("source_text", "target_text", "translation_prompt")


@dataclass
class NormalizationStats:
    total_records: int = 0
    changed_records: int = 0
    field_changes: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.field_changes is None:
            self.field_changes = {}

    def bump(self, field: str) -> None:
        self.field_changes[field] = self.field_changes.get(field, 0) + 1


def load_profile(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def _build_token_pattern(mapping: dict[str, str]) -> re.Pattern[str] | None:
    if not mapping:
        return None
    choices = sorted(mapping, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(choice) for choice in choices) + r")\b")


class VenetianNormalizer:
    def __init__(self, profile: dict) -> None:
        self.profile = profile
        self.character_map = str.maketrans(profile.get("character_map", {}))
        self.field_maps = profile.get("field_maps", {})
        self.field_defaults = profile.get("field_defaults", {})
        self.standardize_translation_prompt = profile.get("standardize_translation_prompt", False)
        self.token_maps = profile.get("token_maps", {})
        self.token_patterns = {
            field: _build_token_pattern(mapping)
            for field, mapping in self.token_maps.items()
        }

    def normalize_record(self, record: dict) -> tuple[dict, dict[str, bool]]:
        normalized = dict(record)
        changed_fields: dict[str, bool] = {}

        for field, mapping in self.field_maps.items():
            value = normalized.get(field, "")
            if isinstance(value, str) and value in mapping:
                updated = mapping[value]
                if updated != value:
                    normalized[field] = updated
                    changed_fields[field] = True

        for field, default_value in self.field_defaults.items():
            value = normalized.get(field, "")
            if not value:
                normalized[field] = default_value
                changed_fields[field] = True

        for field in TEXT_FIELDS:
            value = normalized.get(field)
            if not isinstance(value, str):
                continue

            updated = self.normalize_text(value, field)
            if updated != value:
                normalized[field] = updated
                changed_fields[field] = True

        if self.standardize_translation_prompt:
            source_text = normalized.get("source_text")
            if isinstance(source_text, str) and source_text.strip():
                standard_prompt = TASK_PREFIX + source_text
                if normalized.get("translation_prompt") != standard_prompt:
                    normalized["translation_prompt"] = standard_prompt
                    changed_fields["translation_prompt"] = True

        return normalized, changed_fields

    def normalize_text(self, text: str, field: str) -> str:
        updated = unicodedata.normalize("NFC", text)
        updated = updated.translate(self.character_map)
        updated = re.sub(r"\s+", " ", updated).strip()
        updated = re.sub(r"\s+([,.;:!?])", r"\1", updated)
        updated = re.sub(r"([¿¡(\[])\s+", r"\1", updated)

        pattern = self.token_patterns.get(field)
        mapping = self.token_maps.get(field, {})
        if pattern is not None:
            updated = pattern.sub(lambda match: mapping[match.group(0)], updated)

        return updated


def normalize_records(records: list[dict], normalizer: VenetianNormalizer) -> tuple[list[dict], dict]:
    stats = NormalizationStats(total_records=len(records))
    normalized_records = []

    for record in records:
        normalized, changed_fields = normalizer.normalize_record(record)
        normalized_records.append(normalized)
        if changed_fields:
            stats.changed_records += 1
            for field in changed_fields:
                stats.bump(field)

    return normalized_records, {
        "total_records": stats.total_records,
        "changed_records": stats.changed_records,
        "field_changes": dict(sorted(stats.field_changes.items())),
    }
