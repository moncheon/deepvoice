"""Versioned source-audio provenance ledger and leakage-safe source splitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .manifest import sha256_file


@dataclass(frozen=True)
class SourceRecord:
    """One reviewed, component-level audio source used to build training mixtures.

    Fields through ``generator_revision`` are compatible with the original v1
    source ledger. Production v2 ledgers add an explicit split and enough
    download/generation lineage to reproduce or reject every source.
    """

    source_id: str
    path: str
    sha256: str
    component: str
    fake: int
    source_dataset: str
    source_item_id: str
    license_id: str
    consent_basis: str
    identity_group: str
    content_group: str
    mix_split_group: str | None
    generator_family: str | None
    generator_model: str | None
    generator_revision: str | None
    format_version: int = 2
    source_revision: str = ""
    source_url: str = ""
    license_url: str = ""
    language: str = "und"
    parent_source_ids: tuple[str, ...] = ()
    prompt_id: str | None = None
    generation_seed: int | None = None
    split: str | None = None
    separator_model: str | None = None
    separator_revision: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SourceRecord":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        extra = sorted(set(payload) - known)
        if extra:
            raise ValueError(f"source fields mismatch; extra={extra}")
        required = {
            "source_id", "path", "sha256", "component", "fake", "source_dataset",
            "source_item_id", "license_id", "consent_basis", "identity_group",
            "content_group", "generator_family", "generator_model", "generator_revision",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"source fields mismatch; missing={missing}")
        normalized = dict(payload)
        if "format_version" not in normalized:
            normalized["format_version"] = 1
        normalized.setdefault("mix_split_group", None)
        parents = normalized.get("parent_source_ids", ())
        if not isinstance(parents, (list, tuple)):
            raise ValueError("parent_source_ids must be a JSON array")
        normalized["parent_source_ids"] = tuple(str(item) for item in parents)
        record = cls(**normalized)  # type: ignore[arg-type]
        record.validate()
        return record

    def validate(self) -> None:
        if self.format_version not in {1, 2}:
            raise ValueError(f"unsupported source format_version: {self.format_version}")
        if self.component not in {"voice", "music"} or self.fake not in {0, 1}:
            raise ValueError(f"invalid component/fake label for {self.source_id}")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in self.sha256):
            raise ValueError(f"invalid source SHA-256 for {self.source_id}")
        required = (
            self.source_id, self.path, self.source_dataset, self.source_item_id,
            self.license_id, self.consent_basis, self.identity_group, self.content_group,
        )
        if any(not str(value).strip() for value in required):
            raise ValueError(f"empty provenance field for {self.source_id}")
        if "REPLACE" in self.license_id.upper() or "UNKNOWN" in self.license_id.upper():
            raise ValueError(f"license must be reviewed for {self.source_id}")
        if self.fake and not self.generator_family:
            raise ValueError(f"fake source needs generator_family: {self.source_id}")
        if not self.fake and any((self.generator_family, self.generator_model, self.generator_revision)):
            raise ValueError(f"real source cannot claim a generator: {self.source_id}")
        if self.format_version == 1 and not self.mix_split_group:
            raise ValueError(f"v1 source needs mix_split_group: {self.source_id}")
        if self.format_version == 2:
            if not self.source_revision or not self.source_url or not self.license_url:
                raise ValueError(f"v2 source needs revision/source/license URLs: {self.source_id}")
            if self.split not in {None, "train", "dev", "holdout"}:
                raise ValueError(f"invalid source split for {self.source_id}: {self.split}")
        if (self.separator_model is None) != (self.separator_revision is None):
            raise ValueError(f"separator model/revision must be paired: {self.source_id}")


def read_source_manifest(path: str | Path, *, audit_files: bool = True) -> list[SourceRecord]:
    manifest_path = Path(path)
    records: list[SourceRecord] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = SourceRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid source manifest line {line_number}: {error}") from error
            if record.source_id in seen:
                raise ValueError(f"duplicate source_id: {record.source_id}")
            seen.add(record.source_id)
            if audit_files:
                source_path = Path(record.path)
                if not source_path.is_absolute():
                    source_path = manifest_path.parent / source_path
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                if sha256_file(source_path).lower() != record.sha256.lower():
                    raise ValueError(f"source hash mismatch: {record.source_id}")
            records.append(record)
    if not records:
        raise ValueError("source manifest is empty")
    return records


def write_source_manifest(path: str | Path, records: Iterable[SourceRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record.validate()
            if record.source_id in seen:
                raise ValueError(f"duplicate source_id: {record.source_id}")
            seen.add(record.source_id)
            payload = asdict(record)
            payload["parent_source_ids"] = list(record.parent_source_ids)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def assign_source_splits(
    records: Iterable[SourceRecord],
    *,
    seed: int = 2608,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    holdout_generator_families: set[str] | None = None,
    holdout_source_ids: set[str] | None = None,
) -> list[SourceRecord]:
    """Assign connected identity/content/parent groups before any augmentation."""

    records = list(records)
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("split ratios must leave a positive holdout ratio")
    ids = {record.source_id for record in records}
    if len(ids) != len(records):
        raise ValueError("duplicate source IDs cannot be split")
    union = _UnionFind(ids)
    first_by_key: dict[str, str] = {}
    for record in records:
        keys = (
            f"identity:{record.identity_group}",
            f"content:{record.content_group}",
            f"audio-sha256:{record.sha256.lower()}",
            *(f"parent:{parent}" for parent in record.parent_source_ids),
        )
        for key in keys:
            prior = first_by_key.setdefault(key, record.source_id)
            union.union(prior, record.source_id)
        for parent in record.parent_source_ids:
            if parent in ids:
                union.union(parent, record.source_id)

    grouped: dict[str, list[SourceRecord]] = {}
    for record in records:
        grouped.setdefault(union.find(record.source_id), []).append(record)
    forced_generators = holdout_generator_families or set()
    forced_sources = holdout_source_ids or set()
    assigned: list[SourceRecord] = []
    for root, group in sorted(grouped.items()):
        explicit = {record.split for record in group if record.split is not None}
        if len(explicit) > 1:
            raise ValueError(f"connected source group has conflicting splits: {root}")
        forced = any(
            record.source_id in forced_sources or record.generator_family in forced_generators
            for record in group
        )
        if forced:
            split = "holdout"
        elif explicit:
            split = next(iter(explicit))
        else:
            digest = hashlib.sha256(f"{seed}:{root}".encode()).digest()
            position = int.from_bytes(digest[:8], "big") / 2**64
            split = "train" if position < train_ratio else "dev" if position < train_ratio + dev_ratio else "holdout"
        assigned.extend(replace(record, split=split) for record in group)
    return sorted(assigned, key=lambda item: item.source_id)


def audit_source_split_leakage(records: Iterable[SourceRecord]) -> dict[str, object]:
    """Return a compact report and fail when a leakage key crosses splits."""

    records = list(records)
    crossings: dict[str, set[str]] = {}
    for record in records:
        if record.split is None:
            raise ValueError(f"source has no assigned split: {record.source_id}")
        keys = {
            f"identity:{record.identity_group}",
            f"content:{record.content_group}",
            f"audio-sha256:{record.sha256.lower()}",
            *(f"parent:{parent}" for parent in record.parent_source_ids),
        }
        for key in keys:
            crossings.setdefault(key, set()).add(record.split)
    leaked = {key: sorted(splits) for key, splits in crossings.items() if len(splits) > 1}
    if leaked:
        first = next(iter(leaked.items()))
        raise ValueError(f"source split leakage: {first[0]} -> {first[1]}")
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split or "unassigned"] = counts.get(record.split or "unassigned", 0) + 1
    return {"records": len(records), "split_counts": counts, "leakage_keys": 0}
