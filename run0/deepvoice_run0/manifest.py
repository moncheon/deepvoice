"""Auditable JSONL data ledger and leakage-resistant split assignment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    path: str
    sha256: str
    source_dataset: str
    source_item_id: str
    license_id: str
    consent_basis: str
    speaker_or_artist_group: str
    content_group: str
    generator_family: str | None
    generator_model: str | None
    generator_revision: str | None
    generation_seed: int | None
    voice_present: int
    music_present: int
    voice_fake: int | None
    music_fake: int | None
    file_fake: int
    mix_mode: str
    codec_chain: str
    channel_profile: str
    sample_rate: int
    channels: int
    duration_sec: float
    split_group: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ManifestRecord":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        missing = sorted(known - payload.keys())
        extra = sorted(payload.keys() - known)
        if missing or extra:
            raise ValueError(f"manifest fields mismatch; missing={missing}, extra={extra}")
        record = cls(**payload)  # type: ignore[arg-type]
        record.validate()
        return record

    def validate(self) -> None:
        if not self.sample_id or not self.path or not self.split_group:
            raise ValueError("sample_id, path, and split_group must be non-empty")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in self.sha256):
            raise ValueError(f"invalid sha256 for {self.sample_id}")
        if self.mix_mode not in {"none", "simultaneous", "sequential"}:
            raise ValueError(f"invalid mix_mode for {self.sample_id}: {self.mix_mode}")
        for name in ("voice_present", "music_present", "file_fake"):
            if getattr(self, name) not in (0, 1):
                raise ValueError(f"{name} must be 0 or 1 for {self.sample_id}")
        for present_name, fake_name in (
            ("voice_present", "voice_fake"),
            ("music_present", "music_fake"),
        ):
            present = getattr(self, present_name)
            fake = getattr(self, fake_name)
            if present == 0 and fake is not None:
                raise ValueError(f"{fake_name} must be null when component is absent")
            if present == 1 and fake not in (0, 1):
                raise ValueError(f"{fake_name} must be 0 or 1 when component is present")
        expected_file_fake = int(bool(self.voice_fake) or bool(self.music_fake))
        if self.file_fake != expected_file_fake:
            raise ValueError(
                f"file_fake mismatch for {self.sample_id}: expected {expected_file_fake}"
            )
        if self.sample_rate <= 0 or self.channels <= 0 or self.duration_sec <= 0:
            raise ValueError(f"invalid audio metadata for {self.sample_id}")


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = ManifestRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid manifest line {line_number}: {error}") from error
            if record.sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)
            records.append(record)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record.validate()
            if record.sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def assign_group_splits(
    records: Iterable[ManifestRecord],
    seed: int = 2608,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    holdout_generator_families: set[str] | None = None,
) -> dict[str, str]:
    """Assign every split_group atomically using a stable SHA-256 hash."""

    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("split ratios must leave a positive holdout ratio")
    forced = holdout_generator_families or set()
    group_splits: dict[str, str] = {}
    assignments: dict[str, str] = {}
    for record in records:
        if record.generator_family in forced:
            split = "generator_holdout"
        else:
            digest = hashlib.sha256(f"{seed}:{record.split_group}".encode()).digest()
            position = int.from_bytes(digest[:8], "big") / 2**64
            split = "train" if position < train_ratio else "dev" if position < train_ratio + dev_ratio else "holdout"
        prior = group_splits.setdefault(record.split_group, split)
        if prior != split:
            raise ValueError(f"split_group would leak across splits: {record.split_group}")
        assignments[record.sample_id] = split
    return assignments


def assign_stratified_group_splits(
    records: Iterable[ManifestRecord],
    *,
    seed: int = 2608,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    holdout_generator_families: set[str] | None = None,
) -> dict[str, str]:
    """Balance label signatures while keeping every split_group atomic.

    Strata with at least three groups receive at least one dev and one holdout
    group, which is useful for small bootstrap and ablation corpora.
    """

    records = list(records)
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("split ratios must leave a positive holdout ratio")
    forced = holdout_generator_families or set()
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(record.split_group, []).append(record)
    assignments: dict[str, str] = {}
    strata: dict[tuple[tuple[int, int, int | None, int | None, int], ...], list[str]] = {}
    for group, group_records in grouped.items():
        if any(record.generator_family in forced for record in group_records):
            for record in group_records:
                assignments[record.sample_id] = "generator_holdout"
            continue
        signature = tuple(sorted({
            (
                record.voice_present,
                record.music_present,
                record.voice_fake,
                record.music_fake,
                record.file_fake,
            )
            for record in group_records
        }, key=str))
        strata.setdefault(signature, []).append(group)

    for signature, groups in strata.items():
        groups.sort(key=lambda group: hashlib.sha256(f"{seed}:{signature}:{group}".encode()).digest())
        count = len(groups)
        train_count = max(1, int(count * train_ratio))
        dev_count = int(count * dev_ratio)
        if count >= 3:
            dev_count = max(1, dev_count)
            holdout_count = max(1, count - train_count - dev_count)
            while train_count + dev_count + holdout_count > count and train_count > 1:
                train_count -= 1
        else:
            holdout_count = max(0, count - train_count - dev_count)
        if train_count + dev_count + holdout_count < count:
            train_count += count - train_count - dev_count - holdout_count
        split_by_group: dict[str, str] = {}
        for index, group in enumerate(groups):
            if index < train_count:
                split_by_group[group] = "train"
            elif index < train_count + dev_count:
                split_by_group[group] = "dev"
            else:
                split_by_group[group] = "holdout"
        for group in groups:
            for record in grouped[group]:
                assignments[record.sample_id] = split_by_group[group]
    return assignments


def coarse_generator_domain(record: ManifestRecord) -> tuple[str, str]:
    """Return stable voice/music generator classes without model memorization.

    A mixed record may contain multiple generator-family tokens.  The coarse
    classes deliberately collapse the many MLAAD model names into one domain,
    while retaining the two separately generated families used by this corpus.
    """

    family = record.generator_family or ""
    voice = "mms" if "MMS-TTS-KOR" in family else "mlaad" if "MLAAD:" in family else "none"
    music = "audioldm2" if "AudioLDM2" in family else "musicgen" if "MusicGen" in family else "none"
    return voice, music


def domain_label_signature(record: ManifestRecord) -> tuple[object, ...]:
    """Label signature augmented with coarse fake-generator domains."""

    voice_domain, music_domain = coarse_generator_domain(record)
    return (
        record.voice_present,
        record.music_present,
        record.voice_fake,
        record.music_fake,
        record.file_fake,
        voice_domain,
        music_domain,
    )


def assign_domain_stratified_group_splits(
    records: Iterable[ManifestRecord],
    *,
    seed: int = 2609,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
) -> dict[str, str]:
    """Create source-group-disjoint splits covering coarse generator domains.

    Unlike a generator-holdout diagnostic, this production split represents
    every available coarse generator domain in train/dev/holdout whenever a
    stratum has at least three source groups.  Exact speaker/content parent
    groups remain atomic, so the change does not introduce source leakage.
    """

    records = list(records)
    if train_ratio <= 0 or dev_ratio <= 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("split ratios must leave positive train, dev, and holdout ratios")
    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(record.split_group, []).append(record)

    strata: dict[tuple[tuple[object, ...], ...], list[str]] = {}
    for group, group_records in grouped.items():
        signature = tuple(sorted({domain_label_signature(record) for record in group_records}, key=str))
        strata.setdefault(signature, []).append(group)

    assignments: dict[str, str] = {}
    for signature, groups in strata.items():
        groups.sort(key=lambda group: hashlib.sha256(f"{seed}:{signature}:{group}".encode()).digest())
        count = len(groups)
        train_count = max(1, int(count * train_ratio))
        dev_count = max(1, int(count * dev_ratio)) if count >= 3 else 0
        holdout_count = max(1, count - train_count - dev_count) if count >= 3 else 0
        while train_count + dev_count + holdout_count > count and train_count > 1:
            train_count -= 1
        if train_count + dev_count + holdout_count < count:
            train_count += count - train_count - dev_count - holdout_count
        for index, group in enumerate(groups):
            split = (
                "train" if index < train_count
                else "dev" if index < train_count + dev_count
                else "holdout"
            )
            for record in grouped[group]:
                assignments[record.sample_id] = split
    return assignments
