"""Deterministic generator-disjoint and codec LOGO planning with leakage audits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


def _stable_digest(seed: int, *parts: object) -> bytes:
    joined = ":".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(joined.encode("utf-8")).digest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_manifest_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest line {number} is not an object")
            rows.append(value)
    required = {
        "sample_id", "split_group", "generator_model", "codec_chain",
        "file_fake", "voice_present", "music_present", "voice_fake", "music_fake",
    }
    if not rows:
        raise ValueError("manifest is empty")
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"manifest row {row.get('sample_id')} misses {sorted(missing)}")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("sample_id values must be unique")
    return rows


def atomic_generators(row: Mapping[str, object]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return normalized voice and music generator IDs for one manifest row."""

    raw = str(row.get("generator_model") or "")
    voice: list[str] = []
    music: list[str] = []
    for token in (piece.strip() for piece in raw.split("+")):
        if not token:
            continue
        normalized = token.casefold()
        if "musicgen" in normalized or "audioldm" in normalized:
            music.append(normalized)
        else:
            voice.append(normalized)
    return tuple(sorted(set(voice))), tuple(sorted(set(music)))


def _label_signature(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["file_fake"], row["voice_present"], row["music_present"],
        row["voice_fake"], row["music_fake"], row["codec_chain"],
    )


def _balanced_ids(
    rows: Iterable[Mapping[str, object]], *, limit: int, seed: int, namespace: str
) -> list[str]:
    """Round-robin deterministic strata without using labels outside the manifest."""

    by_stratum: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_stratum[_label_signature(row)].append(row)
    for signature, values in by_stratum.items():
        values.sort(key=lambda row: _stable_digest(seed, namespace, signature, row["sample_id"]))
    signatures = sorted(by_stratum, key=lambda value: _stable_digest(seed, namespace, value))
    selected: list[str] = []
    index = 0
    while len(selected) < limit and signatures:
        signature = signatures[index % len(signatures)]
        values = by_stratum[signature]
        if values:
            selected.append(str(values.pop(0)["sample_id"]))
        if not values:
            signatures.remove(signature)
            if not signatures:
                break
            index %= len(signatures)
        else:
            index += 1
    return selected


def _split_atoms(
    counts: Counter[str], bins: int, *, seed: int, namespace: str
) -> list[list[str]]:
    if len(counts) < bins:
        raise ValueError(f"{namespace} requires at least {bins} generator IDs")
    result: list[list[str]] = [[] for _ in range(bins)]
    totals = [0] * bins
    atoms = sorted(
        counts,
        key=lambda atom: (-counts[atom], _stable_digest(seed, namespace, atom)),
    )
    for atom in atoms:
        target = min(range(bins), key=lambda index: (totals[index], index))
        result[target].append(atom)
        totals[target] += counts[atom]
    return result


def _select_generator_panel(
    rows: list[dict[str, object]],
    *,
    atoms: set[str],
    component: str,
    excluded_groups: set[str],
    panel_size: int,
    seed: int,
    fold_id: str,
) -> list[str]:
    fake_key = f"{component}_fake"
    present_key = f"{component}_present"

    def contains_target(row: Mapping[str, object]) -> bool:
        voice, music = atomic_generators(row)
        values = voice if component == "voice" else music
        return bool(atoms.intersection(values))

    eligible = [
        row for row in rows
        if str(row["split_group"]) not in excluded_groups
    ]
    positives = [
        row for row in eligible
        if contains_target(row) and row[present_key] == 1 and row[fake_key] == 1
    ]
    negatives = [
        row for row in eligible
        if row[present_key] == 1 and row[fake_key] == 0
    ]
    half = panel_size // 2
    positive_ids = _balanced_ids(
        positives, limit=half, seed=seed, namespace=f"{fold_id}:positive"
    )
    negative_ids = _balanced_ids(
        negatives, limit=panel_size - len(positive_ids), seed=seed,
        namespace=f"{fold_id}:negative",
    )
    if len(positive_ids) < max(2, panel_size // 4) or len(negative_ids) < max(2, panel_size // 4):
        raise ValueError(f"{fold_id} cannot form a balanced evaluation panel")
    return positive_ids + negative_ids


def _assign_fold_splits(
    rows: list[dict[str, object]],
    *,
    excluded_groups: set[str],
    evaluation_groups: set[str],
    seed: int,
    fold_id: str,
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    remaining_groups = sorted({
        str(row["split_group"]) for row in rows
        if str(row["split_group"]) not in excluded_groups
    })
    if len(remaining_groups) < 2:
        raise ValueError(f"{fold_id} leaves too few groups for train/dev")
    dev_groups = {
        group for group in remaining_groups
        if int.from_bytes(_stable_digest(seed, fold_id, group)[:8], "big") / 2**64 < 0.10
    }
    if not dev_groups:
        dev_groups.add(min(remaining_groups, key=lambda group: _stable_digest(seed, fold_id, group)))
    if dev_groups == set(remaining_groups):
        dev_groups.remove(max(dev_groups, key=lambda group: _stable_digest(seed, fold_id, group)))
    for row in rows:
        sample_id = str(row["sample_id"])
        group = str(row["split_group"])
        if group in evaluation_groups:
            assignments[sample_id] = "holdout"
        elif group in excluded_groups:
            assignments[sample_id] = "generator_holdout"
        elif group in dev_groups:
            assignments[sample_id] = "dev"
        else:
            assignments[sample_id] = "train"
    return assignments


def _fold_payload(
    rows: list[dict[str, object]],
    *,
    fold_id: str,
    axis: str,
    task: str,
    targets: list[str],
    panel_ids: list[str],
    excluded_groups: set[str],
    seed: int,
) -> dict[str, object]:
    by_id = {str(row["sample_id"]): row for row in rows}
    evaluation_groups = {str(by_id[sample_id]["split_group"]) for sample_id in panel_ids}
    all_excluded = set(excluded_groups) | evaluation_groups
    assignments = _assign_fold_splits(
        rows,
        excluded_groups=all_excluded,
        evaluation_groups=evaluation_groups,
        seed=seed,
        fold_id=fold_id,
    )
    counts = Counter(assignments.values())
    return {
        "fold_id": fold_id,
        "axis": axis,
        "task": task,
        "targets": targets,
        "panel_sample_ids": panel_ids,
        "excluded_split_groups": sorted(all_excluded),
        "split_counts": dict(sorted(counts.items())),
        "assignments": assignments,
    }


def build_logo_plan(
    rows: list[dict[str, object]],
    *,
    seed: int = 2608,
    calibration_size: int = 512,
    panel_size: int = 256,
    voice_folds: int = 6,
    music_folds: int = 2,
) -> dict[str, object]:
    if calibration_size < 2 or panel_size < 4:
        raise ValueError("calibration_size and panel_size are too small")
    calibration_ids = _balanced_ids(
        rows, limit=calibration_size, seed=seed, namespace="rank-calibration"
    )
    by_id = {str(row["sample_id"]): row for row in rows}
    calibration_groups = {str(by_id[sample_id]["split_group"]) for sample_id in calibration_ids}

    voice_counts: Counter[str] = Counter()
    music_counts: Counter[str] = Counter()
    for row in rows:
        voice, music = atomic_generators(row)
        if row["voice_fake"] == 1:
            voice_counts.update(voice)
        if row["music_fake"] == 1:
            music_counts.update(music)
    voice_bins = _split_atoms(voice_counts, voice_folds, seed=seed, namespace="voice")
    music_bins = _split_atoms(music_counts, music_folds, seed=seed, namespace="music")

    folds: list[dict[str, object]] = []
    for component, bins in (("voice", voice_bins), ("music", music_bins)):
        for index, atoms in enumerate(bins, start=1):
            fold_id = f"gen-{component}-{index:02d}"
            target = set(atoms)
            target_groups = {
                str(row["split_group"])
                for row in rows
                if target.intersection(atomic_generators(row)[0 if component == "voice" else 1])
            }
            panel_ids = _select_generator_panel(
                rows,
                atoms=target,
                component=component,
                excluded_groups=calibration_groups,
                panel_size=panel_size,
                seed=seed,
                fold_id=fold_id,
            )
            folds.append(_fold_payload(
                rows,
                fold_id=fold_id,
                axis="generator",
                task=component,
                targets=sorted(target),
                panel_ids=panel_ids,
                excluded_groups=calibration_groups | target_groups,
                seed=seed,
            ))

    codecs = sorted({str(row["codec_chain"]) for row in rows})
    for index, codec in enumerate(codecs, start=1):
        fold_id = f"codec-{index:02d}"
        target_rows = [
            row for row in rows
            if row["codec_chain"] == codec and str(row["split_group"]) not in calibration_groups
        ]
        panel_ids = _balanced_ids(
            target_rows, limit=panel_size, seed=seed, namespace=fold_id
        )
        if len(panel_ids) < panel_size:
            raise ValueError(f"{fold_id} has only {len(panel_ids)} eligible panel rows")
        target_groups = {str(row["split_group"]) for row in rows if row["codec_chain"] == codec}
        folds.append(_fold_payload(
            rows,
            fold_id=fold_id,
            axis="codec",
            task="all",
            targets=[codec],
            panel_ids=panel_ids,
            excluded_groups=calibration_groups | target_groups,
            seed=seed,
        ))

    payload: dict[str, object] = {
        "format_version": 1,
        "seed": seed,
        "calibration_sample_ids": calibration_ids,
        "calibration_split_groups": sorted(calibration_groups),
        "settings": {
            "calibration_size": calibration_size,
            "panel_size": panel_size,
            "voice_folds": voice_folds,
            "music_folds": music_folds,
            "codec_folds": len(codecs),
        },
        "folds": folds,
    }
    audit_logo_plan(rows, payload)
    payload["plan_sha256"] = _canonical_sha256(payload)
    return payload


def audit_logo_plan(rows: list[dict[str, object]], plan: Mapping[str, object]) -> None:
    all_ids = {str(row["sample_id"]) for row in rows}
    calibration = set(str(value) for value in plan["calibration_sample_ids"])  # type: ignore[index]
    if not calibration or not calibration <= all_ids:
        raise ValueError("calibration IDs are invalid")
    folds = plan["folds"]  # type: ignore[index]
    if not isinstance(folds, list) or not folds:
        raise ValueError("LOGO plan has no folds")
    for fold in folds:
        assignments = fold["assignments"]
        if set(assignments) != all_ids:
            raise ValueError(f"{fold['fold_id']} assignments do not match manifest")
        group_splits: dict[str, str] = {}
        train_rows: list[dict[str, object]] = []
        for row in rows:
            split = str(assignments[str(row["sample_id"])])
            group = str(row["split_group"])
            prior = group_splits.setdefault(group, split)
            if prior != split:
                raise ValueError(f"{fold['fold_id']} leaks split_group {group}")
            if split == "train":
                train_rows.append(row)
        if calibration.intersection({str(row["sample_id"]) for row in train_rows}):
            raise ValueError(f"{fold['fold_id']} trains on calibration records")
        targets = set(str(value) for value in fold["targets"])
        if fold["axis"] == "codec":
            if any(str(row["codec_chain"]) in targets for row in train_rows):
                raise ValueError(f"{fold['fold_id']} leaks target codec")
        else:
            target_index = 0 if fold["task"] == "voice" else 1
            if any(targets.intersection(atomic_generators(row)[target_index]) for row in train_rows):
                raise ValueError(f"{fold['fold_id']} leaks target generator")
        panel = set(str(value) for value in fold["panel_sample_ids"])
        if not panel or not panel <= all_ids:
            raise ValueError(f"{fold['fold_id']} panel IDs are invalid")
        if any(assignments[sample_id] != "holdout" for sample_id in panel):
            raise ValueError(f"{fold['fold_id']} panel is not held out")


def write_logo_plan(path: str | Path, plan: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_fold_splits(directory: str | Path, plan: Mapping[str, object]) -> None:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    for fold in plan["folds"]:  # type: ignore[index]
        path = output / f"{fold['fold_id']}.json"
        path.write_text(
            json.dumps(fold["assignments"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
