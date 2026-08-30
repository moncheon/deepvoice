"""Procedural bootstrap and license-ledger-driven training corpus generation."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

from .audio import decode_audio
from .channels import CHANNEL_PROFILES, apply_channel_profile, write_pcm16_wav
from .manifest import ManifestRecord, assign_stratified_group_splits, sha256_file, write_manifest
from .mixing import Component, mix_components
from .source_manifest import SourceRecord, audit_source_split_leakage, read_source_manifest

SAMPLE_RATE = 16_000
COMBINATIONS: tuple[tuple[int | None, int | None], ...] = (
    (0, None),
    (1, None),
    (None, 0),
    (None, 1),
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _voice_proxy(duration: float, fake: int, rng: np.random.Generator) -> np.ndarray:
    length = int(round(duration * SAMPLE_RATE))
    time = np.arange(length, dtype=np.float64) / SAMPLE_RATE
    base = rng.uniform(105.0, 230.0)
    if fake:
        frame = max(1, int(0.08 * SAMPLE_RATE))
        steps = rng.choice([-6.0, 0.0, 6.0], size=length // frame + 1)
        f0 = base + np.repeat(steps, frame)[:length]
    else:
        jitter = _smooth(rng.normal(0.0, 1.0, length), int(0.025 * SAMPLE_RATE))
        f0 = base + 8.0 * np.sin(2 * np.pi * 0.7 * time) + 2.5 * jitter
    phase = np.cumsum(2 * np.pi * f0 / SAMPLE_RATE)
    signal = np.zeros(length, dtype=np.float64)
    formants = rng.choice([520.0, 730.0, 1090.0]) + np.asarray([0.0, 950.0, 2100.0])
    for harmonic in range(1, 25):
        frequency = harmonic * base
        formant_gain = float(np.exp(-np.min((frequency - formants) ** 2) / (2 * 420.0**2)))
        phase_offset = 0.0 if fake else rng.uniform(-np.pi, np.pi)
        signal += (0.18 + formant_gain) / harmonic * np.sin(harmonic * phase + phase_offset)
    syllable_rate = rng.uniform(2.2, 4.2)
    envelope = np.maximum(0.0, np.sin(2 * np.pi * syllable_rate * time + rng.uniform(0, 2*np.pi))) ** 0.6
    envelope *= 0.35 + 0.65 * (np.sin(2 * np.pi * 0.31 * time) ** 2)
    breath = _smooth(rng.normal(0.0, 1.0, length), 7) * (0.018 if fake else 0.035)
    signal = signal * envelope + breath * envelope
    signal /= np.max(np.abs(signal)) + 1e-8
    if fake:
        signal = np.round(signal * 96.0) / 96.0
    return (0.55 * signal).astype(np.float32)[None]


def _music_proxy(duration: float, fake: int, rng: np.random.Generator) -> np.ndarray:
    length = int(round(duration * SAMPLE_RATE))
    output = np.zeros((2, length), dtype=np.float64)
    scale = np.asarray([0, 2, 4, 7, 9])
    beat = 0.25 if fake else rng.uniform(0.22, 0.38)
    cursor = 0
    note_index = 0
    while cursor < length:
        note_duration = beat if fake else beat * rng.uniform(0.75, 1.35)
        note_length = min(length - cursor, max(64, int(note_duration * SAMPLE_RATE)))
        semitone = int(scale[note_index % len(scale)] + 12 * rng.integers(0, 2))
        frequency = 110.0 * 2 ** (semitone / 12.0)
        local_time = np.arange(note_length, dtype=np.float64) / SAMPLE_RATE
        note = np.zeros(note_length, dtype=np.float64)
        harmonic_count = 8 if fake else 12
        for harmonic in range(1, harmonic_count + 1):
            detune = 0.0 if fake else rng.normal(0.0, 0.0015)
            phase = 0.0 if fake else rng.uniform(0, 2*np.pi)
            note += np.sin(2 * np.pi * frequency * harmonic * (1 + detune) * local_time + phase) / harmonic
        decay = np.exp(-local_time * (6.0 if fake else rng.uniform(3.0, 7.0)))
        note *= decay
        pan = 0.0 if fake else rng.uniform(-0.8, 0.8)
        angle = (pan + 1.0) * np.pi / 4.0
        output[0, cursor:cursor+note_length] += note * np.cos(angle)
        output[1, cursor:cursor+note_length] += note * np.sin(angle)
        cursor += note_length
        note_index += 1
    if not fake:
        output += rng.normal(0.0, 0.004, output.shape)
    output /= np.max(np.abs(output)) + 1e-8
    return (0.48 * output).astype(np.float32)


def _fresh_output(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite corpus directory: {output}")
    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True)
    return output, audio_dir


def _generator_fields(voice_fake: int | None, music_fake: int | None) -> tuple[str | None, str | None, str | None]:
    families, models = [], []
    if voice_fake:
        families.append("run0-procedural-voice")
        models.append("formant-harmonic-v1")
    if music_fake:
        families.append("run0-procedural-music")
        models.append("grid-oscillator-v1")
    if not families:
        return None, None, None
    return "+".join(families), "+".join(models), "run0-0.1.0"


def _write_splits(output: Path, records: list[ManifestRecord], seed: int) -> dict[str, int]:
    assignments = assign_stratified_group_splits(records, seed=seed)
    (output / "splits.json").write_text(
        json.dumps(assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for split in assignments.values():
        counts[split] = counts.get(split, 0) + 1
    return counts


def generate_bootstrap_corpus(
    output_dir: str | Path,
    *,
    samples_per_combination: int = 4,
    seed: int = 2608,
    min_duration: float = 4.0,
    max_duration: float = 8.0,
) -> dict[str, object]:
    """Create a self-owned DSP fixture corpus for end-to-end smoke training.

    The real-like class is a procedural proxy, not a recording of genuine human
    speech or human music. It must never be reported as detector performance.
    """

    if samples_per_combination <= 0 or min_duration <= 0 or max_duration < min_duration:
        raise ValueError("invalid bootstrap count or duration range")
    output, audio_dir = _fresh_output(output_dir)
    rng = np.random.default_rng(seed)
    records: list[ManifestRecord] = []
    profile_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for combination_index, (voice_fake, music_fake) in enumerate(COMBINATIONS):
        for variant in range(samples_per_combination):
            duration = float(rng.uniform(min_duration, max_duration))
            voice = None
            music = None
            recipe_seed = int(rng.integers(0, 2**31 - 1))
            if voice_fake is not None:
                voice = Component(_voice_proxy(duration, voice_fake, rng), "voice", voice_fake, f"voice-{recipe_seed}")
            if music_fake is not None:
                music = Component(_music_proxy(duration, music_fake, rng), "music", music_fake, f"music-{recipe_seed}")
            mode = "none" if voice is None or music is None else ("simultaneous" if variant % 2 == 0 else "sequential")
            mixture = mix_components(
                voice, music, mode=mode,
                voice_gain_db=float(rng.uniform(-4.0, 2.0)),
                music_gain_db=float(rng.uniform(-5.0, 1.0)),
                voice_pan=float(rng.uniform(-0.35, 0.35)),
                music_pan=float(rng.uniform(-0.7, 0.7)),
                gap_samples=int(rng.uniform(0.0, 0.35) * SAMPLE_RATE),
            )
            profile = CHANNEL_PROFILES[(combination_index * samples_per_combination + variant) % len(CHANNEL_PROFILES)]
            transformed, codec_chain = apply_channel_profile(
                mixture.samples, profile, sample_rate=SAMPLE_RATE, seed=recipe_seed
            )
            sample_id = f"BOOT_C{combination_index:02d}_V{variant:03d}"
            audio_path = audio_dir / f"{sample_id}.wav"
            write_pcm16_wav(audio_path, transformed, SAMPLE_RATE)
            family, model, revision = _generator_fields(voice_fake, music_fake)
            record = ManifestRecord(
                sample_id=sample_id,
                path=f"audio/{audio_path.name}",
                sha256=sha256_file(audio_path),
                source_dataset="run0-procedural-bootstrap-fixture",
                source_item_id=f"recipe-{recipe_seed}",
                license_id="project-generated-fixture-only",
                consent_basis="no-human-source-procedural-dsp",
                speaker_or_artist_group=f"procedural-identity-{combination_index}-{variant}",
                content_group=f"procedural-content-{combination_index}-{variant}",
                generator_family=family,
                generator_model=model,
                generator_revision=revision,
                generation_seed=recipe_seed,
                voice_present=mixture.voice_present,
                music_present=mixture.music_present,
                voice_fake=mixture.voice_fake,
                music_fake=mixture.music_fake,
                file_fake=mixture.file_fake,
                mix_mode=mixture.mix_mode,
                codec_chain=codec_chain,
                channel_profile=profile,
                sample_rate=SAMPLE_RATE,
                channels=int(transformed.shape[0]),
                duration_sec=float(transformed.shape[1] / SAMPLE_RATE),
                split_group=f"bootstrap-recipe-{combination_index}-{variant}",
            )
            record.validate()
            records.append(record)
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
            key = f"voice={voice_fake},music={music_fake}"
            label_counts[key] = label_counts.get(key, 0) + 1
    manifest_path = output / "manifest.jsonl"
    write_manifest(manifest_path, records)
    split_counts = _write_splits(output, records, seed)
    total_bytes = sum((output / record.path).stat().st_size for record in records)
    summary: dict[str, object] = {
        "format_version": 1,
        "fixture_only": True,
        "warning": "Procedural smoke corpus; not evidence of real-world detector performance.",
        "seed": seed,
        "files": len(records),
        "total_bytes": total_bytes,
        "total_duration_sec": sum(record.duration_sec for record in records),
        "manifest_sha256": sha256_file(manifest_path),
        "label_counts": label_counts,
        "channel_profile_counts": profile_counts,
        "split_counts": split_counts,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _crop(samples: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if samples.shape[1] < length:
        return np.pad(samples, ((0, 0), (0, length - samples.shape[1])))
    start = int(rng.integers(0, samples.shape[1] - length + 1))
    return samples[:, start:start+length]


def _join(values: Iterable[str | None]) -> str | None:
    present = sorted({str(value) for value in values if value})
    return "+".join(present) if present else None


def build_corpus_from_sources(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    variants_per_combination: int = 4,
    seed: int = 2608,
    duration_sec: float = 6.0,
) -> dict[str, object]:
    """Mix reviewed sources only within their already-assigned source split.

    Legacy v1 ledgers retain their original ``mix_split_group`` behavior. A v2
    production ledger pools heterogeneous datasets by ``split`` so a real
    Korean voice can be mixed with reviewed music without pretending every
    dataset contains all four component/label buckets.
    """

    if variants_per_combination <= 0 or duration_sec < 4.0:
        raise ValueError("variants must be positive and duration_sec must be at least 4")
    source_path = Path(source_manifest)
    sources = read_source_manifest(source_path, audit_files=True)
    production_v2 = any(source.format_version == 2 for source in sources)
    if production_v2 and any(source.format_version != 2 for source in sources):
        raise ValueError("production source manifests cannot mix v1 and v2 records")
    if production_v2:
        audit_source_split_leakage(sources)
    grouped_buckets: dict[str, dict[tuple[str, int], list[SourceRecord]]] = {}
    for source in sources:
        pool = source.split if production_v2 else source.mix_split_group
        if not pool:
            raise ValueError(f"source has no split pool: {source.source_id}")
        grouped_buckets.setdefault(pool, {}).setdefault(
            (source.component, source.fake), []
        ).append(source)
    required_buckets = (("voice", 0), ("voice", 1), ("music", 0), ("music", 1))
    if production_v2:
        missing = {
            split: [f"{component}:{fake}" for component, fake in required_buckets if not grouped_buckets.get(split, {}).get((component, fake))]
            for split in ("train", "dev", "holdout")
        }
        missing = {split: buckets for split, buckets in missing.items() if buckets}
        if missing:
            raise ValueError(f"production source buckets are incomplete by split: {missing}")
    eligible_groups = sorted(
        group for group, buckets in grouped_buckets.items()
        if all(buckets.get(key) for key in required_buckets)
    )
    if not eligible_groups:
        raise ValueError(
            "no source split covers real/fake voice/music; assign source splits before mixing"
        )
    output, audio_dir = _fresh_output(output_dir)
    rng = np.random.default_rng(seed)
    target_length = int(duration_sec * SAMPLE_RATE)
    records: list[ManifestRecord] = []
    crop_caches: dict[str, OrderedDict[str, tuple[np.ndarray, ...]]] = {
        "voice": OrderedDict(), "music": OrderedDict()
    }
    cache_bytes = {"voice": 0, "music": 0}
    cache_limits = {"voice": 512 * 1024**2, "music": 1536 * 1024**2}

    def sample_source(source: SourceRecord) -> np.ndarray:
        """Cache bounded fixed-length crops, never entire long recordings."""

        cache = crop_caches[source.component]
        variants = cache.pop(source.source_id, None)
        if variants is None:
            path = Path(source.path)
            if not path.is_absolute():
                path = source_path.parent / path
            decoded = decode_audio(path, SAMPLE_RATE).samples
            variant_count = 4 if source.component == "music" and decoded.shape[1] > target_length else 1
            variants = tuple(
                np.ascontiguousarray(_crop(decoded, target_length, rng), dtype=np.float32)
                for _ in range(variant_count)
            )
            size = sum(item.nbytes for item in variants)
            limit = cache_limits[source.component]
            while cache and cache_bytes[source.component] + size > limit:
                _, removed = cache.popitem(last=False)
                cache_bytes[source.component] -= sum(item.nbytes for item in removed)
            if size <= limit:
                cache[source.source_id] = variants
                cache_bytes[source.component] += size
        else:
            cache[source.source_id] = variants
        return variants[int(rng.integers(0, len(variants)))]

    split_assignments: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    source_use_counts: dict[str, int] = {}
    for combination_index, (voice_fake, music_fake) in enumerate(COMBINATIONS):
        for variant in range(variants_per_combination):
            if production_v2:
                # Weighted 70/15/15 scheduling remains exact for large corpora.
                selector = variant % 20
                preferred = "train" if selector < 14 else "dev" if selector < 17 else "holdout"
                pool = preferred
            else:
                pool = eligible_groups[(combination_index + variant) % len(eligible_groups)]
            buckets = grouped_buckets[pool]
            chosen: list[SourceRecord] = []
            voice = music = None
            if voice_fake is not None:
                voice_bucket = buckets[("voice", voice_fake)]
                voice_source = voice_bucket[int(rng.integers(0, len(voice_bucket)))]
                chosen.append(voice_source)
                voice = Component(sample_source(voice_source), "voice", voice_fake, voice_source.source_id)
            if music_fake is not None:
                music_bucket = buckets[("music", music_fake)]
                music_source = music_bucket[int(rng.integers(0, len(music_bucket)))]
                chosen.append(music_source)
                music = Component(sample_source(music_source), "music", music_fake, music_source.source_id)
            mode = "none" if len(chosen) == 1 else ("simultaneous" if variant % 2 == 0 else "sequential")
            mixture = mix_components(
                voice, music, mode=mode,
                voice_gain_db=float(rng.uniform(-5, 2)), music_gain_db=float(rng.uniform(-6, 1)),
                voice_pan=float(rng.uniform(-0.4, 0.4)), music_pan=float(rng.uniform(-0.8, 0.8)),
                gap_samples=int(rng.uniform(0, 0.4) * SAMPLE_RATE),
            )
            profile = CHANNEL_PROFILES[(combination_index + variant) % len(CHANNEL_PROFILES)]
            transform_seed = int(rng.integers(0, 2**31 - 1))
            transformed, codec_chain = apply_channel_profile(mixture.samples, profile, seed=transform_seed)
            sample_id = f"MIX_C{combination_index:02d}_V{variant:05d}"
            combination_dir = audio_dir / f"C{combination_index:02d}"
            combination_dir.mkdir(exist_ok=True)
            audio_path = combination_dir / f"{sample_id}.wav"
            write_pcm16_wav(audio_path, transformed, SAMPLE_RATE)
            record = ManifestRecord(
                sample_id=sample_id,
                path=f"audio/{combination_dir.name}/{audio_path.name}",
                sha256=sha256_file(audio_path),
                source_dataset="+".join(sorted({item.source_dataset for item in chosen})),
                source_item_id="+".join(sorted(item.source_item_id for item in chosen)),
                license_id="+".join(sorted({item.license_id for item in chosen})),
                consent_basis="+".join(sorted({item.consent_basis for item in chosen})),
                speaker_or_artist_group="+".join(sorted(item.identity_group for item in chosen)),
                content_group="+".join(sorted(item.content_group for item in chosen)),
                generator_family=_join(item.generator_family for item in chosen),
                generator_model=_join(item.generator_model for item in chosen),
                generator_revision=_join(item.generator_revision for item in chosen),
                generation_seed=transform_seed,
                voice_present=mixture.voice_present,
                music_present=mixture.music_present,
                voice_fake=mixture.voice_fake,
                music_fake=mixture.music_fake,
                file_fake=mixture.file_fake,
                mix_mode=mixture.mix_mode,
                codec_chain=codec_chain,
                channel_profile=profile,
                sample_rate=SAMPLE_RATE,
                channels=int(transformed.shape[0]),
                duration_sec=float(transformed.shape[1] / SAMPLE_RATE),
                split_group=(
                    "+".join(sorted({item.mix_split_group or "" for item in chosen}))
                    if not production_v2
                    else "parents:" + "+".join(sorted(item.source_id for item in chosen))
                ),
            )
            record.validate()
            records.append(record)
            split_assignments[sample_id] = pool if production_v2 else ""
            label_key = f"v{voice_fake if voice_fake is not None else 'x'}_m{music_fake if music_fake is not None else 'x'}"
            label_counts[label_key] = label_counts.get(label_key, 0) + 1
            for item in chosen:
                source_use_counts[item.source_id] = source_use_counts.get(item.source_id, 0) + 1
            if len(records) == 1 or len(records) % 1000 == 0:
                print(json.dumps({"corpus_files": len(records), "target_files": len(COMBINATIONS) * variants_per_combination}), flush=True)
    manifest_path = output / "manifest.jsonl"
    write_manifest(manifest_path, records)
    if production_v2:
        (output / "splits.json").write_text(
            json.dumps(split_assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        split_counts: dict[str, int] = {}
        for split in split_assignments.values():
            split_counts[split] = split_counts.get(split, 0) + 1
    else:
        split_counts = _write_splits(output, records, seed)
    summary: dict[str, object] = {
        "format_version": 2 if production_v2 else 1,
        "fixture_only": False,
        "files": len(records),
        "source_files": len(sources),
        "seed": seed,
        "manifest_sha256": sha256_file(manifest_path),
        "total_bytes": sum((output / record.path).stat().st_size for record in records),
        "total_duration_sec": sum(record.duration_sec for record in records),
        "split_counts": split_counts,
        "label_counts": label_counts,
        "unique_sources_used": len(source_use_counts),
        "source_use_min": min(source_use_counts.values()),
        "source_use_max": max(source_use_counts.values()),
        "source_crop_cache_limits_bytes": cache_limits,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
