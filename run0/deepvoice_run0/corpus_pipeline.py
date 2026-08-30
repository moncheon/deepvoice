"""Download-review-prepare-audit pipeline for real Run 0 training audio."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Iterable

from .manifest import read_manifest, sha256_file
from .source_manifest import (
    SourceRecord,
    assign_source_splits,
    audit_source_split_leakage,
    read_source_manifest,
    write_source_manifest,
)

RISKY_SUFFIXES = {".bin", ".ckpt", ".dll", ".exe", ".pkl", ".pickle", ".pt", ".pth", ".so", ".th"}
AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
ADAPTABLE_CC = (
    "creativecommons.org/licenses/by/",
    "creativecommons.org/licenses/by-sa/",
    "creativecommons.org/licenses/by-nc/",
    "creativecommons.org/licenses/by-nc-sa/",
    "creativecommons.org/publicdomain/",
    "license: cc0",
    "license: cc-by",
    "license: cc by",
)
NON_ADAPTABLE = (
    "creativecommons.org/licenses/by-nd/",
    "creativecommons.org/licenses/by-nc-nd/",
    "cc-by-nd",
    "cc-by-nc-nd",
    "cc by-nd",
    "cc by-nc-nd",
)


def load_catalog(path: str | Path) -> dict[str, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not isinstance(payload.get("sources"), dict):
        raise ValueError("unsupported corpus source catalog")
    return payload["sources"]


def _run(command: list[str], *, cwd: Path | None = None, log_path: Path | None = None) -> None:
    if log_path is None:
        result = subprocess.run(command, cwd=cwd, check=False, text=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=cwd, check=False, text=True,
                stdout=log, stderr=subprocess.STDOUT,
            )
    if result.returncode:
        detail = ""
        if log_path and log_path.is_file():
            detail = "\n" + "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}{detail}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_source(name: str, spec: dict[str, object], staging_root: str | Path) -> Path:
    """Download without importing or executing any downloaded Python/model file."""

    destination = Path(staging_root) / name
    destination.mkdir(parents=True, exist_ok=True)
    kind = str(spec["kind"])
    plan: dict[str, object] = {"source": name, "kind": kind, "spec": spec, "status": "planned"}
    _write_json(destination / "DOWNLOAD_PLAN.json", plan)
    log_path = destination / "fetch.log"
    if kind in {"hf_dataset", "hf_model"}:
        command = [
            "hf", "download", str(spec["repo_id"]),
            "--type", "dataset" if kind == "hf_dataset" else "model",
            "--revision", str(spec["revision"]),
            "--local-dir", str(destination),
        ]
        for pattern in spec.get("include", []):
            command.extend(("--include", str(pattern)))
        _run(command + ["--dry-run", "--format", "json"], log_path=log_path)
        _run(command + ["--format", "quiet"], log_path=log_path)
    elif kind == "hf_rows":
        _fetch_hf_rows(name, spec, destination, log_path)
    elif kind == "url_archive":
        target = destination / str(spec["filename"])
        if not target.is_file():
            _run(["curl.exe", "--fail", "--location", "--continue-at", "-", "--output", str(target), str(spec["url"])], log_path=log_path)
    elif kind == "url_bundle":
        for item in spec.get("files", []):
            if not isinstance(item, dict):
                raise ValueError(f"invalid URL bundle item for {name}")
            target = destination / str(item["filename"])
            if not target.is_file():
                _run(["curl.exe", "--fail", "--location", "--continue-at", "-", "--output", str(target), str(item["url"])], log_path=log_path)
    else:
        raise ValueError(f"unsupported source kind: {kind}")
    plan["status"] = "downloaded_unreviewed"
    _write_json(destination / "DOWNLOAD_PLAN.json", plan)
    return destination


def _api_json(url: str, *, attempts: int = 6) -> dict[str, object]:
    """Read a Dataset Server page with bounded retries for transient timeouts."""

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            headers = {"User-Agent": "deepvoice-run0-corpus/1.0"}
            if token := os.environ.get("HF_TOKEN"):
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            retry_after = 0
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                try:
                    retry_after = int(error.headers.get("Retry-After", "0"))
                except ValueError:
                    retry_after = 0
            time.sleep(max(min(retry_after, 30), min(2 ** attempt, 16)))
    raise RuntimeError(f"Dataset Server request failed after {attempts} attempts: {url}") from last_error


def _download_asset(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 44:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "deepvoice-run0-corpus/1.0"})
            with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            os.replace(temporary, target)
            return
        except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"audio asset download failed: {url}") from last_error


def _instrumental_row(row: dict[str, object]) -> bool:
    attribution = str(row.get("attribution", ""))
    description = str(row.get("description", "")).lower()
    license_line = next((line for line in attribution.splitlines() if line.lower().startswith("license:")), "")
    adaptable = _fma_license_allowed(license_line)
    no_vocal = "vocal content: none" in description or "instrumental" in [str(item).lower() for item in row.get("genre_tags", [])]
    return adaptable and no_vocal


def _fetch_hf_rows(name: str, spec: dict[str, object], destination: Path, log_path: Path) -> None:
    """Fetch decoded audio assets through Dataset Server without executing dataset code."""

    repo_id = str(spec["repo_id"])
    _run([
        "hf", "download", repo_id, "--type", "dataset", "--revision", str(spec["revision"]),
        "--include", "README.md", "--local-dir", str(destination), "--format", "quiet",
    ], log_path=log_path)
    selected: list[dict[str, object]] = []
    for split, maximum in spec.get("splits", {}).items():
        offset = 0
        wanted = int(maximum)
        while len([item for item in selected if item["split"] == split]) < wanted:
            length = min(100, wanted - len([item for item in selected if item["split"] == split]))
            query = urllib.parse.urlencode({"dataset": repo_id, "config": "default", "split": split, "offset": offset, "length": length})
            payload = _api_json(f"https://datasets-server.huggingface.co/rows?{query}")
            time.sleep(0.25)
            rows = payload.get("rows", [])
            if not rows:
                break
            for entry in rows:
                row = entry["row"]
                offset = max(offset, int(entry["row_idx"]) + 1)
                if spec.get("filter") == "instrumental_with_attribution" and not _instrumental_row(row):
                    continue
                audio_value = row[str(spec["audio_field"])]
                asset = audio_value[0] if isinstance(audio_value, list) else audio_value
                if not isinstance(asset, dict) or "src" not in asset:
                    raise ValueError(f"Dataset Server returned no audio asset for {name}/{split}/{entry['row_idx']}")
                mime = str(asset.get("type", "audio/wav"))
                suffix = ".mp3" if "mpeg" in mime else ".wav"
                local = Path("audio") / split / f"{int(entry['row_idx']):07d}{suffix}"
                cleaned = dict(row)
                cleaned[str(spec["audio_field"])] = None
                selected.append({
                    "split": split, "row_idx": int(entry["row_idx"]), "local_path": local.as_posix(),
                    "asset_url": str(asset["src"]), "row": cleaned,
                })
                if len([item for item in selected if item["split"] == split]) >= wanted:
                    break
            if offset >= int(payload.get("num_rows_total", offset)):
                break
    if not selected:
        raise ValueError(f"Dataset Server returned no selected rows for {name}")
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(_download_asset, str(item["asset_url"]), destination / str(item["local_path"])): item
            for item in selected
        }
        for future in as_completed(futures):
            future.result()
    metadata = destination / "rows.jsonl"
    with metadata.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(selected, key=lambda value: (str(value["split"]), int(value["row_idx"]))):
            stored = dict(item)
            stored.pop("asset_url", None)
            handle.write(json.dumps(stored, ensure_ascii=False, sort_keys=True) + "\n")


def _sha1_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def review_source(name: str, spec: dict[str, object], staging_root: str | Path) -> dict[str, object]:
    """Hash and license-audit a staged download without executing it."""

    root = Path(staging_root) / name
    if not root.is_dir():
        raise FileNotFoundError(root)
    kind = str(spec["kind"])
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts)
    files = [path for path in files if path.name not in {"DOWNLOAD_PLAN.json", "REVIEW.json"}]
    if not files:
        raise ValueError(f"staged source is empty: {name}")
    risky = [path.relative_to(root).as_posix() for path in files if path.suffix.lower() in RISKY_SUFFIXES]
    if kind == "hf_model" and risky:
        raise ValueError(f"model staging contains pickle/executable weights: {risky[:10]}")
    if kind in {"hf_dataset", "hf_model", "hf_rows"} and not any(path.name.lower() == "readme.md" for path in files):
        raise ValueError(f"HF source has no reviewed README: {name}")
    expected_sha1: dict[str, str] = {}
    for item in spec.get("files", []):
        if isinstance(item, dict) and item.get("sha1"):
            expected_sha1[str(item["filename"])] = str(item["sha1"]).lower()
    for filename, expected in expected_sha1.items():
        actual = _sha1_file(root / filename)
        if actual != expected:
            raise ValueError(f"official SHA-1 mismatch for {name}/{filename}: {actual}")
    hashes = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    report: dict[str, object] = {
        "format_version": 1,
        "source": name,
        "kind": kind,
        "repo_id_or_url": spec.get("repo_id", spec.get("url", "bundle")),
        "revision": spec["revision"],
        "license_id": spec["license_id"],
        "license_url": spec["license_url"],
        "reviewed_for_noncommercial_research": True,
        "blocked_risky_files": risky,
        "file_count": len(hashes),
        "total_bytes": sum(int(item["bytes"]) for item in hashes),
        "files": hashes,
    }
    _write_json(root / "REVIEW.json", report)
    return report


def require_review(root: str | Path, spec: dict[str, object]) -> dict[str, object]:
    path = Path(root) / "REVIEW.json"
    if not path.is_file():
        raise ValueError(f"review gate missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("revision") != spec.get("revision") or report.get("license_id") != spec.get("license_id"):
        raise ValueError(f"review gate does not match current source spec: {root}")
    for item in report.get("files", []):
        candidate = Path(root) / str(item["path"])
        if not candidate.is_file() or sha256_file(candidate) != item["sha256"]:
            raise ValueError(f"post-review file changed: {candidate}")
    return report


def _safe_member(root: Path, name: str) -> Path:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe archive path: {name}")
    target = (root / Path(*normalized.parts)).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"archive member escapes target: {name}")
    return target


def extract_reviewed_archives(name: str, spec: dict[str, object], staging_root: str | Path) -> Path:
    root = Path(staging_root) / name
    require_review(root, spec)
    output = root / "extracted"
    marker = output / "EXTRACTED.json"
    if marker.is_file():
        return output
    if output.exists():
        raise FileExistsError(f"partial extraction must be inspected: {output}")
    output.mkdir(parents=True)
    archives = [path for path in root.iterdir() if path.is_file() and (path.suffix.lower() == ".zip" or path.name.endswith((".tar.gz", ".tgz")))]
    archives += sorted((root / "fake_songs").glob("*.zip")) if (root / "fake_songs").is_dir() else []
    for archive in archives:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as handle:
                for info in handle.infolist():
                    _safe_member(output, info.filename)
                handle.extractall(output)
        else:
            with tarfile.open(archive, "r:gz") as handle:
                for info in handle.getmembers():
                    _safe_member(output, info.name)
                    if info.issym() or info.islnk():
                        raise ValueError(f"archive links are not allowed: {info.name}")
                handle.extractall(output, filter="data")
    _write_json(marker, {"source": name, "archives": [path.name for path in archives]})
    return output


def _source_path(path: Path, manifest_dir: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), manifest_dir.resolve()).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _record(
    *,
    path: Path,
    manifest_dir: Path,
    source_id: str,
    component: str,
    fake: int,
    dataset: str,
    revision: str,
    source_url: str,
    item_id: str,
    license_id: str,
    license_url: str,
    identity: str,
    content: str,
    language: str,
    generator_family: str | None = None,
    generator_model: str | None = None,
    generator_revision: str | None = None,
    split: str | None = None,
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        path=_source_path(path, manifest_dir),
        sha256=sha256_file(path),
        component=component,
        fake=fake,
        source_dataset=dataset,
        source_item_id=item_id,
        license_id=license_id,
        consent_basis="public-corpus-license-reviewed-for-noncommercial-research",
        identity_group=identity,
        content_group=content,
        mix_split_group=None,
        generator_family=generator_family,
        generator_model=generator_model,
        generator_revision=generator_revision,
        format_version=2,
        source_revision=revision,
        source_url=source_url,
        license_url=license_url,
        language=language,
        split=split,
    )


def import_mlaad(root: Path, spec: dict[str, object], manifest_dir: Path) -> list[SourceRecord]:
    require_review(root, spec)
    records: list[SourceRecord] = []
    for fake, subdir in ((0, "original/en"), (1, "fake/en")):
        base = root / subdir
        for path in sorted(base.rglob("*.wav")):
            rel = path.relative_to(base).as_posix()
            family = path.relative_to(base).parts[0] if fake else None
            stem = path.stem
            identity = stem.rsplit("_f", 1)[0].rsplit("_", 1)[0]
            content = stem.rsplit("_f", 1)[0]
            real_group_split = {
                "dorothy_and_wizard_oz": "train",
                "emerald_city_of_oz": "dev",
                "hunters_space": "holdout",
                "jane_eyre": "train",
            }.get(identity) if not fake else None
            records.append(_record(
                path=path, manifest_dir=manifest_dir,
                source_id=f"mlaad-en-{'fake' if fake else 'real'}-{hashlib.sha256(rel.encode()).hexdigest()[:20]}",
                component="voice", fake=fake, dataset="mueller91/MLAAD-tiny",
                revision=str(spec["revision"]), source_url=str(spec["repo_id"]), item_id=rel,
                license_id="CC-BY-NC-4.0" if fake else "M-AILABS-original-license",
                license_url=str(spec["license_url"]), identity=f"mlaad:{identity}",
                content=f"mlaad-content:{content}", language="en",
                generator_family=f"MLAAD:{family}" if family else None,
                generator_model=family, generator_revision=str(spec["revision"]) if fake else None,
                split=real_group_split,
            ))
    return records


def import_zeroth(root: Path, spec: dict[str, object], manifest_dir: Path) -> list[SourceRecord]:
    extracted = extract_reviewed_archives("zeroth-korean", spec, root.parent)
    records: list[SourceRecord] = []
    for path in sorted(item for item in extracted.rglob("*") if item.suffix.lower() in {".wav", ".flac"}):
        rel = path.relative_to(extracted).as_posix()
        lower = rel.lower()
        split = "holdout" if "/test" in "/" + lower or lower.startswith("test") else None
        parts = path.stem.split("_")
        speaker = parts[0] if parts else path.parent.name
        records.append(_record(
            path=path, manifest_dir=manifest_dir,
            source_id=f"zeroth-{hashlib.sha256(rel.encode()).hexdigest()[:20]}",
            component="voice", fake=0, dataset="OpenSLR-SLR40-Zeroth-Korean",
            revision=str(spec["revision"]), source_url=str(spec["url"]), item_id=rel,
            license_id=str(spec["license_id"]), license_url=str(spec["license_url"]),
            identity=f"zeroth-speaker:{speaker}", content=f"zeroth-utterance:{path.stem}",
            language="ko", split=split,
        ))
    return records


def _fma_license_allowed(value: str) -> bool:
    normalized = value.lower().replace("_", "-")
    return bool(normalized) and not any(token in normalized for token in NON_ADAPTABLE) and any(token in normalized for token in ADAPTABLE_CC)


def import_fma(root: Path, spec: dict[str, object], manifest_dir: Path) -> list[SourceRecord]:
    extracted = extract_reviewed_archives("fma-small", spec, root.parent)
    tracks_path = next(iter(extracted.rglob("tracks.csv")), None)
    if tracks_path is None:
        raise FileNotFoundError("FMA tracks.csv was not extracted")
    allowed: dict[int, tuple[str, str, str]] = {}
    with tracks_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        top = next(reader)
        bottom = next(reader)
        columns = [f"{a}.{b}" for a, b in zip(top, bottom)]
        index = {name: i for i, name in enumerate(columns)}
        license_key = next((key for key in index if key.lower() in {"track.license", "track.license_url"}), None)
        artist_key = next((key for key in index if key.lower() in {"artist.id", "artist.name"}), None)
        title_key = next((key for key in index if key.lower() == "track.title"), None)
        subset_key = next((key for key in index if key.lower() == "set.subset"), None)
        if None in {license_key, artist_key, title_key, subset_key}:
            raise ValueError(f"unexpected FMA metadata columns: {columns[:20]}")
        for row in reader:
            if not row or row[0] == "track_id" or row[index[subset_key]] != "small":
                continue
            license_value = row[index[license_key]]
            if _fma_license_allowed(license_value):
                allowed[int(row[0])] = (license_value, row[index[artist_key]], row[index[title_key]])
    records: list[SourceRecord] = []
    for track_id, (license_value, artist, title) in sorted(allowed.items()):
        matches = list(extracted.rglob(f"{track_id:06d}.mp3"))
        if not matches:
            continue
        path = matches[0]
        records.append(_record(
            path=path, manifest_dir=manifest_dir, source_id=f"fma-{track_id:06d}",
            component="music", fake=0, dataset="FMA-small", revision=str(spec["revision"]),
            source_url="https://github.com/mdeff/fma", item_id=str(track_id),
            license_id=license_value or "per-track-creative-commons", license_url=license_value or str(spec["license_url"]),
            identity=f"fma-artist:{artist}", content=f"fma-track:{track_id}", language="und",
        ))
    return records


def import_sonics(root: Path, spec: dict[str, object], manifest_dir: Path) -> list[SourceRecord]:
    extracted = extract_reviewed_archives("sonics", spec, root.parent)
    records: list[SourceRecord] = []
    for path in sorted(item for item in extracted.rglob("*") if item.suffix.lower() in AUDIO_SUFFIXES):
        rel = path.relative_to(extracted).as_posix()
        lower = rel.lower()
        family = "Udio" if "udio" in lower else "Suno" if "suno" in lower else "SONICS-unknown"
        records.append(_record(
            path=path, manifest_dir=manifest_dir,
            source_id=f"sonics-{hashlib.sha256(rel.encode()).hexdigest()[:20]}",
            component="music", fake=1, dataset="awsaf49/sonics", revision=str(spec["revision"]),
            source_url=str(spec["repo_id"]), item_id=rel, license_id=str(spec["license_id"]),
            license_url=str(spec["license_url"]), identity=f"sonics-song:{path.stem}",
            content=f"sonics-song:{path.stem}", language="en", generator_family=f"SONICS:{family}",
            generator_model=family, generator_revision=str(spec["revision"]),
        ))
    return records


def import_generated_manifest(path: str | Path, manifest_dir: Path) -> list[SourceRecord]:
    review_path = Path(path).parent / "GENERATED_REVIEW.json"
    if not review_path.is_file():
        raise ValueError(f"generated-audio review gate missing: {review_path}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "reviewed" or review.get("manifest_sha256") != sha256_file(Path(path)):
        raise ValueError(f"generated-audio review does not match manifest: {path}")
    records = read_source_manifest(path, audit_files=True)
    normalized: list[SourceRecord] = []
    source_dir = Path(path).parent
    for record in records:
        audio = Path(record.path)
        if not audio.is_absolute():
            audio = source_dir / audio
        payload = record.__dict__ | {"path": _source_path(audio, manifest_dir), "sha256": sha256_file(audio)}
        normalized.append(SourceRecord(**payload))
    return normalized


def review_generated_manifest(path: str | Path) -> dict[str, object]:
    """Audit generated WAV hashes and the model/package review evidence."""

    manifest = Path(path)
    root = manifest.parent
    records = read_source_manifest(manifest, audit_files=True)
    if any(not record.fake for record in records):
        raise ValueError("generated manifest contains a source labelled real")
    if any("no-identity-cloning" not in record.consent_basis for record in records):
        raise ValueError("generated manifest lacks the non-impersonation consent boundary")
    model_review_path = root / "MODEL_REVIEW.json"
    wheel_review_path = root / "WHEEL_REVIEW.json"
    for evidence_path in (model_review_path, wheel_review_path):
        if not evidence_path.is_file():
            raise FileNotFoundError(evidence_path)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") != "reviewed":
            raise ValueError(f"review evidence is not approved: {evidence_path}")
    counts = Counter((record.component, record.generator_family, record.split) for record in records)
    report = {
        "format_version": 1,
        "status": "reviewed",
        "manifest_sha256": sha256_file(manifest),
        "records": len(records),
        "total_audio_bytes": sum((root / record.path).stat().st_size for record in records),
        "generator_split_counts": {
            f"{component}:{family}:{split}": count
            for (component, family, split), count in sorted(counts.items())
        },
        "model_review_sha256": sha256_file(model_review_path),
        "wheel_review_sha256": sha256_file(wheel_review_path),
    }
    _write_json(root / "GENERATED_REVIEW.json", report)
    return report


def _rows(root: Path) -> list[dict[str, object]]:
    rows_path = root / "rows.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(rows_path)
    return [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_zeroth_hf(root: Path, spec: dict[str, object], manifest_dir: Path) -> list[SourceRecord]:
    require_review(root, spec)
    records: list[SourceRecord] = []
    for item in _rows(root):
        split_name = str(item["split"])
        row = item["row"]
        text = str(row.get("text", ""))
        row_idx = int(item["row_idx"])
        path = root / str(item["local_path"])
        records.append(_record(
            path=path, manifest_dir=manifest_dir, source_id=f"zeroth-hf-{split_name}-{row_idx:07d}",
            component="voice", fake=0, dataset=str(spec["source_dataset"]), revision=str(spec["revision"]),
            source_url=str(spec["source_url"]), item_id=f"{split_name}:{row_idx}",
            license_id=str(spec["license_id"]), license_url=str(spec["license_url"]),
            identity=f"zeroth-speaker-unknown:{split_name}",
            content=f"zeroth-text:{hashlib.sha256(text.encode()).hexdigest()[:20]}", language="ko",
            split="holdout" if split_name == "test" else "train",
        ))
    return records


def import_fma_instrumental_hf(root: Path, spec: dict[str, object], manifest_dir: Path) -> list[SourceRecord]:
    require_review(root, spec)
    records: list[SourceRecord] = []
    for item in _rows(root):
        row = item["row"]
        attribution = str(row.get("attribution", ""))
        if not _instrumental_row(row):
            continue
        license_line = next((line for line in attribution.splitlines() if line.lower().startswith("license:")), "")
        license_url = license_line.split(":", 1)[1].strip() if ":" in license_line else str(spec["license_url"])
        track_id = str(row.get("track_id", item["row_idx"]))
        artist = str(row.get("artist", "unknown"))
        path = root / str(item["local_path"])
        records.append(_record(
            path=path, manifest_dir=manifest_dir, source_id=f"fma-instrumental-{track_id}",
            component="music", fake=0, dataset=str(spec["source_dataset"]), revision=str(spec["revision"]),
            source_url=str(row.get("original_page", spec["source_url"])), item_id=track_id,
            license_id=license_url, license_url=license_url, identity=f"fma-artist:{artist}",
            content=f"fma-track:{track_id}", language="und",
        ))
    return records


def prepare_source_manifest(
    catalog: dict[str, dict[str, object]],
    staging_root: str | Path,
    output: str | Path,
    *,
    generated_manifests: Iterable[str | Path] = (),
    seed: int = 2608,
) -> dict[str, object]:
    manifest_path = Path(output)
    manifest_dir = manifest_path.parent
    root = Path(staging_root)
    records: list[SourceRecord] = []
    if (root / "mlaad-tiny" / "REVIEW.json").is_file():
        records.extend(import_mlaad(root / "mlaad-tiny", catalog["mlaad-tiny"], manifest_dir))
    if (root / "mlaad-tiny-original-en" / "REVIEW.json").is_file():
        records.extend(import_mlaad(
            root / "mlaad-tiny-original-en", catalog["mlaad-tiny-original-en"], manifest_dir
        ))
    if (root / "zeroth-korean" / "REVIEW.json").is_file():
        records.extend(import_zeroth(root / "zeroth-korean", catalog["zeroth-korean"], manifest_dir))
    if (root / "zeroth-korean-hf" / "REVIEW.json").is_file():
        records.extend(import_zeroth_hf(root / "zeroth-korean-hf", catalog["zeroth-korean-hf"], manifest_dir))
    if (root / "fma-small" / "REVIEW.json").is_file():
        records.extend(import_fma(root / "fma-small", catalog["fma-small"], manifest_dir))
    # SONICS distributes full generated songs rather than isolated components.
    # Keep it as a private challenge source until reviewed vocal/accompaniment
    # stems exist; importing full songs as music-only would corrupt presence labels.
    if (root / "fma-instrumental-hf" / "REVIEW.json").is_file():
        records.extend(import_fma_instrumental_hf(root / "fma-instrumental-hf", catalog["fma-instrumental-hf"], manifest_dir))
    for path in generated_manifests:
        records.extend(import_generated_manifest(path, manifest_dir))
    if not records:
        raise ValueError("no reviewed sources were available to prepare")
    forced = {"MMS-TTS-KOR", "AudioLDM2"}
    assigned = assign_source_splits(records, seed=seed, holdout_generator_families=forced)
    audit = audit_source_split_leakage(assigned)
    write_source_manifest(manifest_path, assigned)
    counts = Counter((record.component, record.fake, record.split) for record in assigned)
    summary = {
        "format_version": 2,
        "records": len(assigned),
        "manifest_sha256": sha256_file(manifest_path),
        "forced_holdout_generators": sorted(forced),
        "bucket_split_counts": {f"{component}_{fake}_{split}": count for (component, fake, split), count in sorted(counts.items())},
        "leakage_audit": audit,
    }
    _write_json(manifest_dir / "SOURCE_SUMMARY.json", summary)
    return summary


def audit_corpus(manifest_path: str | Path, split_path: str | Path | None = None) -> dict[str, object]:
    path = Path(manifest_path)
    records = read_manifest(path)
    splits = json.loads(Path(split_path or path.with_name("splits.json")).read_text(encoding="utf-8"))
    failures: list[str] = []
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    parent_source_splits: dict[str, str] = {}
    hashes: list[str] = []
    for record in records:
        audio = Path(record.path)
        if not audio.is_absolute():
            audio = path.parent / audio
        if not audio.is_file():
            failures.append(f"missing:{record.sample_id}")
            continue
        actual = sha256_file(audio)
        if actual != record.sha256:
            failures.append(f"hash:{record.sample_id}")
        try:
            with wave.open(str(audio), "rb") as handle:
                if handle.getframerate() != 16_000 or handle.getsampwidth() != 2 or handle.getnchannels() not in {1, 2}:
                    failures.append(f"format:{record.sample_id}")
                if handle.getnframes() <= 0:
                    failures.append(f"empty:{record.sample_id}")
        except (wave.Error, EOFError):
            failures.append(f"decode:{record.sample_id}")
        split = splits.get(record.sample_id)
        if split not in {"train", "dev", "holdout"}:
            failures.append(f"split:{record.sample_id}")
        else:
            split_counts[split] += 1
            if record.split_group.startswith("parents:"):
                for parent_source in record.split_group.removeprefix("parents:").split("+"):
                    if not parent_source:
                        failures.append(f"empty-parent:{record.sample_id}")
                        continue
                    prior = parent_source_splits.setdefault(parent_source, split)
                    if prior != split:
                        failures.append(f"parent-split-leak:{parent_source}:{prior}:{split}")
        label_counts[f"v{record.voice_fake if record.voice_present else 'x'}_m{record.music_fake if record.music_present else 'x'}"] += 1
        if record.source_dataset == "run0-procedural-bootstrap-fixture":
            failures.append(f"fixture:{record.sample_id}")
        hashes.append(f"{actual}  {record.path}")
    if failures:
        raise ValueError(f"corpus audit failed ({len(failures)}): {failures[:20]}")
    hash_path = path.with_name("HASHES.sha256")
    hash_path.write_text("\n".join(hashes) + "\n", encoding="utf-8", newline="\n")
    report = {
        "format_version": 1,
        "files": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "parent_source_split_keys": len(parent_source_splits),
        "manifest_sha256": sha256_file(path),
        "hash_list_sha256": sha256_file(hash_path),
        "failures": 0,
    }
    _write_json(path.with_name("AUDIT.json"), report)
    return report
