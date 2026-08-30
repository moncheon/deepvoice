#!/usr/bin/env python3
"""Stage, review, generate, and upload pinned open-model training audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import wave
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR if (SCRIPT_DIR / "deepvoice_run0").is_dir() else SCRIPT_DIR.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from deepvoice_run0.manifest import sha256_file
from deepvoice_run0.source_manifest import SourceRecord, write_source_manifest

CATALOG = WORKSPACE / "corpus_sources.json"
MODEL_NAMES = ("mms-tts-kor", "musicgen-small", "audioldm2-music")
RISKY_SUFFIXES = {".bin", ".ckpt", ".dll", ".exe", ".pkl", ".pickle", ".pt", ".pth", ".so", ".th"}


def _catalog() -> dict[str, dict[str, object]]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))["sources"]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_models(root: Path) -> dict[str, object]:
    """Download pinned safe-file subsets without importing model libraries."""

    from huggingface_hub import snapshot_download

    catalog = _catalog()
    root.mkdir(parents=True, exist_ok=True)
    staged: dict[str, object] = {}
    for name in MODEL_NAMES:
        spec = catalog[name]
        destination = root / name
        snapshot_download(
            repo_id=str(spec["repo_id"]), revision=str(spec["revision"]),
            local_dir=destination, allow_patterns=[str(item) for item in spec["include"]],
        )
        staged[name] = {"repo_id": spec["repo_id"], "revision": spec["revision"]}
    report = {"format_version": 1, "status": "downloaded_unreviewed", "models": staged}
    _write_json(root / "STAGED.json", report)
    return report


def review_models(root: Path) -> dict[str, object]:
    """Reject pickle/executable weights and record every model-file hash."""

    catalog = _catalog()
    if not (root / "STAGED.json").is_file():
        raise ValueError("stage-models must complete before review-models")
    reviewed: dict[str, object] = {}
    for name in MODEL_NAMES:
        spec = catalog[name]
        directory = root / name
        files = sorted(path for path in directory.rglob("*") if path.is_file() and ".cache" not in path.parts)
        risky = [path.relative_to(directory).as_posix() for path in files if path.suffix.lower() in RISKY_SUFFIXES]
        if risky:
            raise ValueError(f"unsafe model files for {name}: {risky[:10]}")
        if not any(path.suffix.lower() == ".safetensors" for path in files):
            raise ValueError(f"no safetensors weights for {name}")
        reviewed[name] = {
            "repo_id": spec["repo_id"], "revision": spec["revision"],
            "license_id": spec["license_id"], "license_url": spec["license_url"],
            "files": [
                {"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in files
            ],
        }
    report = {"format_version": 1, "status": "reviewed", "models": reviewed}
    _write_json(root / "REVIEW.json", report)
    return {"status": "reviewed", "models": {name: len(value["files"]) for name, value in reviewed.items()}}


def require_review(root: Path) -> dict[str, object]:
    path = root / "REVIEW.json"
    if not path.is_file():
        raise ValueError("review-models must complete before model execution")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "reviewed":
        raise ValueError("model review is not approved")
    for name, entry in report["models"].items():
        for item in entry["files"]:
            candidate = root / name / item["path"]
            if not candidate.is_file() or sha256_file(candidate) != item["sha256"]:
                raise ValueError(f"reviewed model changed: {name}/{item['path']}")
    return report


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    values = np.asarray(audio, dtype=np.float32).squeeze()
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"invalid generated audio: {path}")
    peak = float(np.max(np.abs(values)))
    if peak > 0.98:
        values = values * (0.98 / peak)
    pcm = np.round(np.clip(values, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _progress(stage: str, completed: int, total: int, every: int) -> None:
    if completed == total or completed == 1 or completed % every == 0:
        print(json.dumps({"generation_stage": stage, "completed": completed, "total": total}), flush=True)


def _music_prompts() -> list[str]:
    genres = ("ambient", "jazz", "classical", "electronic", "folk", "rock", "funk", "cinematic")
    instruments = ("piano and cello", "acoustic guitar and percussion", "synthesizer and drums", "flute and strings")
    moods = ("calm", "energetic", "melancholic", "bright")
    tempos = ("slow", "medium tempo", "fast")
    return [
        f"{mood} {tempo} {genre} instrumental music with {instrument}, no vocals, no speech, original composition"
        for genre in genres for instrument in instruments for mood in moods for tempo in tempos
    ]


def _source_record(
    *, path: Path, output: Path, sample_id: str, component: str, model_name: str,
    family: str, revision: str, license_id: str, license_url: str,
    prompt_id: str, seed: int, language: str, split: str, identity_group: str | None = None,
) -> SourceRecord:
    return SourceRecord(
        source_id=sample_id, path=path.relative_to(output).as_posix(), sha256=sha256_file(path),
        component=component, fake=1, source_dataset=f"open-model-generation:{model_name}",
        source_item_id=sample_id, license_id=license_id,
        consent_basis="project-authored-generic-prompt-no-identity-cloning",
        identity_group=identity_group or f"generator-source:{family}:{prompt_id}",
        # Prompt text may intentionally be shared across generator families.
        # Keep repetitions within a family atomic without coupling the held-out
        # generator family to the training generator's split.
        content_group=f"prompt:{family}:{prompt_id}",
        mix_split_group=None, generator_family=family, generator_model=model_name,
        generator_revision=revision, format_version=2, source_revision=revision,
        source_url=f"https://huggingface.co/{model_name}", license_url=license_url,
        language=language, prompt_id=prompt_id, generation_seed=seed, split=split,
    )


def generate(root: Path, output: Path, *, mms_count: int, musicgen_count: int, audioldm_count: int, seed: int) -> dict[str, object]:
    """Generate bounded non-impersonation speech/music after the review gate."""

    review = require_review(root)
    if output.exists():
        raise FileExistsError(f"generation output must be new: {output}")
    output.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("generation requires CUDA; CPU fallback is forbidden")
    device = torch.device("cuda")
    records: list[SourceRecord] = []
    prompts_ko = [line.strip() for line in (WORKSPACE / "prompts" / "ko_tts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    if mms_count:
        from transformers import AutoTokenizer, VitsModel
        from uroman import Uroman

        entry = review["models"]["mms-tts-kor"]
        model_dir = root / "mms-tts-kor"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = VitsModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
        romanizer = Uroman()
        sample_rate = int(model.config.sampling_rate)
        for index in range(mms_count):
            prompt = prompts_ko[index % len(prompts_ko)]
            prompt_id = f"ko-{index % len(prompts_ko):04d}"
            generation_seed = seed + index
            torch.manual_seed(generation_seed)
            text = romanizer.romanize_string(prompt)
            inputs = tokenizer(text=text, return_tensors="pt").to(device)
            with torch.inference_mode():
                audio = model(**inputs).waveform[0].float().cpu().numpy()
            sample_id = f"GEN_MMS_{index:05d}"
            path = output / "mms-tts-kor" / "audio" / f"{sample_id}.wav"
            _write_wav(path, audio, sample_rate)
            records.append(_source_record(
                path=path, output=output, sample_id=sample_id, component="voice",
                model_name=str(entry["repo_id"]), family="MMS-TTS-KOR", revision=str(entry["revision"]),
                license_id=str(entry["license_id"]), license_url=str(entry["license_url"]),
                prompt_id=prompt_id, seed=generation_seed, language="ko", split="holdout",
                identity_group="generator-identity:MMS-TTS-KOR-fixed-voice",
            ))
            _progress("mms-tts-kor", index + 1, mms_count, 100)
        del model, tokenizer
        torch.cuda.empty_cache()

    music_prompts = _music_prompts()
    if musicgen_count:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        entry = review["models"]["musicgen-small"]
        model_dir = root / "musicgen-small"
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = MusicgenForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
        ).to(device).eval()
        sample_rate = int(model.config.audio_encoder.sampling_rate)
        for index in range(musicgen_count):
            prompt = music_prompts[index % len(music_prompts)]
            prompt_id = f"music-{index % len(music_prompts):04d}"
            generation_seed = seed + 100_000 + index
            torch.manual_seed(generation_seed)
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=512)[0, 0].float().cpu().numpy()
            sample_id = f"GEN_MUSICGEN_{index:05d}"
            path = output / "musicgen-small" / "audio" / f"{sample_id}.wav"
            _write_wav(path, audio, sample_rate)
            records.append(_source_record(
                path=path, output=output, sample_id=sample_id, component="music",
                model_name=str(entry["repo_id"]), family="MusicGen", revision=str(entry["revision"]),
                license_id=str(entry["license_id"]), license_url=str(entry["license_url"]),
                prompt_id=prompt_id, seed=generation_seed, language="und",
                split="dev" if (index % len(music_prompts)) % 20 in {14, 15, 16} else "train",
            ))
            _progress("musicgen-small", index + 1, musicgen_count, 16)
        del model, processor
        torch.cuda.empty_cache()

    if audioldm_count:
        from diffusers import AudioLDM2Pipeline

        entry = review["models"]["audioldm2-music"]
        model_dir = root / "audioldm2-music"
        pipeline = AudioLDM2Pipeline.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, torch_dtype=torch.float16,
        ).to(device)
        sample_rate = int(pipeline.vocoder.config.sampling_rate)
        for index in range(audioldm_count):
            prompt = music_prompts[(index * 3) % len(music_prompts)]
            prompt_id = f"music-{(index * 3) % len(music_prompts):04d}"
            generation_seed = seed + 200_000 + index
            generator = torch.Generator(device=device).manual_seed(generation_seed)
            with torch.inference_mode():
                audio = pipeline(
                    prompt, negative_prompt="speech, vocals, low quality, distortion",
                    num_inference_steps=50, audio_length_in_s=10.0, generator=generator,
                ).audios[0]
            sample_id = f"GEN_AUDIOLDM2_{index:05d}"
            path = output / "audioldm2-music" / "audio" / f"{sample_id}.wav"
            _write_wav(path, audio, sample_rate)
            records.append(_source_record(
                path=path, output=output, sample_id=sample_id, component="music",
                model_name=str(entry["repo_id"]), family="AudioLDM2", revision=str(entry["revision"]),
                license_id=str(entry["license_id"]), license_url=str(entry["license_url"]),
                prompt_id=prompt_id, seed=generation_seed, language="und", split="holdout",
            ))
            _progress("audioldm2-music", index + 1, audioldm_count, 8)

    manifest = output / "sources.jsonl"
    write_source_manifest(manifest, records)
    summary = {
        "format_version": 2, "fixture_only": False, "files": len(records),
        "counts": {"mms_tts_kor": mms_count, "musicgen": musicgen_count, "audioldm2": audioldm_count},
        "manifest_sha256": sha256_file(manifest),
        "total_bytes": sum((output / record.path).stat().st_size for record in records),
    }
    _write_json(output / "SUMMARY.json", summary)
    return summary


def upload_generated(output: Path, repo_id: str) -> dict[str, object]:
    """Upload reviewed generated audio to an existing private dataset repo."""

    from huggingface_hub import HfApi

    if not (output / "SUMMARY.json").is_file() or not (output / "sources.jsonl").is_file():
        raise ValueError("generated output is incomplete")
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id, repo_type="dataset", folder_path=output,
        path_in_repo="generated-v1", commit_message="Add reviewed Run0 open-model audio",
    )
    info = api.dataset_info(repo_id)
    return {"repo_id": repo_id, "revision": info.sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stage-models", "review-models"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, default=Path("/data/staging/models"))
    item = sub.add_parser("generate")
    item.add_argument("--root", type=Path, default=Path("/data/staging/models"))
    item.add_argument("--output", type=Path, default=Path("/data/output/generated-v1"))
    item.add_argument("--mms-count", type=int, default=2000)
    item.add_argument("--musicgen-count", type=int, default=512)
    item.add_argument("--audioldm-count", type=int, default=128)
    item.add_argument("--seed", type=int, default=2608)
    item = sub.add_parser("upload")
    item.add_argument("--output", type=Path, default=Path("/data/output/generated-v1"))
    item.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    if args.command == "stage-models":
        result = stage_models(args.root)
    elif args.command == "review-models":
        result = review_models(args.root)
    elif args.command == "generate":
        result = generate(
            args.root, args.output, mms_count=args.mms_count,
            musicgen_count=args.musicgen_count, audioldm_count=args.audioldm_count,
            seed=args.seed,
        )
    else:
        result = upload_generated(args.output, args.repo_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
