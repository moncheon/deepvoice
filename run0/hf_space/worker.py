#!/usr/bin/env python3
"""Minimal token-protected controller for enumerated private batch jobs."""

from __future__ import annotations

import hmac
import json
import os
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKSPACE = Path("/workspace")
STATE_LOCK = threading.Lock()
STATE: dict[str, object] = {"stage": "idle", "started_at": None, "ended_at": None, "returncode": None, "detail": ""}


def _command(stage: str) -> list[str]:
    manifest = os.environ.get("RUN0_MANIFEST", "/data/input/manifest.jsonl")
    if stage == "validate":
        return ["python3", "manifest_cli.py", "validate", manifest]
    if stage == "bootstrap":
        return [
            "python3", "generate_audio.py", "bootstrap",
            "--output", os.environ.get("RUN0_BOOTSTRAP_OUTPUT", "/data/output/bootstrap-v1"),
            "--samples-per-combination", os.environ.get("RUN0_BOOTSTRAP_COUNT", "4"),
        ]
    if stage == "build-corpus":
        if os.environ.get("RUN0_LICENSES_REVIEWED", "").lower() != "true":
            raise ValueError("RUN0_LICENSES_REVIEWED=true is required for source corpus generation")
        return [
            "python3", "generate_audio.py", "build",
            "--source-manifest", os.environ.get("RUN0_SOURCE_MANIFEST", "/data/input/sources.jsonl"),
            "--output", os.environ.get("RUN0_CORPUS_OUTPUT", "/data/output/corpus-v1"),
        ]
    if stage == "stage-models":
        return [
            "python3", "generate_open_audio.py", "stage-models",
            "--root", os.environ.get("RUN0_GENERATOR_MODEL_ROOT", "/data/staging/models"),
        ]
    if stage == "review-models":
        return [
            "python3", "generate_open_audio.py", "review-models",
            "--root", os.environ.get("RUN0_GENERATOR_MODEL_ROOT", "/data/staging/models"),
        ]
    if stage == "generate-open-audio":
        if os.environ.get("RUN0_LICENSES_REVIEWED", "").lower() != "true":
            raise ValueError("RUN0_LICENSES_REVIEWED=true is required for open-model generation")
        return [
            "python3", "generate_open_audio.py", "generate",
            "--root", os.environ.get("RUN0_GENERATOR_MODEL_ROOT", "/data/staging/models"),
            "--output", os.environ.get("RUN0_GENERATED_OUTPUT", "/data/output/generated-v1"),
            "--mms-count", os.environ.get("RUN0_MMS_COUNT", "2000"),
            "--musicgen-count", os.environ.get("RUN0_MUSICGEN_COUNT", "512"),
            "--audioldm-count", os.environ.get("RUN0_AUDIOLDM_COUNT", "128"),
        ]
    if stage == "generate-open-audio-smoke":
        if os.environ.get("RUN0_LICENSES_REVIEWED", "").lower() != "true":
            raise ValueError("RUN0_LICENSES_REVIEWED=true is required for open-model generation")
        return [
            "python3", "generate_open_audio.py", "generate",
            "--root", os.environ.get("RUN0_GENERATOR_MODEL_ROOT", "/data/staging/models"),
            "--output", "/data/output/generated-smoke",
            "--mms-count", "4",
            "--musicgen-count", "4",
            "--audioldm-count", "1",
        ]
    if stage == "upload-generated":
        repo_id = os.environ.get("RUN0_GENERATED_REPO", "")
        if not repo_id:
            raise ValueError("RUN0_GENERATED_REPO is required")
        return [
            "python3", "generate_open_audio.py", "upload",
            "--output", os.environ.get("RUN0_GENERATED_OUTPUT", "/data/output/generated-v1"),
            "--repo-id", repo_id,
        ]
    if stage == "train":
        if os.environ.get("RUN0_LICENSES_REVIEWED", "").lower() != "true":
            raise ValueError("RUN0_LICENSES_REVIEWED=true is required for training")
        return [
            "python3", "train.py", "--manifest", manifest, "--output-dir", "/data/output/model",
            "--backbone-path", os.environ.get("RUN0_BACKBONE_DIR", "/data/output/model/backbone"),
            "--device", "cuda", "--acknowledge-licenses-reviewed",
        ]
    if stage == "smoke":
        return [
            "python3", "script.py", "--test-dir", os.environ.get("RUN0_TEST_DIR", "/data/input/test"),
            "--sample-submission", os.environ.get("RUN0_SAMPLE_SUBMISSION", "/data/input/sample_submission.csv"),
            "--model-dir", os.environ.get("RUN0_MODEL_DIR", "/data/output/model"),
            "--output", "/data/output/submission.csv", "--device", "cuda",
        ]
    raise ValueError(f"unsupported stage: {stage}")


def _run(stage: str) -> None:
    started = time.time()
    with STATE_LOCK:
        STATE.update(stage=stage, started_at=started, ended_at=None, returncode=None, detail="running")
    log_path = Path("/data/output") / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        command = _command(stage)
        maximum = int(os.environ.get("RUN0_MAX_SECONDS", "14400"))
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                timeout=maximum, check=False, text=True,
            )
        detail = f"completed; inspect {log_path} and artifacts before upload"
        returncode = result.returncode
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        returncode = -1
    with STATE_LOCK:
        STATE.update(stage="idle", ended_at=time.time(), returncode=returncode, detail=detail)


class Handler(BaseHTTPRequestHandler):
    server_version = "Run0Worker/0.1"

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"ok": True})
        elif self.path == "/status":
            with STATE_LOCK:
                self._json(HTTPStatus.OK, dict(STATE))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        expected = os.environ.get("RUN0_JOB_TOKEN", "")
        # A private Space consumes Authorization for Hub authentication. Keep
        # the worker's independent capability token in a separate header.
        provided = self.headers.get("X-Run0-Token", "")
        if not provided:
            provided = self.headers.get("Authorization", "").removeprefix("Bearer ")
        if not expected or not hmac.compare_digest(provided, expected):
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            stage = str(payload["stage"])
            _command(stage)  # Validate before acquiring the worker.
        except Exception as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        with STATE_LOCK:
            if STATE["stage"] != "idle":
                self._json(HTTPStatus.CONFLICT, {"error": "a job is already running"})
                return
            STATE["stage"] = "starting"
        threading.Thread(target=_run, args=(stage,), daemon=True).start()
        self._json(HTTPStatus.ACCEPTED, {"accepted": stage})

    def log_message(self, format: str, *args: object) -> None:
        print(f"http {self.address_string()} {format % args}", flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 7860), Handler).serve_forever()
