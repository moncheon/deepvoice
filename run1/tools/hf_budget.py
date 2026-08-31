#!/usr/bin/env python3
"""Reserve and settle a hard-capped Hugging Face Jobs budget ledger."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


FORMAT_VERSION = 1
DEFAULT_CAP_USD = 10.0


def _read(path: Path, cap: float) -> dict[str, object]:
    if not path.is_file():
        return {
            "format_version": FORMAT_VERSION,
            "cap_usd": cap,
            "reservations": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported budget ledger format")
    if abs(float(payload["cap_usd"]) - cap) > 1e-9:
        raise ValueError("requested cap differs from existing ledger")
    return payload


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _totals(payload: dict[str, object]) -> tuple[float, float]:
    spent = 0.0
    active = 0.0
    for record in payload["reservations"]:  # type: ignore[index]
        if record["status"] == "reserved":
            active += float(record["maximum_cost_usd"])
        else:
            spent += float(record["actual_cost_usd"])
    return spent, active


def reserve(args: argparse.Namespace) -> None:
    path = Path(args.ledger)
    payload = _read(path, args.cap_usd)
    spent, active = _totals(payload)
    maximum = args.timeout_minutes / 60.0 * args.rate_per_hour
    projected = spent + active + maximum
    if projected > args.cap_usd + 1e-9:
        raise SystemExit(
            f"budget blocked: spent=${spent:.4f}, active=${active:.4f}, "
            f"new_max=${maximum:.4f}, cap=${args.cap_usd:.2f}"
        )
    reservation = {
        "reservation_id": uuid.uuid4().hex,
        "name": args.name,
        "flavor": args.flavor,
        "rate_per_hour_usd": args.rate_per_hour,
        "timeout_minutes": args.timeout_minutes,
        "maximum_cost_usd": maximum,
        "status": "reserved",
        "reserved_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["reservations"].append(reservation)  # type: ignore[index]
    _atomic_write(path, payload)
    print(json.dumps(reservation, sort_keys=True))


def settle(args: argparse.Namespace) -> None:
    path = Path(args.ledger)
    payload = _read(path, args.cap_usd)
    matches = [
        record for record in payload["reservations"]  # type: ignore[index]
        if record["reservation_id"] == args.reservation_id
    ]
    if len(matches) != 1:
        raise SystemExit("reservation ID was not found exactly once")
    record = matches[0]
    if record["status"] != "reserved":
        raise SystemExit("reservation is already settled")
    if record.get("job_id") not in (None, args.job_id):
        raise SystemExit("settlement job ID differs from attached job")
    actual = args.runtime_seconds / 3600.0 * float(record["rate_per_hour_usd"])
    if actual > float(record["maximum_cost_usd"]) + 1e-6:
        raise SystemExit("actual cost exceeds reserved maximum; inspect billing before continuing")
    record.update({
        "status": args.status,
        "job_id": args.job_id,
        "runtime_seconds": args.runtime_seconds,
        "actual_cost_usd": actual,
        "settled_at": datetime.now(timezone.utc).isoformat(),
    })
    spent, active = _totals(payload)
    payload["summary"] = {
        "spent_usd": spent,
        "active_reserved_usd": active,
        "remaining_unreserved_usd": float(payload["cap_usd"]) - spent - active,
    }
    _atomic_write(path, payload)
    print(json.dumps({"reservation": record, "summary": payload["summary"]}, sort_keys=True))


def attach(args: argparse.Namespace) -> None:
    path = Path(args.ledger)
    payload = _read(path, args.cap_usd)
    matches = [
        record for record in payload["reservations"]  # type: ignore[index]
        if record["reservation_id"] == args.reservation_id
    ]
    if len(matches) != 1 or matches[0]["status"] != "reserved":
        raise SystemExit("active reservation ID was not found exactly once")
    record = matches[0]
    prior = record.get("job_id")
    if prior is not None and prior != args.job_id:
        raise SystemExit("reservation is already attached to a different job")
    record["job_id"] = args.job_id
    _atomic_write(path, payload)
    print(json.dumps(record, sort_keys=True))


def status(args: argparse.Namespace) -> None:
    payload = _read(Path(args.ledger), args.cap_usd)
    spent, active = _totals(payload)
    print(json.dumps({
        "cap_usd": float(payload["cap_usd"]),
        "spent_usd": spent,
        "active_reserved_usd": active,
        "remaining_unreserved_usd": float(payload["cap_usd"]) - spent - active,
    }, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--ledger", required=True)
    result.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    subparsers = result.add_subparsers(dest="command", required=True)
    reserve_parser = subparsers.add_parser("reserve")
    reserve_parser.add_argument("--name", required=True)
    reserve_parser.add_argument("--flavor", required=True)
    reserve_parser.add_argument("--rate-per-hour", required=True, type=float)
    reserve_parser.add_argument("--timeout-minutes", required=True, type=float)
    reserve_parser.set_defaults(function=reserve)
    attach_parser = subparsers.add_parser("attach")
    attach_parser.add_argument("--reservation-id", required=True)
    attach_parser.add_argument("--job-id", required=True)
    attach_parser.set_defaults(function=attach)
    settle_parser = subparsers.add_parser("settle")
    settle_parser.add_argument("--reservation-id", required=True)
    settle_parser.add_argument("--job-id", required=True)
    settle_parser.add_argument("--runtime-seconds", required=True, type=float)
    settle_parser.add_argument(
        "--status", choices=("completed", "error", "canceled"), required=True
    )
    settle_parser.set_defaults(function=settle)
    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(function=status)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
