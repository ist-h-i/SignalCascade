from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .price_scale import normalize_price_scale_payload
from ..domain.entities import OHLCVBar


ARTIFACT_SOURCE_SCHEMA_VERSION = 2
STATE_RESET_BOUNDARY_SPEC_VERSION = 1
EXPLICIT_V2_CONFIG_ORIGIN = "explicit_v2"
LEGACY_INFERRED_DEFAULTS_CONFIG_ORIGIN = "legacy_inferred_defaults"
_GIT_EXCLUDED_PATHS = (
    ":(exclude)PyTorch/artifacts",
    ":(exclude)Frontend/public/dashboard-data.json",
)
_PROVENANCE_FIELDS = {
    "artifact_schema_version",
    "artifact_kind",
    "artifact_dir",
    "artifact_id",
    "parent_artifact_dir",
    "parent_artifact_id",
    "generated_at_utc",
    "git",
    "sub_artifacts",
    "data_snapshot_sha256",
    "config_sha256",
    "config_origin",
    "state_reset_boundary_spec_version",
}


def materialize_artifact_source(
    source_payload: dict[str, object],
    artifact_dir: Path,
    *,
    base_bars: Sequence[OHLCVBar] | None = None,
) -> dict[str, object]:
    artifact_dir = artifact_dir.expanduser().resolve()
    payload = normalize_price_scale_payload(
        {
            key: value for key, value in source_payload.items() if key not in _PROVENANCE_FIELDS
        }
    )
    if (
        str(payload.get("kind")) != "csv"
        or base_bars is None
        or not _can_materialize_snapshot(base_bars)
    ):
        return payload

    snapshot_path = artifact_dir / "data_snapshot.csv"
    _write_snapshot_csv(snapshot_path, base_bars)
    original_path = payload.get("source_origin_path", payload.get("path"))
    if original_path is not None:
        payload["source_origin_path"] = str(Path(str(original_path)).expanduser().resolve())
    payload["path"] = str(snapshot_path)
    payload["data_snapshot_row_count"] = len(base_bars)
    payload["data_snapshot_start_timestamp"] = (
        base_bars[0].timestamp.isoformat() if base_bars else None
    )
    payload["data_snapshot_end_timestamp"] = (
        base_bars[-1].timestamp.isoformat() if base_bars else None
    )
    return payload


def build_artifact_source_payload(
    source_payload: dict[str, object],
    artifact_dir: Path,
    *,
    artifact_kind: str,
    parent_artifact_dir: Path | None = None,
    generated_at_utc: str | None = None,
    sub_artifacts: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    artifact_dir = artifact_dir.expanduser().resolve()
    payload = normalize_price_scale_payload(
        {
            key: value for key, value in source_payload.items() if key not in _PROVENANCE_FIELDS
        }
    )
    payload["artifact_schema_version"] = ARTIFACT_SOURCE_SCHEMA_VERSION
    payload["artifact_kind"] = artifact_kind
    payload["artifact_dir"] = str(artifact_dir)
    payload["generated_at_utc"] = generated_at_utc or datetime.now(timezone.utc).isoformat()
    payload["state_reset_boundary_spec_version"] = STATE_RESET_BOUNDARY_SPEC_VERSION

    data_snapshot_path = _source_data_path(payload)
    if data_snapshot_path is not None:
        data_snapshot_sha256 = _sha256_file(data_snapshot_path)
        if data_snapshot_sha256 is not None:
            payload["data_snapshot_sha256"] = data_snapshot_sha256

    config_path = artifact_dir / "config.json"
    config_sha256 = _sha256_file(config_path)
    if config_sha256 is not None:
        payload["config_sha256"] = config_sha256
    payload["config_origin"] = _resolve_config_origin(config_path)

    if parent_artifact_dir is not None:
        resolved_parent_dir = parent_artifact_dir.expanduser().resolve()
        payload["parent_artifact_dir"] = str(resolved_parent_dir)
        parent_artifact_id = _load_artifact_id(resolved_parent_dir)
        if parent_artifact_id is not None:
            payload["parent_artifact_id"] = parent_artifact_id

    git_metadata = _collect_git_metadata(artifact_dir)
    if git_metadata is not None:
        payload["git"] = git_metadata

    if sub_artifacts:
        payload["sub_artifacts"] = _materialize_sub_artifacts(artifact_dir, sub_artifacts)

    payload["artifact_id"] = _build_artifact_id(payload)

    return payload


def build_subartifact_lineage(
    entries: dict[str, str],
    *,
    source_artifact_dir: Path | None = None,
) -> dict[str, dict[str, object]]:
    lineage: dict[str, dict[str, object]] = {}
    source_artifact_id = (
        _load_artifact_id(source_artifact_dir.expanduser().resolve())
        if source_artifact_dir is not None
        else None
    )
    for name, materialization in entries.items():
        lineage[name] = {"materialization": materialization}
        if source_artifact_dir is not None:
            lineage[name]["source_artifact_dir"] = str(source_artifact_dir.expanduser().resolve())
        if source_artifact_id is not None:
            lineage[name]["source_artifact_id"] = source_artifact_id
    return lineage


def _materialize_sub_artifacts(
    artifact_dir: Path,
    sub_artifacts: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for name, metadata in sorted(sub_artifacts.items()):
        enriched = dict(metadata)
        artifact_path = artifact_dir / name
        if artifact_path.exists():
            if artifact_path.name != "source.json":
                sha256 = _sha256_file(artifact_path)
                if sha256 is not None:
                    enriched["sha256"] = sha256
            schema_version = _resolve_schema_version(artifact_path)
            if schema_version is not None:
                enriched["schema_version"] = schema_version
        payload[name] = enriched
    return payload


def _source_data_path(payload: dict[str, object]) -> Path | None:
    if str(payload.get("kind")) != "csv" or payload.get("path") is None:
        return None
    return Path(str(payload["path"])).expanduser().resolve()


def _resolve_config_origin(config_path: Path) -> str | None:
    payload = _load_json(config_path)
    if payload is None:
        return None
    config_schema_version = payload.get("config_schema_version")
    if isinstance(config_schema_version, int):
        return (
            EXPLICIT_V2_CONFIG_ORIGIN
            if config_schema_version >= 2
            else LEGACY_INFERRED_DEFAULTS_CONFIG_ORIGIN
        )
    return LEGACY_INFERRED_DEFAULTS_CONFIG_ORIGIN


def _resolve_schema_version(path: Path) -> int | None:
    payload = _load_json(path)
    if payload is None:
        return None
    if path.name == "config.json":
        value = payload.get("config_schema_version")
    elif path.name == "source.json":
        value = payload.get("artifact_schema_version")
    else:
        value = payload.get("schema_version", payload.get("diagnostics_schema_version"))
    return int(value) if isinstance(value, int) else None


def _load_artifact_id(artifact_dir: Path) -> str | None:
    payload = _load_json(artifact_dir / "source.json")
    if payload is None:
        return None
    artifact_id = payload.get("artifact_id")
    return str(artifact_id) if artifact_id is not None else None


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_artifact_id(payload: dict[str, object]) -> str:
    seed_payload = {
        "artifact_kind": payload.get("artifact_kind"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "data_snapshot_sha256": payload.get("data_snapshot_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "parent_artifact_id": payload.get("parent_artifact_id"),
        "source_kind": payload.get("kind"),
        "source_path": payload.get("path"),
        "price_scale": payload.get("effective_price_scale", payload.get("price_scale")),
    }
    encoded = json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_snapshot_csv(path: Path, bars: Sequence[OHLCVBar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("timestamp", "open", "high", "low", "close", "volume"),
        )
        writer.writeheader()
        for bar in bars:
            writer.writerow(
                {
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )


def _can_materialize_snapshot(bars: Sequence[object]) -> bool:
    if not bars:
        return False
    first = bars[0]
    required_attributes = ("timestamp", "open", "high", "low", "close", "volume")
    return all(hasattr(first, attribute) for attribute in required_attributes)


def _collect_git_metadata(path: Path) -> dict[str, object] | None:
    repo_root = _run_git(path, "rev-parse", "--show-toplevel")
    if repo_root is None:
        return None

    commit_sha = _run_git(Path(repo_root), "rev-parse", "HEAD")
    tree_sha = _run_git(Path(repo_root), "rev-parse", "HEAD^{tree}")
    if commit_sha is None:
        return None

    status_output = _run_git(
        Path(repo_root),
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        ".",
        *_GIT_EXCLUDED_PATHS,
    )
    diff_output = _run_git(
        Path(repo_root),
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        ".",
        *_GIT_EXCLUDED_PATHS,
    )
    dirty = bool(status_output)
    dirty_patch_sha256 = (
        hashlib.sha256(diff_output.encode("utf-8")).hexdigest() if diff_output else None
    )
    return {
        "repo_root": repo_root,
        "head": commit_sha,
        "commit_sha": commit_sha,
        "git_commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "git_tree_sha": tree_sha,
        "dirty": dirty,
        "git_dirty": dirty,
        "dirty_patch_sha256": dirty_patch_sha256,
    }


def _run_git(cwd: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
