from __future__ import annotations

from collections.abc import Mapping

from .price_scale import price_scale_manifest_fields


ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


def build_artifact_entrypoints(
    source_payload: Mapping[str, object],
    *,
    include_model: bool = False,
) -> dict[str, str]:
    entrypoints = {
        "analysis": "analysis.json",
        "config": "config.json",
        "forecast_summary": "forecast_summary.json",
        "horizon_diagnostics": "horizon_diag.csv",
        "metrics": "metrics.json",
        "policy_summary": "policy_summary.csv",
        "prediction": "prediction.json",
        "research_report": "research_report.md",
        "source": "source.json",
        "validation_rows": "validation_rows.csv",
        "validation_summary": "validation_summary.json",
    }
    if include_model:
        entrypoints["model"] = "model.pt"
    if str(source_payload.get("kind")) == "csv":
        entrypoints["data_snapshot"] = "data_snapshot.csv"
    return dict(sorted(entrypoints.items()))


def build_artifact_manifest(
    *,
    artifact_kind: str,
    artifact_id: str,
    parent_artifact_id: str | None,
    generated_at_utc: str,
    source_payload: Mapping[str, object],
    entrypoints: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "artifact_id": artifact_id,
        "parent_artifact_id": parent_artifact_id,
        "generated_at": generated_at_utc,
        "generated_at_utc": generated_at_utc,
        **price_scale_manifest_fields(source_payload),
        "entrypoints": dict(sorted(entrypoints.items())),
    }
