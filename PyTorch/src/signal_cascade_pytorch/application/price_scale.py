from __future__ import annotations

from typing import Mapping


DEFAULT_PRICE_SCALE = 1.0
DEFAULT_PRICE_SCALE_ORIGIN = "default"
LEGACY_PRICE_SCALE_ORIGIN = "legacy"
MANUAL_PRICE_SCALE_ORIGIN = "manual"


def coerce_positive_price_scale(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0.0 else None


def resolve_requested_price_scale(payload: Mapping[str, object] | None) -> float | None:
    if payload is None:
        return None
    return coerce_positive_price_scale(payload.get("requested_price_scale"))


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def resolve_effective_price_scale(payload: Mapping[str, object] | None) -> float:
    if payload is None:
        return DEFAULT_PRICE_SCALE
    effective_value = payload.get("effective_price_scale", payload.get("price_scale"))
    resolved = coerce_positive_price_scale(effective_value)
    if resolved is not None:
        return resolved
    requested = resolve_requested_price_scale(payload)
    return requested if requested is not None else DEFAULT_PRICE_SCALE


def normalize_price_scale_payload(
    payload: Mapping[str, object] | None,
    *,
    requested_price_scale: float | None = None,
) -> dict[str, object]:
    normalized = dict(payload or {})
    explicit_requested = coerce_positive_price_scale(requested_price_scale)
    requested = explicit_requested
    if requested is None:
        requested = resolve_requested_price_scale(normalized)
    effective = resolve_effective_price_scale(
        {
            **normalized,
            **({"requested_price_scale": requested} if requested is not None else {}),
        }
    )
    origin = str(normalized.get("price_scale_origin", "") or "").strip()
    if explicit_requested is not None:
        origin = MANUAL_PRICE_SCALE_ORIGIN
    elif not origin:
        if requested is not None:
            origin = MANUAL_PRICE_SCALE_ORIGIN
        elif coerce_positive_price_scale(normalized.get("price_scale")) is not None:
            origin = LEGACY_PRICE_SCALE_ORIGIN
        else:
            origin = DEFAULT_PRICE_SCALE_ORIGIN
    provider_scale_confirmed = (
        False
        if explicit_requested is not None
        else _coerce_bool(normalized.get("provider_scale_confirmed", False))
    )

    normalized["effective_price_scale"] = effective
    normalized["price_scale"] = effective
    normalized["price_scale_origin"] = origin
    normalized["provider_scale_confirmed"] = provider_scale_confirmed
    if requested is not None:
        normalized["requested_price_scale"] = requested
    else:
        normalized.pop("requested_price_scale", None)
    return normalized


def price_scale_manifest_fields(payload: Mapping[str, object] | None) -> dict[str, object]:
    normalized = normalize_price_scale_payload(payload)
    return {
        "effective_price_scale": normalized["effective_price_scale"],
        "price_scale_origin": normalized["price_scale_origin"],
        "provider_scale_confirmed": normalized["provider_scale_confirmed"],
    }
