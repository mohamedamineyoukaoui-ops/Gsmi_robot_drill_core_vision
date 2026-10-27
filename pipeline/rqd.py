"""
Step 6 — RQD and fracture frequency.

RQD (Rock Quality Designation, Deere 1964):

    RQD = total length of intact pieces >= 10 cm / total core length x 100

The numerator is the UNION of the counting pieces' depth ranges, not their
sum, so two overlapping pieces (e.g. rock enclosed in a crack loop and the
rock around it) are counted once and RQD can never exceed the core length.

Fracture frequency (fractures per metre) is computed next to it. RQD throws
information away at the 10 cm cutoff (a core broken every 9 cm and a crushed
core both score 0 %); fracture frequency does not, and it only needs to know
that two pieces are separate, not their exact lengths.
"""

from __future__ import annotations

from typing import Any

QUALITY_CLASSES = [(90, "Excellent"), (75, "Good"), (50, "Fair"), (25, "Poor"), (0, "Very Poor")]


def quality_class(rqd_percent: float | None) -> str | None:
    if rqd_percent is None:
        return None
    return next(name for limit, name in QUALITY_CLASSES if rqd_percent >= limit)


def union_length_px(pieces: list[dict]) -> int:
    """Total length covered by the counting pieces, overlaps counted once."""
    intervals = sorted((int(p["y_start"]), int(p["y_end"])) for p in pieces if p.get("counts_for_rqd"))
    total, cur_s, cur_e = 0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def rqd_from_pieces(pieces: list[dict], core_length_cm: float, px_per_cm: float) -> dict[str, Any]:
    intact_cm = round(union_length_px(pieces) / px_per_cm, 2) if px_per_cm > 0 else 0.0
    percent = min(intact_cm / core_length_cm * 100.0, 100.0) if core_length_cm > 0 else 0.0
    return {"rqd_percent": round(percent, 2), "rqd_class": quality_class(percent), "rqd_length_cm": intact_cm}


def rqd_from_zones(segments: list[dict], total_cm: float) -> float | None:
    """Fallback when the crack mask is switched off: intact zone segments >= 10 cm."""
    if total_cm <= 0:
        return None
    intact = sum(s["length_cm"] for s in segments if s["type"] == "intact" and s["length_cm"] >= 10.0)
    return max(0.0, min(100.0, intact / total_cm * 100.0))


def overall_rqd(columns: list[dict], run_length_cm: float | None = None) -> dict[str, Any]:
    """Whole tray: sum of intact lengths over a denominator.

    Denominator: the drilled run length when the start/end depths are known
    (Deere's definition: lost core counts against RQD), otherwise the core
    length detected in the image. With the run length, total core recovery
    (TCR = recovered core / run) is reported too.
    """
    intact_cm = total_cm = 0.0
    for col in columns:
        total_cm += float(col.get("core_length_cm", col.get("height_cm", 0.0)) or 0.0)
        if "rqd_length_cm" in col:
            intact_cm += float(col["rqd_length_cm"])
        else:
            intact_cm += sum(s["length_cm"] for s in col.get("segments", [])
                             if s["type"] == "intact" and s["length_cm"] >= 10.0)

    out: dict[str, Any] = {"core_length_cm": round(total_cm, 1), "rqd_length_cm": round(intact_cm, 1),
                           "run_length_cm": run_length_cm, "core_recovery_percent": None}
    if run_length_cm and run_length_cm > 0:
        denominator, source = float(run_length_cm), "run"
        out["core_recovery_percent"] = round(max(0.0, min(100.0, total_cm / run_length_cm * 100.0)), 1)
        if total_cm > 1.1 * run_length_cm:
            out["denominator_warning"] = (f"Detected core ({total_cm:.0f} cm) is longer than the run "
                                          f"({run_length_cm:.0f} cm): check the start/end depths")
    else:
        denominator, source = total_cm, "detected"
    out["rqd_denominator_cm"], out["rqd_denominator_source"] = round(denominator, 1), source

    if denominator <= 0:
        out.update(rqd_percent=None, rqd_class=None)
    else:
        percent = max(0.0, min(100.0, intact_cm / denominator * 100.0))
        out.update(rqd_percent=percent, rqd_class=quality_class(percent))
    return out


def run_length_cm(depth_from_m: float | None, depth_to_m: float | None) -> float | None:
    """Drilled run length from the depth markers, or None when they are not set."""
    a, b = float(depth_from_m or 0.0), float(depth_to_m or 0.0)
    return round((b - a) * 100.0, 2) if b > a else None


def fracture_frequency(columns: list[dict]) -> dict[str, Any]:
    """Fractures per metre, and the RQD it implies (Palmstrom: RQD ~ 115 - 3.3 Jv).

    Inside a column, every gap between two pieces is a fracture. Between two
    columns, the gap is a fracture only if the boundary pieces were not
    chained by pieces.merge_across_columns (same chain_id = no crack there).
    Jv is approximated by the frequency along the core axis, the usual
    simplification when only one scanline is available.
    """
    count, total_cm = 0, 0.0
    for col in columns:
        pieces = col.get("pieces") or []
        total_cm += float(col.get("core_length_cm", col.get("height_cm", 0.0)) or 0.0)
        count += max(0, len(pieces) - 1)

    for left, right in zip(columns[:-1], columns[1:]):
        lp, rp = left.get("pieces") or [], right.get("pieces") or []
        if not lp or not rp:
            continue                        # cannot tell: do not invent a fracture
        healed = lp[-1].get("chain_id") is not None and lp[-1].get("chain_id") == rp[0].get("chain_id")
        if not healed:
            count += 1

    if total_cm <= 0:
        return {"fracture_count": None, "fracture_frequency_per_m": None, "rqd_from_jv": None}
    per_m = round(count / (total_cm / 100.0), 2)
    return {"fracture_count": count, "fracture_frequency_per_m": per_m,
            "rqd_from_jv": round(max(0.0, min(100.0, 115.0 - 3.3 * per_m)), 1)}
