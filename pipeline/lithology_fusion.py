"""
Final rock type from two independent sources: the image and the XRF reading.

    XRF elements (%) -> oxides -> P_XRF(rock)           chemistry
    image probabilities      -> P_image(rock)           what it looks like
    P(rock) ∝ P_image^w_image × P_XRF^w_xrf            weighted product (log-linear pool)
    + a conflict flag when the chemistry rules out the rock family the image prefers

The image cannot separate a rock from its "mixed" variants (a phosphatic
limestone looks like a limestone): that is the XRF's job. So the decision is
made in a richer set of 10 TARGET classes, and each image class spreads its
probability over the target classes it looks like.

Two XRF scorers:
  A3 (default, no training)  Aitchison distance to published Benguerir lab-XRF
                             compositions, turned into a likelihood with a Gaussian kernel.
  A4 (optional, trained)     Random forest on oxide log-ratios
                             (models/classification/xrf-randomforest). Used as A3 x A4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pipeline import weights

NAN = float("nan")

TARGET_CLASSES = [
    "phosphate", "phosphatic_limestone", "phosphatic_marl",
    "limestone", "dolomitic_limestone", "marl", "siliceous_marl",
    "flint", "phosphate_flint", "out_of_series",
]

# ── chemistry ───────────────────────────────────────────────────────────────
# element -> (oxide, mass factor), e.g. 1 % P = 2.2914 % P2O5
OXIDE_FACTORS = {
    "P": ("P2O5", 2.2914), "Ca": ("CaO", 1.3992), "Si": ("SiO2", 2.1393),
    "Al": ("Al2O3", 1.8895), "Fe": ("Fe2O3", 1.4297), "K": ("K2O", 1.2046), "Mg": ("MgO", 1.6583),
}
DETECTION_LIMIT_PCT = {"P2O5": 0.10, "CaO": 0.05, "SiO2": 0.30, "Al2O3": 0.30,
                       "Fe2O3": 0.05, "MgO": 1.50, "K2O": 0.05}
ZERO_REPLACEMENT_FRACTION = 0.65          # an undetected oxide = 65 % of its detection limit
A3_OXIDES = ("P2O5", "CaO", "SiO2", "Al2O3", "Fe2O3", "MgO")
A4_OXIDES = ("P2O5", "CaO", "SiO2", "Al2O3", "Fe2O3", "MgO", "K2O")

# Published Benguerir lab-XRF means (wt %, order = A3_OXIDES):
# Minerals 2026 16:619 (marl, limestone, phosphate, siliceous marl) and
# Buildings 2024 14:2375 (flint, phosphate flint, dolomite).
REFERENCES = {
    "marl":                (5.2, 32.4, 21.7, 3.4, 2.1, 1.5),
    "limestone":           (2.4, 47.2, 9.8, 1.4, 0.9, 0.8),
    "phosphate":           (18.6, 44.8, 13.2, 2.0, 1.3, 0.6),
    "siliceous_marl":      (4.7, 16.5, 48.7, 5.1, 2.6, 1.0),
    "flint":               (3.57, 6.97, 81.8, 0.27, 1.07, 1.77),
    "phosphate_flint":     (13.7, 25.7, 50.23, 0.23, 0.43, 1.7),
    "dolomitic_limestone": (2.6, 32.8, 6.97, 0.4, 0.2, 18.3),
}
# No published analysis for these two: 50/50 mix of their end-members.
REFERENCES["phosphatic_limestone"] = tuple((a + b) / 2 for a, b in zip(REFERENCES["phosphate"], REFERENCES["limestone"]))
REFERENCES["phosphatic_marl"] = tuple((a + b) / 2 for a, b in zip(REFERENCES["phosphate"], REFERENCES["marl"]))

A3_TAU = 0.4                    # kernel width (log-ratio units)
A3_OUT_OF_SERIES_DISTANCE = 3.5  # beyond this distance, "out of series" wins
A4_UNSEEN_CLASS_LIK = 0.01
PLAUSIBLE_FRACTION = 0.10
BLIND_FLOOR = 0.05
POOL_EPS = 1e-4


def oxides_from_elements(elements: dict[str, float]) -> dict[str, float]:
    """Element % -> oxide %; below the detection limit -> NaN."""
    ox = {o: NAN for o in A4_OXIDES}
    for sym, val in (elements or {}).items():
        if sym in OXIDE_FACTORS and val is not None:
            name, f = OXIDE_FACTORS[sym]
            v = float(val) * f
            ox[name] = v if v >= DETECTION_LIMIT_PCT.get(name, 0.0) else NAN
    return ox


def _measured(ox: dict, name: str) -> bool:
    v = ox.get(name, NAN)
    return v is not None and not math.isnan(v)


def closed(ox: dict[str, float], comps: tuple[str, ...]) -> np.ndarray:
    """Composition rescaled to sum to 1, undetected parts replaced by a small positive value."""
    x = []
    for c in comps:
        v = ox.get(c, NAN)
        if v is None or math.isnan(v) or v <= 0:
            v = DETECTION_LIMIT_PCT.get(c, 0.05) * ZERO_REPLACEMENT_FRACTION
        x.append(v)
    x = np.asarray(x, float)
    return x / x.sum()


def clr(x: np.ndarray) -> np.ndarray:
    """Centred log-ratio: log of each part relative to the geometric mean."""
    lx = np.log(x)
    return lx - lx.mean()


def a4_features(ox: dict[str, float]) -> np.ndarray:
    """All pairwise log-ratios of the 7 oxides (21) + a 'MgO not measured' flag."""
    lx = np.log(closed(ox, A4_OXIDES))
    n = len(lx)
    feats = [lx[i] - lx[j] for i in range(n) for j in range(i + 1, n)]
    feats.append(0.0 if _measured(ox, "MgO") else 1.0)
    return np.asarray(feats, float)


def _normalize(d: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in d.values())
    if s <= 0:
        return {c: 1.0 / len(TARGET_CLASSES) for c in TARGET_CLASSES}
    return {c: max(0.0, d.get(c, 0.0)) / s for c in TARGET_CLASSES}


def a3_probs(ox: dict[str, float]) -> dict[str, float]:
    """Likelihood exp(-d² / 2τ²) of each reference, d = Aitchison distance.
    MgO is left out when the analyser did not report it (too weak to mean zero)."""
    comps = tuple(c for c in A3_OXIDES if c != "MgO" or _measured(ox, "MgO"))
    x = clr(closed(ox, comps))
    lik = {}
    for cls, ref in REFERENCES.items():
        r = clr(closed({c: ref[A3_OXIDES.index(c)] for c in comps}, comps))
        lik[cls] = math.exp(-float(np.linalg.norm(x - r)) ** 2 / (2 * A3_TAU ** 2))
    lik["out_of_series"] = math.exp(-A3_OUT_OF_SERIES_DISTANCE ** 2 / (2 * A3_TAU ** 2))
    return _normalize(lik)


_rf = None


def a4_probs(ox: dict[str, float]) -> dict[str, float]:
    global _rf
    if _rf is None:
        import joblib
        _rf = joblib.load(weights.path(weights.LITHOLOGY_DIR) / "xrf_a4_world.joblib")["model"]
    p = _rf.predict_proba(a4_features(ox)[None, :])[0]
    lik = {c: A4_UNSEEN_CLASS_LIK for c in TARGET_CLASSES}
    for c, v in zip(_rf.classes_, p):
        if str(c) in lik:
            lik[str(c)] = float(v) + A4_UNSEEN_CLASS_LIK
    return _normalize(lik)


def xrf_probs(elements: dict[str, float], method: str = "a3") -> dict[str, float] | None:
    ox = oxides_from_elements(elements)
    if not any(_measured(ox, c) for c in ("CaO", "SiO2", "P2O5")):
        return None
    p3 = a3_probs(ox)
    if method == "a3xa4":
        p4 = a4_probs(ox)
        return _normalize({c: p3[c] * p4[c] for c in TARGET_CLASSES})
    return p3


# ── image classes -> target classes ─────────────────────────────────────────
IMAGE_TO_TARGET = {
    "limestone":           ("limestone", "phosphatic_limestone"),
    "dolomitic_limestone": ("dolomitic_limestone",),
    "marl":                ("marl", "phosphatic_marl", "siliceous_marl"),
    "flint":               ("flint", "phosphate_flint"),
    "out_of_series":       ("out_of_series",),
}


def image_to_target(image_probs: dict[str, float]) -> tuple[dict[str, float] | None, float]:
    """(target-class probabilities, no-rock probability). A class the image
    cannot see gets a small floor: 'never learned' is not 'no'."""
    out = {c: 0.0 for c in TARGET_CLASSES}
    no_rock = 0.0
    for name, p in (image_probs or {}).items():
        if name == "no_rock":
            no_rock += float(p)
            continue
        targets = IMAGE_TO_TARGET.get(name, ("out_of_series",))
        for t in targets:
            out[t] += float(p) / len(targets)
    if sum(out.values()) <= 0:
        return None, no_rock
    out = _normalize(out)
    return _normalize({c: v if v > 0 else BLIND_FLOOR for c, v in out.items()}), no_rock


# When the image alone cannot separate a rock from its mixed variant, report the plain rock.
_TIE_PRIORITY = ["limestone", "dolomitic_limestone", "marl", "flint", "phosphate", "out_of_series",
                 "siliceous_marl", "phosphatic_limestone", "phosphatic_marl", "phosphate_flint"]


def best_class(probs: dict[str, float]) -> str:
    return max(probs, key=lambda c: (round(probs[c], 6), -_TIE_PRIORITY.index(c)))


def pool(image: dict[str, float], xrf: dict[str, float], w_image: float, w_xrf: float) -> dict[str, float]:
    """Weighted product of the two distributions, computed in log space."""
    logs = {c: w_image * math.log(image.get(c, 0.0) + POOL_EPS) + w_xrf * math.log(xrf.get(c, 0.0) + POOL_EPS)
            for c in TARGET_CLASSES}
    m = max(logs.values())
    return _normalize({c: math.exp(v - m) for c, v in logs.items()})


# ── conflict detection, at the level of rock FAMILIES ───────────────────────
# Families a target class can LOOK like (a phosphatic limestone looks like
# either parent, so "limestone" from the image does not contradict it).
_LOOKS_LIKE = {
    "phosphate": {"phosphatic"}, "phosphatic_limestone": {"phosphatic", "carbonate"},
    "phosphatic_marl": {"phosphatic", "clay"}, "limestone": {"carbonate"},
    "dolomitic_limestone": {"carbonate"}, "marl": {"clay"}, "siliceous_marl": {"clay", "siliceous"},
    "flint": {"siliceous"}, "phosphate_flint": {"siliceous", "phosphatic"}, "out_of_series": {"other"},
}
_IMAGE_FAMILY = {"limestone": "carbonate", "dolomitic_limestone": "carbonate", "marl": "clay",
                 "flint": "siliceous", "out_of_series": "other"}


def image_families(image_probs: dict[str, float]) -> dict[str, float]:
    fam: dict[str, float] = {}
    for name, p in (image_probs or {}).items():
        if name != "no_rock":
            f = _IMAGE_FAMILY.get(name, "other")
            fam[f] = fam.get(f, 0.0) + float(p)
    s = sum(fam.values())
    return {f: m / s for f, m in fam.items()} if s > 0 else {}


def conflict_score(families: dict[str, float], xrf: dict[str, float]) -> float:
    """0.5 + 0.5 x (best family ruled out by XRF - best family still plausible), in [0, 1].
    A family is plausible when one of its classes has >= 10 % of the top XRF probability."""
    best = max(xrf.values())
    plausible: set[str] = set()
    for c in TARGET_CLASSES:
        if xrf[c] >= PLAUSIBLE_FRACTION * best:
            plausible |= _LOOKS_LIKE[c]
    ok = max((m for f, m in families.items() if f in plausible), default=0.0)
    bad = max((m for f, m in families.items() if f not in plausible), default=0.0)
    return float(max(0.0, min(1.0, 0.5 + 0.5 * (bad - ok))))


@dataclass
class Decision:
    target: str | None            # None = no rock
    confidence: float
    conflict: bool
    conflict_score: float
    source: str                   # "image + XRF" | "XRF only" | "image only"
    probs: dict[str, float] = field(default_factory=dict)


def decide(image_probs: dict[str, float] | None, elements: dict[str, float] | None,
           xrf_method: str = "a3", w_image: float = 1.0, w_xrf: float = 1.0,
           conflict_threshold: float = 0.40) -> Decision | None:
    image_probs = image_probs if w_image > 0 else None
    img, no_rock = image_to_target(image_probs) if image_probs else (None, 0.0)
    xrf = xrf_probs(elements, xrf_method) if elements and w_xrf > 0 else None

    # An empty tray must not become a confident rock prediction from its small
    # residual rock probability. Chemistry may still decide, flagged as conflict.
    empty = no_rock > 0.5
    if empty:
        img = None
        if xrf is None:
            return Decision(None, no_rock, False, 0.0, "image only")

    if img is not None and xrf is not None:
        probs = pool(img, xrf, w_image, w_xrf)
        score, source = conflict_score(image_families(image_probs), xrf), "image + XRF"
    elif xrf is not None:
        probs, score, source = xrf, (1.0 if empty else 0.0), "XRF only"
    elif img is not None:
        probs, score, source = img, 0.0, "image only"
    else:
        return None

    best = best_class(probs)
    return Decision(best, float(probs[best]), score >= conflict_threshold, score, source, probs)
