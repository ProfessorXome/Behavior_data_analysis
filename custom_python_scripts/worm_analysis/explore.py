"""
Details tables, across-strain plots, and pairwise stats for worm summaries.

Typical notebook flow
---------------------
1. Build ``dataset_tables`` with per-worm and continuous-minute summary frames.
2. ``build_strain_details`` / ``build_minute_details`` → tidy analysis tables.
3. ``plot_metric_across_strains`` / ``plot_metrics_across_strains`` on those
   details frames (pairwise stars via ``pairwise_group_tests``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Match plotting.py: Arial for all explore / seaborn figures.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Arial")

# ---------------------------------------------------------------------------
# Details-table builders
# ---------------------------------------------------------------------------

# (column_name, table_suffix) — suffix is appended to "{strain_key}_"
WORM_DETAILS_SPEC: List[Tuple[str, str]] = [
    ("worm_id", "general"),
    ("recording_date", "general"),
    ("video_number", "general"),
    ("fps_true", "general"),
    ("frames_per_minute", "general"),
    ("blob_perimeter_median", "general"),
    ("reversals", "general"),
    ("active_reversals", "active"),
    ("min_active_reversals_per_minute", "active"),
    ("max_active_reversals_per_minute", "active"),
    ("active_reversals_reorient", "active"),
    ("active_reversals_reorient_scored", "active"),
    ("pct_active_reversals_reorient", "active"),
    ("reorient_per_reversal_rate", "active"),
    ("active_duration_min", "active"),
    ("active_reversals_per_active_min", "active_per_active_duration"),
    ("active_reversals_reorient_per_active_min", "active_per_active_duration"),
    ("avg_active_fwd_bout_s", "active_per_active_duration"),
    ("avg_active_rev_bout_s", "active_per_active_duration"),
    ("active_time_pos_per_active_min", "active_per_active_duration"),
    ("active_time_neg_per_active_min", "active_per_active_duration"),
    ("pct_active", "active"),
    ("active_pct_pos", "active"),
    ("active_pct_neg", "active"),

]

# Continuous-minute tables use minute_id; meta carries provenance columns.
MINUTE_DETAILS_SPEC: List[Tuple[str, str]] = [
    ("minute_id", "continuous_minutes_meta"),
    ("recording_date", "continuous_minutes_meta"),
    ("video_number", "continuous_minutes_meta"),
    ("source_worm_id", "continuous_minutes_meta"),
    ("minute_in_source", "continuous_minutes_meta"),
    ("time_start", "continuous_minutes_meta"),
    ("time_end", "continuous_minutes_meta"),
    ("n_frames", "continuous_minutes_meta"),
    ("frames_per_minute", "continuous_minutes_meta"),
    ("backfilled", "continuous_minutes_meta"),
    ("blob_perimeter_median", "continuous_minutes_general"),
    ("reversals", "continuous_minutes_general"),
    ("active_reversals", "continuous_minutes_active"),
    ("min_active_reversals_per_minute", "continuous_minutes_active"),
    ("max_active_reversals_per_minute", "continuous_minutes_active"),
    ("active_reversals_reorient", "continuous_minutes_active"),
    ("active_reversals_reorient_scored", "continuous_minutes_active"),
    ("pct_active_reversals_reorient", "continuous_minutes_active"),
    ("reorient_per_reversal_rate", "continuous_minutes_active"),
    ("active_duration_min", "continuous_minutes_active"),
    ("active_reversals_per_active_min", "continuous_minutes_active_per_active_duration"),
    ("active_reversals_reorient_per_active_min", "continuous_minutes_active_per_active_duration"),
    ("avg_active_fwd_bout_s", "continuous_minutes_active_per_active_duration"),
    ("avg_active_rev_bout_s", "continuous_minutes_active_per_active_duration"),
    ("active_time_pos_per_active_min", "continuous_minutes_active_per_active_duration"),
    ("active_time_neg_per_active_min", "continuous_minutes_active_per_active_duration"),
]

# Back-compat alias used by older notebook cells
DETAILS_SPEC = WORM_DETAILS_SPEC

def build_details_table(
    dataset_tables: Dict[str, pd.DataFrame],
    strain_key: str,
    spec: Sequence[Tuple[str, str]],
    *,
    id_col: str,
    sort_by: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Merge selected columns from ``dataset_tables`` onto ``id_col``.

    Table keys are ``f"{strain_key}_{suffix}"``. Duplicate column names in
    ``spec`` are renamed ``{col}_{suffix}``.
    """
    base_suffix = None
    for col, suffix in spec:
        if col == id_col:
            base_suffix = suffix
            break
    if base_suffix is None:
        base_suffix = spec[0][1]

    base_key = f"{strain_key}_{base_suffix}"
    if base_key not in dataset_tables:
        raise KeyError(f"Missing {base_key}")
    base = dataset_tables[base_key]
    if id_col not in base.columns:
        raise KeyError(f"{id_col!r} not in {base_key}")
    if base.empty:
        return base.iloc[0:0].copy()

    out = base[[id_col]].drop_duplicates().copy()
    seen = {id_col}
    for col, suffix in spec:
        if col == id_col:
            continue
        out_col = col if col not in seen else f"{col}_{suffix}"
        table_key = f"{strain_key}_{suffix}"
        if table_key not in dataset_tables:
            raise KeyError(f"Missing {table_key} (needed for {col})")
        tbl = dataset_tables[table_key]
        if tbl.empty:
            out[out_col] = np.nan
            seen.add(out_col)
            continue
        if id_col not in tbl.columns:
            raise KeyError(f"{id_col!r} not in {table_key}")
        if col not in tbl.columns:
            raise KeyError(f"{col!r} not in {table_key}; have {list(tbl.columns)}")
        piece = tbl[[id_col, col]].drop_duplicates(subset=[id_col]).rename(
            columns={col: out_col}
        )
        out = out.merge(piece, on=id_col, how="left")
        seen.add(out_col)

    for cov in ("video_number", "recording_date", "source_worm_id"):
        if cov in out.columns:
            out[cov] = out[cov].astype(str)

    if sort_by is None:
        sort_by = id_col
    sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)
    sort_cols = [c for c in sort_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out

def build_strain_details(
    dataset_tables: Dict[str, pd.DataFrame],
    strain_key: str,
    spec: Sequence[Tuple[str, str]] = WORM_DETAILS_SPEC,
) -> pd.DataFrame:
    """Per-worm details table (one row per ``worm_id``)."""
    return build_details_table(
        dataset_tables,
        strain_key,
        spec,
        id_col="worm_id",
        sort_by="worm_id",
    )

def build_minute_details(
    dataset_tables: Dict[str, pd.DataFrame],
    strain_key: str,
    spec: Sequence[Tuple[str, str]] = MINUTE_DETAILS_SPEC,
) -> pd.DataFrame:
    """
    Continuous-minute details table (one row per ``minute_id``).

    Expects keys like ``{strain}_continuous_minutes_general`` produced by
    ``compute_continuous_minute_metrics``.
    """
    return build_details_table(
        dataset_tables,
        strain_key,
        spec,
        id_col="minute_id",
        sort_by=["recording_date", "video_number", "minute_id"],
    )

def build_all_details(
    dataset_tables: Dict[str, pd.DataFrame],
    strain_keys: Iterable[str],
    *,
    include_worms: bool = True,
    include_minutes: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Write ``{strain}_details`` and/or ``{strain}_minute_details`` into
    ``dataset_tables`` (in place) and return the same dict.
    """
    for strain_key in strain_keys:
        if include_worms:
            key = f"{strain_key}_details"
            dataset_tables[key] = build_strain_details(dataset_tables, strain_key)
            print(
                f"{key}: {len(dataset_tables[key])} worms, "
                f"cols={list(dataset_tables[key].columns)}"
            )
        if include_minutes:
            meta_key = f"{strain_key}_continuous_minutes_meta"
            key = f"{strain_key}_minute_details"
            if meta_key not in dataset_tables:
                print(f"skip {key}: missing {meta_key}")
                continue
            dataset_tables[key] = build_minute_details(dataset_tables, strain_key)
            print(
                f"{key}: {len(dataset_tables[key])} minutes, "
                f"cols={list(dataset_tables[key].columns)}"
            )
    return dataset_tables

# ---------------------------------------------------------------------------
# Across-strain collection / comparison
# ---------------------------------------------------------------------------

_STRAIN_TABLE_ALIASES = {
    "worm": {
        "details_suffix": "details",
        "id_col": "worm_id",
        "default_table": "active_per_active_duration",
    },
    "minute": {
        "details_suffix": "minute_details",
        "id_col": "minute_id",
        "default_table": "continuous_minutes_active_per_active_duration",
        # If caller passes a short worm-style table name, map it.
        "table_map": {
            "general": "continuous_minutes_general",
            "active": "continuous_minutes_active",
            "general_per_duration": "continuous_minutes_general_per_duration",
            "active_per_duration": "continuous_minutes_active_per_duration",
            "active_per_active_duration": "continuous_minutes_active_per_active_duration",
            "meta": "continuous_minutes_meta",
        },
    },
}

def _strain_label(dataset_key: str) -> str:
    return dataset_key.removesuffix("_worms")

def resolve_strain_table_key(
    dataset_key: str,
    *,
    unit: str = "worm",
    table: Optional[str] = None,
    prefer_details: bool = True,
    dataset_tables: Optional[Dict[str, pd.DataFrame]] = None,
    metric: Optional[str] = None,
) -> str:
    """
    Pick the ``dataset_tables`` key for one strain + unit.

    Prefers ``{strain}_details`` / ``{strain}_minute_details`` when present and
    (if ``metric`` given) containing that column.
    """
    unit = str(unit).strip().lower()
    if unit not in _STRAIN_TABLE_ALIASES:
        raise ValueError("unit must be 'worm' or 'minute'")
    info = _STRAIN_TABLE_ALIASES[unit]
    details_key = f"{dataset_key}_{info['details_suffix']}"
    if prefer_details and dataset_tables is not None and details_key in dataset_tables:
        det = dataset_tables[details_key]
        if metric is None or metric in det.columns:
            return details_key

    raw_table = table or info["default_table"]
    if unit == "minute":
        raw_table = info["table_map"].get(raw_table, raw_table)
    return f"{dataset_key}_{raw_table}"

def collect_metric_across_strains(
    dataset_tables: Dict[str, pd.DataFrame],
    metric: str,
    *,
    unit: str = "worm",
    table: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    prefer_details: bool = True,
    extra_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Stack one metric across strains into a long DataFrame for plotting/stats.

    Returns columns: ``strain``, ``id``, ``metric`` value under ``metric``,
    plus any ``extra_cols`` found (e.g. ``video_number``, ``recording_date``).

    Parameters
    ----------
    unit : "worm" | "minute"
        Worm-level rows vs continuous-minute rows.
    table : optional table suffix when details tables are missing / lack the
        column (e.g. ``active_per_active_duration``).
    """
    unit = str(unit).strip().lower()
    if unit not in _STRAIN_TABLE_ALIASES:
        raise ValueError("unit must be 'worm' or 'minute'")
    info = _STRAIN_TABLE_ALIASES[unit]
    id_col = info["id_col"]
    if datasets is None:
        datasets = [
            k.removesuffix(f"_{info['details_suffix']}")
            for k in dataset_tables
            if k.endswith(f"_{info['details_suffix']}")
        ]
        if not datasets:
            datasets = [
                "N2_worms",
                "VM487_worms",
                "HML1641_worms",
                "HML1630_worms",
            ]

    want_extra = list(extra_cols) if extra_cols is not None else [
        "recording_date",
        "video_number",
        "source_worm_id",
        "backfilled",
    ]
    rows: List[pd.DataFrame] = []
    missing: List[str] = []
    for ds in datasets:
        key = resolve_strain_table_key(
            ds,
            unit=unit,
            table=table,
            prefer_details=prefer_details,
            dataset_tables=dataset_tables,
            metric=metric,
        )
        if key not in dataset_tables:
            missing.append(key)
            continue
        df = dataset_tables[key]
        if metric not in df.columns:
            raise KeyError(
                f"Column {metric!r} not in {key}. Available: {list(df.columns)}"
            )
        cols = [c for c in [id_col, metric, *want_extra] if c in df.columns]
        piece = df[cols].copy()
        piece = piece.rename(columns={id_col: "id"})
        piece.insert(0, "strain", _strain_label(ds))
        piece.insert(1, "unit", unit)
        piece.insert(2, "source_table", key)
        rows.append(piece)

    if missing:
        raise KeyError(
            "Missing tables: "
            + ", ".join(missing)
            + f". Have: {sorted(k for k in dataset_tables if any(d in k for d in datasets))}"
        )
    if not rows:
        raise ValueError("No rows collected — check datasets / unit / table.")

    out = pd.concat(rows, ignore_index=True)
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    return out

def pvalue_to_stars(p: float) -> str:
    """Map a p-value to the conventional star string (ns / * / ** / *** / ****)."""
    if not np.isfinite(p):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"

# Canonical name → aliases users may type in the notebook
STATS_TEST_ALIASES: Dict[str, str] = {
    # non-parametric
    "mannwhitney": "mannwhitney",
    "mannwhitneyu": "mannwhitney",
    "mw": "mannwhitney",
    "mwu": "mannwhitney",
    "nonparametric": "mannwhitney",
    "non-parametric": "mannwhitney",
    "wilcoxon_ranksum": "mannwhitney",
    "ranksum": "mannwhitney",
    "brunner_munzel": "brunner_munzel",
    "bm": "brunner_munzel",
    "ks": "ks",
    "kolmogorov_smirnov": "ks",
    "kstest": "ks",
    # parametric
    "welch": "welch",
    "welch_ttest": "welch",
    "ttest": "welch",
    "parametric": "welch",
    "student": "student",
    "students": "student",
    "student_ttest": "student",
    "ttest_equal_var": "student",
    "pooled": "student",
}

STATS_TEST_LABELS: Dict[str, str] = {
    "mannwhitney": "Mann–Whitney U",
    "brunner_munzel": "Brunner–Munzel",
    "ks": "Kolmogorov–Smirnov",
    "welch": "Welch t-test",
    "student": "Student t-test",
}

# Canonical p-adjust name → aliases users may type in the notebook
P_ADJUST_ALIASES: Dict[str, str] = {
    "none": "none",
    "": "none",
    "raw": "none",
    "unadjusted": "none",
    "holm": "holm",
    "holm_bonferroni": "holm",
    "bonferroni": "bonferroni",
    "bonf": "bonferroni",
    "bh": "benjamini_hochberg",
    "fdr": "benjamini_hochberg",
    "fdr_bh": "benjamini_hochberg",
    "benjamini_hochberg": "benjamini_hochberg",
    "benjaminihochberg": "benjamini_hochberg",
}

P_ADJUST_LABELS: Dict[str, str] = {
    "none": "none",
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "benjamini_hochberg": "Benjamini–Hochberg",
}

def resolve_stats_test(name: str) -> str:
    """Normalize a stats-test name to a canonical key."""
    key = str(name).strip().lower().replace(" ", "_")
    if key not in STATS_TEST_ALIASES:
        raise ValueError(
            f"Unknown stats_test {name!r}. Choose one of: "
            + ", ".join(sorted(set(STATS_TEST_ALIASES.values())))
            + f" (aliases ok). Got aliases map keys like {sorted(STATS_TEST_ALIASES)[:8]}…"
        )
    return STATS_TEST_ALIASES[key]

def resolve_p_adjust(name: str) -> str:
    """Normalize a p-value adjustment name to a canonical key."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in P_ADJUST_ALIASES:
        raise ValueError(
            f"Unknown p_adjust {name!r}. Choose one of: "
            + ", ".join(sorted(set(P_ADJUST_ALIASES.values())))
            + " (aliases: holm, bonferroni, bh / benjamini_hochberg / fdr, none)."
        )
    return P_ADJUST_ALIASES[key]

def _pair_arrays(xa: np.ndarray, xb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    return xa[np.isfinite(xa)], xb[np.isfinite(xb)]

def _run_pairwise_test(
    xa: np.ndarray,
    xb: np.ndarray,
    *,
    test: str,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Run one two-sample test. Returns statistic, p, plus mean/median summaries.
    """
    xa, xb = _pair_arrays(xa, xb)
    out: Dict[str, float] = {
        "n_a": float(len(xa)),
        "n_b": float(len(xb)),
        "mean_a": float(np.mean(xa)) if len(xa) else float("nan"),
        "mean_b": float(np.mean(xb)) if len(xb) else float("nan"),
        "median_a": float(np.median(xa)) if len(xa) else float("nan"),
        "median_b": float(np.median(xb)) if len(xb) else float("nan"),
        "statistic": float("nan"),
        "p": float("nan"),
    }
    if len(xa) == 0 or len(xb) == 0:
        return out

    test = resolve_stats_test(test)
    try:
        if test == "mannwhitney":
            try:
                stat, p = stats.mannwhitneyu(
                    xa, xb, alternative=alternative, method="auto"
                )
            except TypeError:
                stat, p = stats.mannwhitneyu(xa, xb, alternative=alternative)
        elif test == "brunner_munzel":
            if not hasattr(stats, "brunner_munzel"):
                raise ValueError(
                    "stats_test='brunner_munzel' needs SciPy with "
                    "scipy.stats.brunner_munzel (SciPy ≥ 1.7). "
                    "Use 'mannwhitney', 'welch', 'student', or 'ks' instead."
                )
            # SciPy ≥1.7; alternative added later
            try:
                stat, p = stats.brunner_munzel(xa, xb, alternative=alternative)
            except TypeError:
                stat, p = stats.brunner_munzel(xa, xb)
        elif test == "ks":
            # two-sided KS; alternative only on newer SciPy
            try:
                res = stats.ks_2samp(xa, xb, alternative=alternative, method="auto")
                stat, p = res.statistic, res.pvalue
            except TypeError:
                stat, p = stats.ks_2samp(xa, xb)
        elif test == "welch":
            stat, p = stats.ttest_ind(
                xa, xb, equal_var=False, alternative=alternative, nan_policy="omit"
            )
        elif test == "student":
            stat, p = stats.ttest_ind(
                xa, xb, equal_var=True, alternative=alternative, nan_policy="omit"
            )
        else:  # pragma: no cover
            raise ValueError(f"Unhandled stats_test {test!r}")
        out["statistic"] = float(stat)
        out["p"] = float(p)
    except TypeError:
        # SciPy without alternative= on t-test
        if test in ("welch", "student"):
            stat, p = stats.ttest_ind(
                xa, xb, equal_var=(test == "student"), nan_policy="omit"
            )
            out["statistic"] = float(stat)
            out["p"] = float(p)
        else:
            raise
    except ValueError:
        pass
    return out

def _apply_p_adjust(pvals: np.ndarray, p_adjust: str) -> np.ndarray:
    canonical = resolve_p_adjust(p_adjust)
    pvals = np.asarray(pvals, dtype=float)
    if canonical == "none":
        return pvals.copy()
    if canonical == "bonferroni":
        padj = np.minimum(pvals * np.isfinite(pvals).sum(), 1.0)
        return np.where(np.isfinite(pvals), padj, np.nan)
    if canonical == "holm":
        return _holm_adjust(pvals)
    if canonical == "benjamini_hochberg":
        return _bh_adjust(pvals)
    raise ValueError("p_adjust must be 'none', 'holm', 'bonferroni', or 'benjamini_hochberg'")

def pairwise_group_tests(
    df: pd.DataFrame,
    metric: str,
    group: str = "strain",
    *,
    test: str = "mannwhitney",
    order: Optional[Sequence[str]] = None,
    alternative: str = "two-sided",
    p_adjust: str = "none",
) -> pd.DataFrame:
    """
    Pairwise two-sample tests across every pair of ``group`` levels.

    ``test`` options (aliases in parentheses):
      - ``mannwhitney`` (nonparametric, mw, ranksum)
      - ``brunner_munzel`` (bm)
      - ``ks`` (kolmogorov_smirnov)
      - ``welch`` (parametric, ttest) — Welch t-test (unequal variance)
      - ``student`` (ttest_equal_var, pooled) — Student t-test

    ``p_adjust``: ``none`` | ``holm`` | ``bonferroni`` | ``benjamini_hochberg``
    (aliases: ``bh``, ``fdr``, ``fdr_bh``). Stars use adjusted p.
    """
    canonical = resolve_stats_test(test)
    work, groups = _numeric_groups(df, metric, group)
    labels = list(order) if order is not None else sorted(groups)
    labels = [str(g) for g in labels if str(g) in groups]
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            res = _run_pairwise_test(
                groups[a], groups[b], test=canonical, alternative=alternative
            )
            rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "test": canonical,
                    "test_label": STATS_TEST_LABELS.get(canonical, canonical),
                    "level_a": a,
                    "level_b": b,
                    "i_a": labels.index(a),
                    "i_b": labels.index(b),
                    "n_a": int(res["n_a"]),
                    "n_b": int(res["n_b"]),
                    "mean_a": res["mean_a"],
                    "mean_b": res["mean_b"],
                    "median_a": res["median_a"],
                    "median_b": res["median_b"],
                    "statistic": res["statistic"],
                    # Back-compat aliases used by older notebook displays
                    "U": res["statistic"] if canonical == "mannwhitney" else float("nan"),
                    "t": res["statistic"]
                    if canonical in ("welch", "student")
                    else float("nan"),
                    "p": res["p"],
                    "alternative": alternative,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out["p_adj"] = []
        out["stars"] = []
        out["significant"] = []
        out["p_adjust"] = []
        return out

    out["p_adj"] = _apply_p_adjust(out["p"].to_numpy(dtype=float), p_adjust)
    out["stars"] = [pvalue_to_stars(p) for p in out["p_adj"]]
    out["significant"] = out["stars"].ne("ns")
    out["p_adjust"] = resolve_p_adjust(p_adjust)
    return out

def mannwhitney_pairwise(
    df: pd.DataFrame,
    metric: str,
    group: str = "strain",
    *,
    order: Optional[Sequence[str]] = None,
    alternative: str = "two-sided",
    p_adjust: str = "none",
) -> pd.DataFrame:
    """Mann–Whitney U for every pair of ``group`` levels (wrapper)."""
    return pairwise_group_tests(
        df,
        metric,
        group,
        test="mannwhitney",
        order=order,
        alternative=alternative,
        p_adjust=p_adjust,
    )

def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    out = np.full(m, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="mergesort")]
    sorted_p = p[order]
    m_f = len(sorted_p)
    holm = np.empty(m_f, dtype=float)
    running = 0.0
    for i in range(m_f):
        running = max(running, (m_f - i) * float(sorted_p[i]))
        holm[i] = min(running, 1.0)
    out[order] = holm
    return out

def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    out = np.full(m, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.flatnonzero(finite)
    order = idx[np.argsort(p[idx], kind="mergesort")]
    sorted_p = p[order]
    m_f = len(sorted_p)
    adj = np.empty(m_f, dtype=float)
    running = 1.0
    for i in range(m_f - 1, -1, -1):
        running = min(running, (m_f / (i + 1)) * float(sorted_p[i]))
        adj[i] = min(running, 1.0)
    out[order] = adj
    return out

def _annotate_pairwise_stars(
    ax,
    pairs: pd.DataFrame,
    *,
    order: Sequence[str],
    only_significant: bool = True,
    fontsize: int = 11,
    line_color: str = "#222222",
    tip: float = 0.02,
    stack_pad: float = 0.08,
):
    """
    Draw significance brackets + star labels for pairwise comparisons.

    ``pairs`` needs columns ``level_a``, ``level_b``, ``stars`` (and preferably
    ``i_a`` / ``i_b``). Y positions stack above the current axis data range.
    """
    if pairs is None or len(pairs) == 0:
        return
    work = pairs.copy()
    if only_significant:
        work = work.loc[work["stars"].ne("ns")].copy()
    if work.empty:
        return

    # Prefer shorter brackets first so they sit lower.
    if "i_a" in work.columns and "i_b" in work.columns:
        work = work.assign(span=(work["i_b"] - work["i_a"]).abs())
        work = work.sort_values(["span", "i_a", "i_b"])
    else:
        idx = {s: i for i, s in enumerate(order)}
        work = work.assign(
            i_a=work["level_a"].map(idx),
            i_b=work["level_b"].map(idx),
        )
        work = work.dropna(subset=["i_a", "i_b"])
        work["span"] = (work["i_b"] - work["i_a"]).abs()
        work = work.sort_values(["span", "i_a", "i_b"])

    y0, y1 = ax.get_ylim()
    data_span = (y1 - y0) if y1 > y0 else 1.0
    tip_h = tip * data_span
    layer_h = stack_pad * data_span
    base = y1 + 0.04 * data_span

    for layer, (_, row) in enumerate(work.iterrows()):
        x1 = float(row["i_a"])
        x2 = float(row["i_b"])
        if x2 < x1:
            x1, x2 = x2, x1
        y = base + layer * layer_h
        ax.plot([x1, x1, x2, x2], [y, y + tip_h, y + tip_h, y], color=line_color, lw=1.0)
        ax.text(
            0.5 * (x1 + x2),
            y + tip_h,
            str(row["stars"]),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=line_color,
            clip_on=False,
        )

    # Room for brackets above the plot
    ax.set_ylim(y0, base + max(len(work), 1) * layer_h + 2 * tip_h)

def plot_metric_across_strains(
    dataset_tables: Dict[str, pd.DataFrame],
    metric: str,
    *,
    unit: str = "worm",
    table: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    kind: str = "bar",
    error: str = "sem",
    strain_order: Optional[Sequence[str]] = None,
    stats: bool = False,
    stats_test: str = "mannwhitney",
    p_adjust: str = "none",
    annotate_ns: bool = False,
    ax=None,
    figsize=(7.5, 4.5),
    title: Optional[str] = None,
):
    """
    Bar/box/violin of ``metric`` across strains (N2, VM487, …).

    ``unit="worm"`` uses per-worm rows; ``unit="minute"`` uses continuous
    minutes. Prefers ``*_details`` / ``*_minute_details`` when available.

    When ``stats=True``, run pairwise tests (``stats_test``) and draw star
    brackets for significant pairs (* p<0.05, ** <0.01, *** <0.001, **** <0.0001).

    ``stats_test``: ``mannwhitney`` (nonparametric), ``welch`` / ``parametric``,
    ``student``, ``brunner_munzel``, ``ks`` — see ``pairwise_group_tests``.
    ``p_adjust``: ``none`` | ``holm`` | ``bonferroni`` | ``benjamini_hochberg``
    (aliases ``bh``, ``fdr``).
    """
    long_df = collect_metric_across_strains(
        dataset_tables,
        metric,
        unit=unit,
        table=table,
        datasets=datasets,
    )
    work = long_df.dropna(subset=[metric]).copy()
    if work.empty:
        raise ValueError(f"No finite values for {metric!r} ({unit})")

    DEFAULT_STRAIN_ORDER = ["N2", "VM487", "HML1641", "HML1630"]
    if strain_order is None:
        strain_order = DEFAULT_STRAIN_ORDER

    result = plot_metric_by_group(
        work,
        metric,
        "strain",
        kind=kind,
        error=error,
        order=strain_order,
        ax=ax,
        figsize=figsize,
    )
    order = result["order"]
    n_by = work.groupby("strain", observed=False)[metric].count().reindex(order)
    label = title or f"{metric}  ({unit})"
    result["ax"].set_title(label)
    # Annotate n under each tick
    ymin, ymax = result["ax"].get_ylim()
    for i, s in enumerate(order):
        n = int(n_by.get(s, 0) or 0)
        result["ax"].text(
            i,
            ymin + 0.02 * (ymax - ymin),
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444",
        )
    result["long_df"] = long_df
    result["order"] = order
    result["summary"] = (
        work.groupby("strain", observed=False)[metric]
        .agg(mean="mean", sd="std", n="count", median="median")
        .reindex(order)
    )
    result["summary"]["sem"] = result["summary"]["sd"] / np.sqrt(
        result["summary"]["n"].clip(lower=1)
    )

    pairs = pd.DataFrame()
    if stats:
        pairs = pairwise_group_tests(
            work,
            metric,
            "strain",
            test=stats_test,
            order=order,
            p_adjust=p_adjust,
        )
        _annotate_pairwise_stars(
            result["ax"],
            pairs,
            order=order,
            only_significant=not annotate_ns,
        )
        method = pairs["test_label"].iloc[0] if len(pairs) else resolve_stats_test(stats_test)
        adj = resolve_p_adjust(p_adjust)
        if adj != "none":
            method = f"{method}, {P_ADJUST_LABELS.get(adj, adj)}-adjusted"
        result["ax"].set_title(f"{label}\n{method}")
    result["stats"] = pairs
    result["stats_test"] = resolve_stats_test(stats_test) if stats else None
    return result

def plot_metrics_across_strains(
    dataset_tables: Dict[str, pd.DataFrame],
    metrics: Sequence[str],
    *,
    unit: str = "worm",
    table: Optional[str] = None,
    datasets: Optional[Sequence[str]] = None,
    kind: str = "bar",
    error: str = "sem",
    stats: bool = False,
    stats_test: str = "mannwhitney",
    p_adjust: str = "none",
    annotate_ns: bool = False,
    figsize_per: Tuple[float, float] = (5.5, 4.0),
):
    """Panel of across-strain plots for several metrics (same unit)."""
    metrics = list(metrics)
    if not metrics:
        raise ValueError("metrics must be non-empty")
    n = len(metrics)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_per[0] * n, figsize_per[1]),
        squeeze=False,
    )
    results = []
    for ax, metric in zip(axes[0], metrics):
        res = plot_metric_across_strains(
            dataset_tables,
            metric,
            unit=unit,
            table=table,
            datasets=datasets,
            kind=kind,
            error=error,
            stats=stats,
            stats_test=stats_test,
            p_adjust=p_adjust,
            annotate_ns=annotate_ns,
            ax=ax,
        )
        results.append(res)
    fig.tight_layout()
    return {"fig": fig, "axes": axes[0], "results": results}

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_by_group(
    df: pd.DataFrame,
    metric: str,
    group: str,
    *,
    kind: str = "bar",
    error: str = "sem",
    order: Optional[Sequence[str]] = None,
    ax=None,
    figsize=(7, 4.5),
):
    """
    Distribution of a metric across categories (video_number, recording_date, …).

    kind: 'bar' (mean ± error + strip) | 'box' | 'violin'
    error: 'sem' | 'sd' | 'ci95' (bar only)
    order: optional category order; default sorts by (len, name)
    """
    work = df[[group, metric]].dropna().copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    work[group] = work[group].astype(str)
    if work.empty:
        raise ValueError("No data after dropping NaNs")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if order is None:
        order = sorted(work[group].unique(), key=lambda s: (len(s), s))
    else:
        order = [str(s) for s in order if str(s) in set(work[group])]
        extras = sorted(set(work[group]) - set(order), key=lambda s: (len(s), s))
        order = list(order) + extras
    if kind == "bar":
        summary = (
            work.groupby(group)[metric]
            .agg(mean="mean", sd="std", n="count")
            .reindex(order)
        )
        summary["sem"] = summary["sd"] / np.sqrt(summary["n"].clip(lower=1))
        summary["ci95"] = 1.96 * summary["sem"]
        err = summary[error] if error in summary.columns else summary["sem"]
        ax.bar(range(len(summary)), summary["mean"], color="#4a6fa5", zorder=2)
        ax.errorbar(
            range(len(summary)),
            summary["mean"],
            yerr=err.fillna(0),
            fmt="none",
            ecolor="black",
            capsize=4,
            zorder=3,
        )
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary.index.astype(str), rotation=45, ha="right")
        rng = np.random.default_rng(0)
        for i, g in enumerate(order):
            vals = work.loc[work[group] == g, metric].to_numpy()
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                c="#222",
                s=22,
                alpha=0.7,
                zorder=4,
            )
    elif kind == "box":
        sns.boxplot(data=work, x=group, y=metric, order=order, ax=ax, color="#a8c0d8")
        sns.stripplot(
            data=work,
            x=group,
            y=metric,
            order=order,
            ax=ax,
            color="#222",
            size=5,
            alpha=0.65,
            jitter=0.2,
        )
    elif kind == "violin":
        sns.violinplot(
            data=work,
            x=group,
            y=metric,
            order=order,
            ax=ax,
            color="#a8c0d8",
            cut=0,
            inner="quartile",
        )
    else:
        raise ValueError("kind must be bar|box|violin")

    ax.set_xlabel(group)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by {group}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return {"fig": fig, "ax": ax, "order": order}

def _numeric_groups(df, metric, group):
    work = df[[group, metric]].dropna().copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    work[group] = work[group].astype(str)
    groups = {
        g: sub[metric].to_numpy(dtype=float)
        for g, sub in work.groupby(group)
        if len(sub) > 0
    }
    return work, groups

# ---------------------------------------------------------------------------
# Speed-trace gallery (one worm per strain)
# ---------------------------------------------------------------------------

def _pick_worm_id(
    worms: Dict[str, pd.DataFrame],
    preferred: Optional[str] = None,
) -> str:
    if preferred is not None and preferred in worms:
        return preferred
    if "W4" in worms:
        return "W4"
    # Prefer plain W# ids over W#_2 replicates when both exist.
    keys = sorted(worms.keys(), key=lambda w: (("_" in w), w))
    if not keys:
        raise ValueError("empty worm dict")
    return keys[0]

def plot_speed_one_worm_per_strain(
    strain_worms: Dict[str, Dict[str, pd.DataFrame]],
    *,
    worm_ids: Optional[Dict[str, str]] = None,
    strain_order: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = True,
    show_legend: bool = False,
    eps_dir: Optional[Union[str, Path]] = None,
    eps_prefix: str = "speed_one_per_strain",
    panel_figsize: Tuple[float, float] = (10.0, 2.8),
    x_unit: str = "seconds",
    xlim: Optional[Tuple[float, float]] = (0.0, 600.0),
    line_color: str = "#2c6e49",
    reverse_color: str = "#c45c5c",
    reversal_marker: str = "o",
    reversal_color: str = "#ffdd33",
    reversal_marker_size: float = 36,
    inactive_color: str = "#9aa0a6",
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Stack the section-2 speed panel (``plot_speed_trace``) for one worm per strain.

    X limits default to ``(0, 600)`` seconds so t=0 is the left edge and
    t=600 is the right edge (no extra x padding). Pass ``xlim=None`` to
    autoscale from the data. Y limits are independent when ``sharey=False``.
    Legends are off by default (no keys). Inactive speed uses ``inactive_color``
    from the setup-cell activity filter (no background shading). Reverse vs
    forward line color follows ``compute_metrics`` (setup-cell dwell,
    activity): reverse-state runs after kept reversals. Color flips are
    interpolated at y=0. Style knobs: ``line_color`` (forward),
    ``reverse_color``, ``inactive_color``, ``reversal_marker``,
    ``reversal_color``, ``reversal_marker_size``.

    If ``eps_dir`` is set, writes:
      - ``{eps_prefix}_stack.eps`` — all strains stacked
      - ``{eps_prefix}_{strain}_{worm_id}.eps`` — one file per strain
    Individual files use the same x limits as the stack when ``sharex`` is True,
    and the same y limits when ``sharey`` is True.
    """
    from worm_analysis.metrics import compute_metrics
    from worm_analysis.plotting import plot_speed_trace

    # Embed TrueType fonts in EPS (editable text in Illustrator / Affinity).
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["pdf.fonttype"] = 42

    worm_ids = dict(worm_ids or {})
    if strain_order is None:
        strain_order = list(strain_worms.keys())
    strains = [s for s in strain_order if s in strain_worms and strain_worms[s]]
    if not strains:
        raise ValueError("No non-empty strains in strain_worms")

    # Compute once; reuse for stack + per-strain EPS.
    results = []
    rows: List[dict] = []
    for strain in strains:
        worms = strain_worms[strain]
        wid = _pick_worm_id(worms, worm_ids.get(strain))
        result = compute_metrics(worms[wid])
        results.append((strain, wid, result))
        rows.append(
            {
                "strain": strain,
                "worm_id": wid,
                "n_frames": len(worms[wid]),
            }
        )

    n = len(results)
    if figsize is None:
        figsize = (11.0, max(2.4 * n, 3.5))
    fig, axes = plt.subplots(
        n, 1, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=False
    )
    axes_flat = axes.ravel()

    speed_kw = dict(
        show_legend=show_legend,
        x_unit=x_unit,
        line_color=line_color,
        shade_activity=False,
        shade_gaps=False,
        reversal_color=reversal_color,
        reversal_marker=reversal_marker,
        reversal_marker_size=reversal_marker_size,
        reversal_vlines=False,
        color_reverse_bouts=True,
        reverse_color=reverse_color,
        inactive_color=inactive_color,
    )

    for ax, (strain, wid, result) in zip(axes_flat, results):
        m = result.metrics
        title = f"{strain}  {m.worm_id}"
        plot_speed_trace(
            result.valid,
            frames=result.frames,
            bouts=result.bouts,
            title=title,
            ax=ax,
            frames_per_minute=m.frames_per_minute,
            reversal_dwell_frames=(
                1
                if not getattr(m, "use_reversal_sustain_filter", True)
                else m.reversal_dwell_frames
            ),
            **speed_kw,
        )

    xlim_used = None
    ylim = None
    if xlim is not None:
        xlim_used = (float(xlim[0]), float(xlim[1]))
        for ax in axes_flat:
            ax.margins(x=0)
            ax.set_xlim(xlim_used)
    elif sharex:
        xlim_used = axes_flat[0].get_xlim()
        for ax in axes_flat[1:]:
            x0, x1 = ax.get_xlim()
            xlim_used = (min(xlim_used[0], x0), max(xlim_used[1], x1))
        for ax in axes_flat:
            ax.set_xlim(xlim_used)
    if sharey:
        ylim = axes_flat[0].get_ylim()
        for ax in axes_flat[1:]:
            y0, y1 = ax.get_ylim()
            ylim = (min(ylim[0], y0), max(ylim[1], y1))
        for ax in axes_flat:
            ax.set_ylim(ylim)
    else:
        for ax in axes_flat:
            ax.autoscale(axis="y")
        if xlim_used is not None:
            for ax in axes_flat:
                ax.set_xlim(xlim_used)

    for ax in axes_flat:
        if not show_legend:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    for ax in axes_flat[:-1]:
        ax.set_xlabel("")
    fig.tight_layout()
    if xlim_used is not None:
        for ax in axes_flat:
            ax.set_xlim(xlim_used)

    if eps_dir is not None:
        out = Path(eps_dir)
        out.mkdir(parents=True, exist_ok=True)
        stack_path = out / f"{eps_prefix}_stack.eps"
        fig.savefig(stack_path, format="eps", bbox_inches="tight")
        print(f"Wrote {stack_path}")

        for strain, wid, result in results:
            m = result.metrics
            fig_i, ax_i = plt.subplots(figsize=panel_figsize)
            plot_speed_trace(
                result.valid,
                frames=result.frames,
                    bouts=result.bouts,
                title=f"{strain}  {m.worm_id}",
                ax=ax_i,
                frames_per_minute=m.frames_per_minute,
                reversal_dwell_frames=(
                    1
                    if not getattr(m, "use_reversal_sustain_filter", True)
                    else m.reversal_dwell_frames
                ),
                **speed_kw,
            )
            if xlim_used is not None:
                ax_i.margins(x=0)
                ax_i.set_xlim(xlim_used)
            if ylim is not None:
                ax_i.set_ylim(ylim)
            if not show_legend:
                leg = ax_i.get_legend()
                if leg is not None:
                    leg.remove()
            fig_i.tight_layout()
            if xlim_used is not None:
                ax_i.set_xlim(xlim_used)
            safe_wid = str(wid).replace("/", "_")
            path_i = out / f"{eps_prefix}_{strain}_{safe_wid}.eps"
            fig_i.savefig(path_i, format="eps", bbox_inches="tight")
            plt.close(fig_i)
            print(f"Wrote {path_i}")

    return fig, pd.DataFrame(rows)