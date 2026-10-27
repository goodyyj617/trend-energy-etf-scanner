"""Static figures (PNG) of aggregated study results, for the research memos and the experiment log.

Only summaries go into a figure (never raw prices). matplotlib is imported when a figure is drawn,
so the studies and the tests run without it; `available()` says whether it is installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLORS = ["#2b6cb0", "#dd6b20", "#2f855a", "#9b2c2c"]


def available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    korean = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR") if f in installed]
    plt.rcParams["font.family"] = korean[:1] + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def paired_bars(labels: list[str], series: dict[str, np.ndarray], title: str, xlabel: str, path: Path,
                pct: bool = True) -> Path:
    """Horizontal bars, one group per label and one bar per series (e.g. per sample)."""
    plt = _plt()
    n, k = len(labels), len(series)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.32 * n + 1.2)))
    h = 0.8 / max(k, 1)
    y = np.arange(n)
    for j, (name, vals) in enumerate(series.items()):
        v = np.asarray(vals, dtype=float) * (100 if pct else 1)
        ax.barh(y + (j - (k - 1) / 2) * h, v, height=h, label=name, color=COLORS[j % len(COLORS)])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def scatter_labeled(x: np.ndarray, y: np.ndarray, labels: list[str], title: str, xlabel: str, ylabel: str, path: Path,
                    note: str = "", pct: bool = True, label_top: int = 10) -> Path:
    """Scatter with a y = x line; the `label_top` points with the largest x + y are named."""
    plt = _plt()
    s = 100 if pct else 1
    x, y = np.asarray(x, dtype=float) * s, np.asarray(y, dtype=float) * s
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=22, color=COLORS[0], alpha=0.8)
    lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], color="#888", lw=0.8, ls="--")
    ax.axhline(0, color="#444", lw=0.6)
    ax.axvline(0, color="#444", lw=0.6)
    for i in np.argsort(-(x + y))[:label_top]:
        ax.annotate(labels[i], (x[i], y[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=11)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def heatmap(frames: dict[str, pd.DataFrame], title: str, path: Path, fmt: str = "{:+.1f}", scale: float = 100.0,
            center: float = 0.0, cbar: str = "") -> Path:
    """One panel per frame (rows x columns), diverging colours around `center`, values written in cells."""
    plt = _plt()
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(1, len(frames), figsize=(6.2 * len(frames), 0.42 * max(len(f) for f in frames.values()) + 2.2),
                             squeeze=False)
    vals = np.concatenate([f.to_numpy(dtype=float).ravel() * scale for f in frames.values()])
    vals = vals[np.isfinite(vals)]
    lo, hi = (min(vals.min(), center - 1e-9), max(vals.max(), center + 1e-9)) if len(vals) else (-1.0, 1.0)
    norm = TwoSlopeNorm(vcenter=center, vmin=lo, vmax=hi)
    im = None
    for k, (ax, (name, f)) in enumerate(zip(axes[0], frames.items())):
        v = f.to_numpy(dtype=float) * scale
        im = ax.imshow(v, cmap="RdBu", norm=norm, aspect="auto")
        ax.set_xticks(range(f.shape[1]), [str(c) for c in f.columns], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(f.shape[0]), [str(r) for r in f.index] if k == 0 else [], fontsize=8)
        for r in range(v.shape[0]):
            for c in range(v.shape[1]):
                if np.isfinite(v[r, c]):
                    ax.text(c, r, fmt.format(v[r, c]), ha="center", va="center", fontsize=7)
        ax.set_title(name)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label=cbar)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def cagr_mdd(groups: dict[str, pd.DataFrame], spy: dict[str, tuple[float, float]], title: str, path: Path) -> Path:
    """One panel per sample: CAGR vs MDD of every run, SPY as a star. groups[name] has columns cagr, mdd
    (and optionally 'group' to colour by); spy[name] = (cagr, mdd)."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)
    for ax, (name, g) in zip(axes[0], groups.items()):
        if "group" in g:
            for j, (k, sub) in enumerate(g.groupby("group", sort=False)):
                ax.scatter(sub["mdd"] * 100, sub["cagr"] * 100, s=8, alpha=0.5, label=str(k), color=COLORS[j % len(COLORS)])
            ax.legend(fontsize=8, loc="lower left")
        else:
            ax.scatter(g["mdd"] * 100, g["cagr"] * 100, s=8, alpha=0.5, color=COLORS[0])
        if name in spy:
            c, m = spy[name]
            ax.scatter([m * 100], [c * 100], marker="*", s=220, color="#d69e2e", edgecolor="k", zorder=5, label="SPY")
            ax.axhline(c * 100, color="#d69e2e", lw=0.7, ls="--")
            ax.axvline(m * 100, color="#d69e2e", lw=0.7, ls="--")
        ax.set_xlabel("MDD (%)")
        ax.set_ylabel("CAGR (%)")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    return _save(fig, path)
