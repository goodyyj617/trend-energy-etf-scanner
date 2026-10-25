"""Run a CRSP study on another computer without leaving data there (DECISIONS 77).

    python scripts/crsp_remote.py --drive "G:\\내 드라이브\\CRSP" --study screens [--top 3000] [--workers N]

Steps: (1) a temporary folder is created; (2) the panel for --top is copied from the Drive folder
(lab_crsp/panel_top<N>), or built there from the raw CRSP_*.parquet files if missing; (3) the study
runs with its outputs in the temporary folder, syncing the checkpoint folder back to Drive after
every family so an interrupted run resumes from Drive; (4) results go to Drive
(lab_crsp/<study>_top<N>) and the temporary folder is deleted, panel included.

What stays on this computer: nothing from CRSP once the run ends (the yfinance cash series cache
under lab_data/yf and the repository do stay). While the run lasts, the panel sits in the
temporary folder and in memory. The raw files are never copied; they are read from Drive only
when a panel has to be built.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {"screens": "lab.screen_study", "families": "lab.crsp_study", "scores": "lab.score_study"}


def copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)


def sync_loop(src: Path, dst: Path, stop: threading.Event, every: int = 600) -> None:
    """Copy new/changed checkpoint files to Drive every `every` seconds while the study runs."""
    while not stop.wait(every):
        try:
            copytree(src, dst)
        except OSError as err:
            print(f"[sync] {err}", flush=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drive", required=True, help="Drive folder holding CRSP_*.parquet (e.g. G:\\내 드라이브\\CRSP)")
    ap.add_argument("--study", choices=list(STUDIES), default="screens")
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=0, help="0 = by RAM (about 8 GB per worker)")
    ap.add_argument("--extra", default="", help="extra arguments for the study, quoted")
    a = ap.parse_args(argv)
    drive = Path(a.drive)
    if not drive.exists():
        print(f"Drive folder not found: {drive}")
        return 1
    panel_src = drive / "lab_crsp" / f"panel_top{a.top}"
    out_dst = drive / "lab_crsp" / f"{a.study}_top{a.top}"
    tmp = Path(tempfile.mkdtemp(prefix="crsp_run_"))
    panel = tmp / "panel"
    out = tmp / "out"
    python = sys.executable
    try:
        has_sic = (panel_src / "info.csv").exists() and "sic" in (panel_src / "info.csv").read_text(encoding="utf-8").splitlines()[0]
        if (panel_src / "meta.json").exists() and not has_sic:
            print("[1/3] panel on Drive has no industry codes (SIC): rebuilding it", flush=True)
        if (panel_src / "meta.json").exists() and has_sic:
            print(f"[1/3] panel: copying {panel_src} -> temporary folder", flush=True)
            copytree(panel_src, panel)
        else:
            print(f"[1/3] panel: building top {a.top} from {drive} (10-30 min, read from Drive)", flush=True)
            subprocess.run([python, "-m", "lab.crsp", "--src", str(drive), "--out", str(panel), "--top", str(a.top),
                           "--years", "1991", "2024"], check=True, cwd=ROOT)
            copytree(panel, panel_src)  # the built panel is kept on Drive for the next run
        if out_dst.exists():
            print("[2/3] resuming: copying finished parts from Drive", flush=True)
            copytree(out_dst, out)
        out.mkdir(exist_ok=True)
        workers = a.workers
        if workers <= 0:
            try:
                import psutil
                ram = psutil.virtual_memory().total / 1e9
            except ImportError:
                ram = 16.0
            # memory per worker grows with the panel: about 16 float64 copies of one (days x symbols) table
            # at the peak (prices, signals, dollar volume, screens). Top 3,000 (11,900 symbols): ~13 GB.
            try:
                import json as _json
                n_sym = _json.loads((panel / "meta.json").read_text(encoding="utf-8"))["symbols"]
            except (OSError, KeyError, ValueError):
                n_sym = 4000
            per_worker = max(8.0, n_sym * 8600 * 8 * 16 / 1e9)
            workers = max(1, min((os.cpu_count() or 2) - 1, int((ram * 0.85 - 3) // per_worker)))
            print(f"memory {ram:.0f} GB, about {per_worker:.0f} GB per worker -> {workers} workers", flush=True)
        print(f"[3/3] study {a.study}: {workers} workers, checkpoints synced to {out_dst} every 10 min", flush=True)
        stop = threading.Event()
        th = threading.Thread(target=sync_loop, args=(out, out_dst, stop), daemon=True)
        th.start()
        cmd = [python, "-m", STUDIES[a.study], "--panel", str(panel), "--out", str(out), "--workers", str(workers)]
        if a.extra:
            cmd += a.extra.split()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
        code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        stop.set()
        th.join(timeout=5)
        copytree(out, out_dst)
        print(f"results: {out_dst} (exit code {code})", flush=True)
        return code
    finally:
        t = time.time()
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"temporary folder removed ({time.time() - t:.0f}s): {tmp}", flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
