#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ==================== Config (edit as needed) ====================
# Run Draw_New/Visualization scripts
RUN_VISUALIZATION = True

# Used only when RUN_VISUALIZATION=True.
VISUALIZATION_SUBDIR_DATES: dict[str, list[str]] = {
    "AIR_rb": ["2024-02-01"],
    "AIR_wb": ["2024-06-01"],
    "Surface_rb": ["2024-08-15"],
    "Surface_wb": ["2024-04-01"],
}
VISUALIZATION_DATES = ["2024-01-15"]

# Significance heatmap
RUN_SIGNIFICANCE_HEATMAPS = True
RUN_SIGNIFICANCE_RMSE_DUAL = True  # RMSE dual-panel ED+WD (recommended)
RUN_SIGNIFICANCE_RMSE_SINGLE = False  # RMSE separate Rb/Wb (legacy)
RUN_SIGNIFICANCE_ETS_DUAL = True  # ETS dual-panel ED+WD (recommended)
RUN_SIGNIFICANCE_ETS_SINGLE = False  # ETS separate Rb/Wb (legacy)
SIGNIFICANCE_REGIONS = ("Rb", "Wb")
SIGNIFICANCE_STEPS = (48, 72)
# ETS events: high=P95, low=P5
SIGNIFICANCE_ETS_EVENTS = ("high", "low")

# Stop pipeline if a script fails
STOP_ON_ERROR = True

# Print commands only (dry run)
DRY_RUN = False

# If set, run scripts whose path contains any keyword
SCRIPT_FILTER: list[str] = []

# Subprocess env: Agg backend when headless
USE_AGG_BACKEND = True

# Map subplot title size (pt) via env for paths_config
MAP_SUBPLOT_TITLE_SIZE = 24
# Map subplot hspace (typical 0.15-0.45)
MAP_SUBPLOT_HSPACE = 0.25
# ============================================================

DRAW_NEW = Path(__file__).resolve().parent
SKIP_NAMES = frozenset({"paths_config.py", "run_all_drawings.py"})
CATEGORY_ORDER = ("RMSE", "ACC", "Ob_RMSE", "Extreme", "Visualization")

def _category(script: Path) -> str:
    rel = script.relative_to(DRAW_NEW)
    return rel.parts[0] if len(rel.parts) > 1 else ""


def _viz_subdir(script: Path) -> str:
    rel = script.relative_to(DRAW_NEW)
    if len(rel.parts) >= 3 and rel.parts[0] == "Visualization":
        return rel.parts[1]
    return ""


def viz_dates_for_script(script: Path, fallback_dates: list[str]) -> list[str]:
    subdir = _viz_subdir(script)
    if subdir and VISUALIZATION_SUBDIR_DATES.get(subdir):
        return VISUALIZATION_SUBDIR_DATES[subdir]
    return fallback_dates


def collect_scripts(include_visualization: bool) -> list[Path]:
    scripts: list[Path] = []
    for py in DRAW_NEW.rglob("*.py"):
        if py.name in SKIP_NAMES or not py.name.startswith("Draw_"):
            continue
        if _category(py) == "Visualization" and not include_visualization:
            continue
        if SCRIPT_FILTER and not any(k in py.as_posix() for k in SCRIPT_FILTER):
            continue
        scripts.append(py)

    def sort_key(p: Path) -> tuple:
        cat = _category(p)
        try:
            order = CATEGORY_ORDER.index(cat)
        except ValueError:
            order = len(CATEGORY_ORDER)
        return (order, p.as_posix())

    return sorted(scripts, key=sort_key)


def expand_script_jobs(script: Path, base_extra: list[str]) -> list[list[str]]:
    """Expand heatmap script to CLI variants (regions/leads/events)."""
    name = script.name
    if name == "Draw_Test_Significance_heatmap_dual_region.py":
        if RUN_SIGNIFICANCE_HEATMAPS and RUN_SIGNIFICANCE_RMSE_DUAL:
            return [["--steps", "both", *base_extra]]
        return []
    if name == "Draw_Test_Significance_heatmap.py":
        if RUN_SIGNIFICANCE_HEATMAPS and RUN_SIGNIFICANCE_RMSE_SINGLE:
            return [["--region", "both", "--steps", "both", *base_extra]]
        return []
    if name == "Draw_Test_Significance_ets_heatmap_dual_region.py":
        if RUN_SIGNIFICANCE_HEATMAPS and RUN_SIGNIFICANCE_ETS_DUAL:
            return [["--steps", "both", "--event", "both", *base_extra]]
        return []
    if name == "Draw_Test_Significance_ets_heatmap.py":
        if RUN_SIGNIFICANCE_HEATMAPS and RUN_SIGNIFICANCE_ETS_SINGLE:
            return [
                ["--region", "both", "--steps", "both", "--event", "both", *base_extra]
            ]
        return []
    return [base_extra]


def run_one(script: Path, extra_args: list[str], dry_run: bool) -> int:
    cmd = [sys.executable, str(script), *extra_args]
    cwd = script.parent
    env = os.environ.copy()
    if USE_AGG_BACKEND:
        env["MPLBACKEND"] = "Agg"
    env["DRAW_MAP_SUBPLOT_TITLE_SIZE"] = str(MAP_SUBPLOT_TITLE_SIZE)
    env["DRAW_MAP_SUBPLOT_HSPACE"] = str(MAP_SUBPLOT_HSPACE)

    rel = script.relative_to(DRAW_NEW)
    print(f"\n{'=' * 72}")
    print(f"[RUN] {rel}")
    print(f"      cwd: {cwd}")
    print(f"      cmd: {' '.join(cmd)}")
    if dry_run:
        return 0

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"[{status}] {rel}  ({elapsed:.1f}s)")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run Draw_New plotting scripts")
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable Visualization (override RUN_VISUALIZATION=False)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable Visualization (override RUN_VISUALIZATION=True)",
    )
    parser.add_argument(
        "--date",
        action="append",
        dest="viz_dates",
        metavar="YYYY-MM-DD",
        help="Visualization init date(s), repeatable",
    )
    parser.add_argument(
        "--no-significance",
        action="store_true",
        help="Do not expand RMSE/ETS significance heatmap jobs",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a script failure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only (dry run)",
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEYWORD",
        help="Run scripts matching path keywords",
    )
    parser.add_argument(
        "--map-title-size",
        type=int,
        metavar="PT",
        help="Map subplot title size (pt), overrides MAP_SUBPLOT_TITLE_SIZE",
    )
    parser.add_argument(
        "--map-hspace",
        type=float,
        metavar="RATIO",
        help="Map subplot hspace, overrides MAP_SUBPLOT_HSPACE",
    )
    args = parser.parse_args()

    include_viz = RUN_VISUALIZATION
    if args.viz:
        include_viz = True
    if args.no_viz:
        include_viz = False

    global RUN_SIGNIFICANCE_HEATMAPS
    if args.no_significance:
        RUN_SIGNIFICANCE_HEATMAPS = False

    stop_on_error = STOP_ON_ERROR and not args.continue_on_error
    dry_run = DRY_RUN or args.dry_run

    global SCRIPT_FILTER
    if args.filters:
        SCRIPT_FILTER = args.filters

    global MAP_SUBPLOT_TITLE_SIZE, MAP_SUBPLOT_HSPACE
    if args.map_title_size is not None:
        MAP_SUBPLOT_TITLE_SIZE = args.map_title_size
    if args.map_hspace is not None:
        MAP_SUBPLOT_HSPACE = args.map_hspace

    fallback_dates = args.viz_dates or VISUALIZATION_DATES
    scripts = collect_scripts(include_viz)

    if not scripts:
        print("No plotting scripts to run.")
        return 1

    print("Draw_New batch plotting")
    print(f"  Root: {DRAW_NEW}")
    print(f"  Include Visualization: {include_viz}")
    print(f"  Significance heatmap: {RUN_SIGNIFICANCE_HEATMAPS}")
    if RUN_SIGNIFICANCE_HEATMAPS:
        print(f"    RMSE dual ED+WD: {RUN_SIGNIFICANCE_RMSE_DUAL}")
        print(f"    RMSE single-region (legacy): {RUN_SIGNIFICANCE_RMSE_SINGLE}")
        print(f"    ETS dual ED+WD: {RUN_SIGNIFICANCE_ETS_DUAL}")
        print(f"    ETS single-region (legacy): {RUN_SIGNIFICANCE_ETS_SINGLE}")
        print(f"    Regions: {SIGNIFICANCE_REGIONS}")
        print(f"    Lead times: {SIGNIFICANCE_STEPS}")
        print(f"    ETS events: {SIGNIFICANCE_ETS_EVENTS}")
    print(f"  Script count: {len(scripts)}")
    if include_viz:
        if VISUALIZATION_SUBDIR_DATES:
            print(f"  Visualization subdir dates: {VISUALIZATION_SUBDIR_DATES}")
        print(f"  Visualization fallback dates: {fallback_dates}")
    if SCRIPT_FILTER:
        print(f"  Filter keywords: {SCRIPT_FILTER}")
    print(f"  Map title size: {MAP_SUBPLOT_TITLE_SIZE} pt")
    print(f"  Map hspace: {MAP_SUBPLOT_HSPACE}")
    if dry_run:
        print("  Mode: DRY RUN")

    failed: list[tuple[Path, int]] = []
    jobs: list[tuple[Path, list[str]]] = []

    for script in scripts:
        if _category(script) == "Visualization":
            dates = viz_dates_for_script(script, fallback_dates)
            for d in dates:
                for extra in expand_script_jobs(script, ["--date", d]):
                    jobs.append((script, extra))
        else:
            for extra in expand_script_jobs(script, []):
                jobs.append((script, extra))

    for script, extra in jobs:
        code = run_one(script, extra, dry_run)
        if code != 0:
            failed.append((script, code))
            if stop_on_error:
                print("\nAborted: previous command failed and STOP_ON_ERROR=True.")
                break

    print(f"\n{'=' * 72}")
    print(f"Done: {len(jobs)} jobs, {len(failed)} failed")
    if failed:
        for script, code in failed:
            print(f"  - {script.relative_to(DRAW_NEW)} (exit {code})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
