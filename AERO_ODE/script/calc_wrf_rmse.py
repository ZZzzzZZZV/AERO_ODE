from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import h5py
import numpy as np

from common.paths import find_hrrr_daily_file
import paths_config as pc
from script.config import OUTPUT_ROOT, YEAR

LAT_SIZE = 440
LON_SIZE = 408
WRF_STEPS = 48
TRUTH_TOTAL_VARS = 39
_LEVELS = (300, 500, 700, 850, 925)
CANONICAL_VAR_NAMES = (
    [f"z{level}" for level in _LEVELS]
    + [f"t{level}" for level in _LEVELS]
    + [f"s{level}" for level in _LEVELS]
    + [f"u{level}" for level in _LEVELS]
    + [f"v{level}" for level in _LEVELS]
    + [f"cw{level}" for level in _LEVELS]
    + [f"ci{level}" for level in _LEVELS]
    + ["mslp", "u10", "v10", "t2m"]
)
assert len(CANONICAL_VAR_NAMES) == TRUTH_TOTAL_VARS
AIR_SLICE = slice(0, 25)
SURFACE_SLICE = slice(35, 39)

def iter_dates(start_dt: date, end_dt: date) -> Iterator[date]:
    cur = start_dt
    while cur <= end_dt:
        yield cur
        cur += timedelta(days=1)

def format_seconds(seconds: float) -> str:
    sec = int(max(0, seconds))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

def read_h5_array(path: Path) -> Optional[np.ndarray]:
    if path is None or not Path(path).exists():
        return None
    try:
        with h5py.File(path, "r") as handle:
            for key in ("data", "fields"):
                if key in handle and isinstance(handle[key], h5py.Dataset):
                    return handle[key][:]
            for key in handle.keys():
                if isinstance(handle[key], h5py.Dataset):
                    return handle[key][:]
    except Exception as exc:
        print(f"[WARN] failed to read H5: {path} -> {exc}", flush=True)
    return None

def _as_datetime(when) -> datetime:
    if isinstance(when, datetime):
        return when.replace(hour=0, minute=0, second=0, microsecond=0)
    if isinstance(when, date):
        return datetime(when.year, when.month, when.day)
    raise TypeError(f"unsupported date type: {type(when)}")

def load_daily_h5(roots: Sequence[Path], when) -> Optional[np.ndarray]:
    found = find_hrrr_daily_file(list(roots), _as_datetime(when))
    return read_h5_array(found) if found is not None else None

class TruthAccessor:

    def __init__(self, truth_roots: Sequence[Path], max_cache_days: int = 4):
        self.truth_roots = list(truth_roots)
        self.max_cache_days = max_cache_days
        self._cache: "OrderedDict[str, Optional[np.ndarray]]" = OrderedDict()

    def _cache_key(self, day: date) -> str:
        return f"{day.year:04d}-{day.month:02d}-{day.day:02d}"

    def _load_day(self, day: date) -> Optional[np.ndarray]:
        data = load_daily_h5(self.truth_roots, day)
        if data is None:
            return None
        if data.ndim != 4 or data.shape[1] < TRUTH_TOTAL_VARS:
            print(f"[WARN] bad truth shape: {day}, shape={getattr(data, 'shape', None)}", flush=True)
            return None
        return data

    def get_truth_hour(self, target_dt: datetime) -> Optional[np.ndarray]:
        day = target_dt.date()
        key = self._cache_key(day)
        if key not in self._cache:
            if len(self._cache) >= self.max_cache_days:
                self._cache.popitem(last=False)
            self._cache[key] = self._load_day(day)
        else:
            self._cache.move_to_end(key)

        day_data = self._cache[key]
        if day_data is None or target_dt.hour >= day_data.shape[0]:
            return None
        return day_data[target_dt.hour, :TRUTH_TOTAL_VARS]

def validate_wrf_forecast(pred: np.ndarray, steps: int = WRF_STEPS) -> bool:
    return (
        pred.ndim == 4
        and pred.shape[0] >= steps
        and pred.shape[1] >= TRUTH_TOTAL_VARS
        and pred.shape[2] == LAT_SIZE
        and pred.shape[3] == LON_SIZE
    )

def spatial_mse(pred_step: np.ndarray, truth_step: np.ndarray) -> np.ndarray:
    diff = pred_step - truth_step
    return np.mean(diff * diff, axis=(-2, -1))

def compute_wrf_sample_rmse(
    init_date: date,
    *,
    forecast_root: Optional[Path] = None,
    truth_root: Optional[Path] = None,
    steps: int = WRF_STEPS,
    truth_accessor: Optional[TruthAccessor] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    forecast_roots = [Path(forecast_root or pc.hrrr_forecast_root())]
    accessor = truth_accessor or TruthAccessor([Path(truth_root or pc.hrrr_test_root())])
    pred = load_daily_h5(forecast_roots, init_date)
    rmse = np.full((steps, TRUTH_TOTAL_VARS), np.nan, dtype=np.float64)
    stats = {"forecast_missing": 0, "forecast_invalid_shape": 0, "truth_missing": 0, "valid_steps": 0}

    if pred is None:
        stats["forecast_missing"] = 1
        return rmse, stats
    if not validate_wrf_forecast(pred, steps):
        stats["forecast_invalid_shape"] = 1
        return rmse, stats

    init_dt = datetime(init_date.year, init_date.month, init_date.day)
    for step in range(steps):
        truth = accessor.get_truth_hour(init_dt + timedelta(hours=step + 1))
        if truth is None:
            stats["truth_missing"] += 1
            continue
        pred_step = pred[step, :TRUTH_TOTAL_VARS].astype(np.float64, copy=False)
        truth_step = truth.astype(np.float64, copy=False)
        if not np.all(np.isfinite(pred_step)) or not np.all(np.isfinite(truth_step)):
            continue
        rmse[step] = np.sqrt(spatial_mse(pred_step, truth_step))
        stats["valid_steps"] += 1
    return rmse, stats

def evaluate_wrf_rmse(
    start_date: date,
    end_date: date,
    *,
    forecast_root: Optional[Path] = None,
    truth_root: Optional[Path] = None,
    steps: int = WRF_STEPS,
    progress_every: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    forecast_roots = [Path(forecast_root or pc.hrrr_forecast_root())]
    accessor = TruthAccessor([Path(truth_root or pc.hrrr_test_root())])
    mse_sum = np.zeros((steps, TRUTH_TOTAL_VARS), dtype=np.float64)
    count = np.zeros(steps, dtype=np.int64)
    stats = {
        "init_dates_total": 0,
        "forecast_missing_or_failed": 0,
        "forecast_invalid_shape": 0,
        "truth_missing": 0,
        "valid_pairs": 0,
    }

    all_dates = list(iter_dates(start_date, end_date))
    start_ts = time.time()
    for idx, init_date in enumerate(all_dates, start=1):
        stats["init_dates_total"] += 1
        pred = load_daily_h5(forecast_roots, init_date)
        if pred is None:
            stats["forecast_missing_or_failed"] += 1
        elif not validate_wrf_forecast(pred, steps):
            stats["forecast_invalid_shape"] += 1
            print(f"[WARN] bad WRF shape: {init_date}, shape={pred.shape}", flush=True)
        else:
            init_dt = datetime(init_date.year, init_date.month, init_date.day)
            for step in range(steps):
                truth = accessor.get_truth_hour(init_dt + timedelta(hours=step + 1))
                if truth is None:
                    stats["truth_missing"] += 1
                    continue
                pred_step = pred[step, :TRUTH_TOTAL_VARS].astype(np.float64, copy=False)
                truth_step = truth.astype(np.float64, copy=False)
                if not np.all(np.isfinite(pred_step)) or not np.all(np.isfinite(truth_step)):
                    continue
                mse_sum[step] += spatial_mse(pred_step, truth_step)
                count[step] += 1
                stats["valid_pairs"] += 1

        if progress_every > 0 and (idx % progress_every == 0 or idx == len(all_dates)):
            elapsed = time.time() - start_ts
            speed = idx / max(elapsed, 1e-6)
            eta = (len(all_dates) - idx) / max(speed, 1e-6)
            print(
                f"[WRF_RMSE] {idx}/{len(all_dates)} ({idx / len(all_dates):.1%}) "
                f"valid_pairs={stats['valid_pairs']}, "
                f"fcst_missing={stats['forecast_missing_or_failed']}, "
                f"truth_missing={stats['truth_missing']}, "
                f"elapsed={format_seconds(elapsed)}, eta={format_seconds(eta)}",
                flush=True,
            )

    rmse = np.full((steps, TRUTH_TOTAL_VARS), np.nan, dtype=np.float64)
    has = count > 0
    if np.any(has):
        rmse[has] = np.sqrt(mse_sum[has] / count[has, None])
    stats["rmse_steps_with_samples"] = int(np.sum(has))
    stats["elapsed_sec"] = int(time.time() - start_ts)
    return rmse, count, stats

def wrf_rmse_air_surface(rmse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return rmse[:, AIR_SLICE].T.copy(), rmse[:, SURFACE_SLICE].copy()

def print_rmse_summary(rmse: np.ndarray, count: np.ndarray, leads: Sequence[int] = (6, 12, 24, 48)) -> None:
    print("\nWRF RMSE summary")
    header = f"{'Variable':<8}" + "".join(f" | +{lead:>3}h" for lead in leads)
    print(header)
    print("-" * len(header))
    for idx, name in enumerate(CANONICAL_VAR_NAMES):
        row = f"{name:<8}"
        for lead in leads:
            step = lead - 1
            if 0 <= step < rmse.shape[0] and np.isfinite(rmse[step, idx]):
                row += f" | {rmse[step, idx]:8.4f}"
            else:
                row += f" | {'nan':>8}"
        print(row)
    print("-" * len(header))
    print("count   " + "".join(
        f" | {int(count[lead - 1]):8d}" if 0 <= lead - 1 < count.size else " |      nan"
        for lead in leads
    ))

def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()

def _default_range() -> Tuple[date, date]:
    start = date(YEAR, pc.DEMO_MONTH, 1)
    if start.month == 12:
        end = date(start.year, 12, 31)
    else:
        end = date(start.year, start.month + 1, 1) - timedelta(days=1)
    return start, end

def main(argv: Optional[Sequence[str]] = None) -> Path:
    default_start, default_end = _default_range()
    parser = argparse.ArgumentParser(description="Compute WRF / HRRR forecast RMSE")
    parser.add_argument("--start", type=_parse_date, default=default_start, help="inclusive start initialization date")
    parser.add_argument("--end", type=_parse_date, default=default_end, help="inclusive end initialization date")
    parser.add_argument("--date", type=_parse_date, default=None, help="spatial RMSE for a single initialization day")
    parser.add_argument("--forecast_root", type=Path, default=None, help="WRF forecast root")
    parser.add_argument("--truth_root", type=Path, default=None, help="HRRR analysis root")
    parser.add_argument("--steps", type=int, default=WRF_STEPS)
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output npy (default quick-start_output/wrf_rmse.npy)",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output or (OUTPUT_ROOT / "wrf_rmse.npy"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.date is not None:
        print(f"single-sample WRF RMSE: {args.date} 00 UTC", flush=True)
        rmse, stats = compute_wrf_sample_rmse(
            args.date,
            forecast_root=args.forecast_root,
            truth_root=args.truth_root,
            steps=args.steps,
        )
        count = np.isfinite(rmse).all(axis=1).astype(np.int64)
        np.save(output_path, rmse.astype(np.float32))
        print_rmse_summary(rmse, count)
        print(f"stats: {stats}")
        print(f"saved: {output_path}")
        return output_path

    if args.end < args.start:
        raise ValueError(f"--end {args.end} before --start {args.start}")

    print("=" * 70)
    print("WRF-ARW / HRRR forecast RMSE")
    print("=" * 70)
    print(f"forecast: {args.forecast_root or pc.hrrr_forecast_root()}")
    print(f"truth: {args.truth_root or pc.hrrr_test_root()}")
    print(f"inits: {args.start} ~ {args.end} 00 UTC")
    print(f"leads: 1~{args.steps} h")
    print("=" * 70)

    rmse, count, stats = evaluate_wrf_rmse(
        args.start,
        args.end,
        forecast_root=args.forecast_root,
        truth_root=args.truth_root,
        steps=args.steps,
        progress_every=args.progress_every,
    )
    np.save(output_path, rmse.astype(np.float32))
    np.save(output_path.with_name(output_path.stem + "_count.npy"), count)
    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print_rmse_summary(rmse, count)
    print(f"stats: {stats}")
    print(f"saved: {output_path}")
    print(f"saved: {stats_path}")
    return output_path

if __name__ == "__main__":
    main()
