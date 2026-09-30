from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import paths_config as pc
from script.calc_wrf_rmse import (
    CANONICAL_VAR_NAMES,
    LAT_SIZE,
    LON_SIZE,
    TRUTH_TOTAL_VARS,
    TruthAccessor,
    _as_datetime,
    format_seconds,
    iter_dates,
    load_daily_h5,
    read_h5_array,
    spatial_mse,
)
from script.config import OUTPUT_ROOT, YEAR, TIME_STEPS, forecast_skill_slice

AERO_STEPS = 72
WRF_STEPS = 48
UPPER_IDX = tuple(range(25))
SURFACE_IDX = (35, 36, 37, 38)
AIR_SURFACE_IDX = UPPER_IDX + SURFACE_IDX
AERO_PRED = tuple(range(29))
AERO_TRUTH = AIR_SURFACE_IDX
WRF_RMSE_PRED = tuple(range(39))
WRF_RMSE_TRUTH = tuple(range(39))

@dataclass
class ModelSpec:
    name: str
    stub: str
    steps: int
    loader: str
    rmse_pred: Tuple[int, ...]
    rmse_truth: Tuple[int, ...]

def _find_dated_h5(root: Path, when, suffix: str = "") -> Optional[Path]:
    dt = _as_datetime(when)
    stem = f"{dt.day:02d}{suffix}.h5"
    for rel in (
        Path(f"{dt.year:04d}") / f"{dt.month:02d}" / stem,
        Path(f"{dt.month:02d}") / stem,
        Path(stem),
    ):
        cand = root / rel
        if cand.exists():
            return cand
    return None

def load_aero_forecast(root: Path, init_date: date) -> Optional[np.ndarray]:
    air_path = _find_dated_h5(root, init_date)
    sfc_path = _find_dated_h5(root, init_date, suffix="_surface")
    upper = read_h5_array(air_path) if air_path is not None else None
    surface = read_h5_array(sfc_path) if sfc_path is not None else None
    if upper is None or surface is None:
        return None
    if upper.ndim != 4 or surface.ndim != 4:
        return None
    upper = forecast_skill_slice(upper)
    surface = forecast_skill_slice(surface)
    if upper.shape[0] != surface.shape[0] or upper.shape[2:] != surface.shape[2:]:
        return None
    return np.concatenate([upper, surface], axis=1)

def load_wrf_forecast(root: Path, init_date: date) -> Optional[np.ndarray]:
    return load_daily_h5([root], init_date)

def validate_pred(pred: np.ndarray, steps: int, min_channels: int) -> bool:
    return (
        pred.ndim == 4
        and pred.shape[0] >= steps
        and pred.shape[1] >= min_channels
        and pred.shape[2] == LAT_SIZE
        and pred.shape[3] == LON_SIZE
    )

def valid_00z_init_dates() -> list[date]:
    air_wd = Path(pc.AIR_ROOT)
    if str(air_wd) not in sys.path:
        sys.path.insert(0, str(air_wd))
    import AERO_ODE_Dataset_v3 as aero_dataset

    ds = aero_dataset.ERA5XarrayDataset(
        str(pc.era5_root()),
        hrrr_root_dirs=[str(pc.hrrr_test_root())],
        predict_lead_time=TIME_STEPS,
    )
    out: list[date] = []
    for idx in ds.valid_indices:
        ts = ds.timeline[idx]["timestamp"]
        if ts.hour == 0:
            out.append(ts.date())
    return out


def compute_truth_mean_field(
    truth_root: Path,
    start_date: date,
    end_date: date,
    progress_every: int = 5,
) -> np.ndarray:
    sum_field = None
    total_hours = 0
    dates = list(iter_dates(start_date, end_date))
    start_ts = time.time()
    for idx, day in enumerate(dates, start=1):
        data = load_daily_h5([truth_root], day)
        if data is not None and data.ndim == 4 and data.shape[1] >= TRUTH_TOTAL_VARS:
            data = data[:, :TRUTH_TOTAL_VARS]
            if sum_field is None:
                sum_field = np.zeros((TRUTH_TOTAL_VARS, data.shape[2], data.shape[3]), dtype=np.float64)
            sum_field += data.sum(axis=0, dtype=np.float64)
            total_hours += int(data.shape[0])
        if idx % progress_every == 0 or idx == len(dates):
            print(
                f"[ACC_MEAN] {idx}/{len(dates)} hours={total_hours} "
                f"elapsed={format_seconds(time.time() - start_ts)}",
                flush=True,
            )
    if sum_field is None or total_hours == 0:
        raise RuntimeError("no usable truth for ACC climate mean")
    return sum_field / total_hours

def _acc_one(pred_step: np.ndarray, truth_step: np.ndarray, mean_sub: np.ndarray) -> np.ndarray:
    pred_anom = pred_step - mean_sub
    truth_anom = truth_step - mean_sub
    numerator = np.sum(pred_anom * truth_anom, axis=(1, 2))
    denominator = np.sqrt(
        np.sum(pred_anom * pred_anom, axis=(1, 2)) * np.sum(truth_anom * truth_anom, axis=(1, 2))
    )
    out = np.full(pred_step.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out

def evaluate_model(
    spec: ModelSpec,
    init_dates: Sequence[date],
    *,
    load_pred,
    truth_accessor: TruthAccessor,
    mean_field: np.ndarray,
    progress_every: int = 1,
) -> Dict[str, np.ndarray]:
    rmse_pi = list(spec.rmse_pred)
    rmse_ti = np.asarray(spec.rmse_truth, dtype=np.int64)
    min_ch = max(rmse_pi) + 1

    mse_sum = np.zeros((spec.steps, TRUTH_TOTAL_VARS), dtype=np.float64)
    rmse_count = np.zeros(spec.steps, dtype=np.int64)
    acc_sum = np.zeros((spec.steps, TRUTH_TOTAL_VARS), dtype=np.float64)
    acc_count = np.zeros((spec.steps, TRUTH_TOTAL_VARS), dtype=np.int64)
    mean_sub = mean_field[rmse_ti]

    stats = {
        "init_dates_total": 0,
        "forecast_missing_or_failed": 0,
        "forecast_invalid_shape": 0,
        "truth_missing": 0,
        "valid_pairs": 0,
    }
    dates = list(init_dates)
    start_ts = time.time()
    for idx, init_date in enumerate(dates, start=1):
        stats["init_dates_total"] += 1
        pred = load_pred(init_date)
        if pred is None:
            stats["forecast_missing_or_failed"] += 1
        elif not validate_pred(pred, spec.steps, min_ch):
            stats["forecast_invalid_shape"] += 1
            print(f"[WARN] {spec.name} bad shape: {init_date} {getattr(pred, 'shape', None)}", flush=True)
        else:
            init_dt = datetime(init_date.year, init_date.month, init_date.day)
            for step in range(spec.steps):
                truth = truth_accessor.get_truth_hour(init_dt + timedelta(hours=step + 1))
                if truth is None:
                    stats["truth_missing"] += 1
                    continue
                pred_m = pred[step, rmse_pi].astype(np.float64, copy=False)
                truth_m = truth[rmse_ti].astype(np.float64, copy=False)
                if np.all(np.isfinite(pred_m)) and np.all(np.isfinite(truth_m)):
                    mse_sum[step, rmse_ti] += spatial_mse(pred_m, truth_m)
                    rmse_count[step] += 1
                    acc_one = _acc_one(pred_m, truth_m, mean_sub)
                    finite = np.isfinite(acc_one)
                    acc_sum[step, rmse_ti[finite]] += acc_one[finite]
                    acc_count[step, rmse_ti[finite]] += 1
                stats["valid_pairs"] += 1

        if idx % progress_every == 0 or idx == len(dates):
            elapsed = time.time() - start_ts
            print(
                f"[{spec.stub.upper()}] {idx}/{len(dates)} ({idx / len(dates):.1%}) "
                f"valid_pairs={stats['valid_pairs']}, missing={stats['forecast_missing_or_failed']}, "
                f"truth_missing={stats['truth_missing']}, elapsed={format_seconds(elapsed)}",
                flush=True,
            )

    rmse = np.full((spec.steps, TRUTH_TOTAL_VARS), np.nan, dtype=np.float64)
    has = rmse_count > 0
    if np.any(has):
        rmse[np.ix_(has, rmse_ti)] = np.sqrt(mse_sum[np.ix_(has, rmse_ti)] / rmse_count[has, None])
    acc = np.full((spec.steps, TRUTH_TOTAL_VARS), np.nan, dtype=np.float64)
    has_acc = acc_count > 0
    acc[has_acc] = acc_sum[has_acc] / acc_count[has_acc]
    stats["elapsed_sec"] = int(time.time() - start_ts)
    return {
        "rmse": rmse,
        "acc": acc,
        "rmse_count": rmse_count,
        "acc_count": acc_count,
        "stats": stats,
    }

def print_lead_table(title: str, arr: np.ndarray, leads: Sequence[int] = (6, 12, 24, 48)) -> None:
    print(f"\n{title}")
    header = f"{'Variable':<8}" + "".join(f" | +{lead:>3}h" for lead in leads)
    print(header)
    print("-" * len(header))
    for idx, name in enumerate(CANONICAL_VAR_NAMES):
        if not np.any(np.isfinite(arr[:, idx] if arr.ndim == 2 else arr[:, :, idx])):
            continue
        row = f"{name:<8}"
        for lead in leads:
            step = lead - 1
            if arr.ndim == 2:
                val = arr[step, idx] if 0 <= step < arr.shape[0] else np.nan
            else:
                val = np.nan
            row += f" | {val:8.4f}" if np.isfinite(val) else f" | {'nan':>8}"
        print(row)

def _load_cached_scores(
    output_dir: Path,
    spec: ModelSpec,
    start_date: date,
    end_date: date,
    n_00z: int,
) -> Optional[Dict[str, np.ndarray]]:
    rmse_path = output_dir / f"{spec.stub}_rmse.npy"
    acc_path = output_dir / f"{spec.stub}_acc.npy"
    if not rmse_path.exists() or not acc_path.exists():
        missing = [p.name for p in (rmse_path, acc_path) if not p.exists()]
        print(
            f"[INFO] {spec.name} missing {', '.join(missing)}; recomputing",
            flush=True,
        )
        return None
    stats: Dict = {}
    stats_path = output_dir / f"{spec.stub}_stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except json.JSONDecodeError:
            stats = {}
    cached_start = stats.get("start")
    cached_end = stats.get("end")
    want_start, want_end = start_date.isoformat(), end_date.isoformat()
    if cached_start != want_start or cached_end != want_end:
        print(
            f"[INFO] {spec.name} cache period {cached_start}~{cached_end} "
            f"!= {want_start}~{want_end}; recomputing",
            flush=True,
        )
        return None
    if int(stats.get("init_dates_total", -1)) != n_00z:
        print(
            f"[INFO] {spec.name} cache inits {stats.get('init_dates_total')} "
            f"!= {n_00z}; recomputing",
            flush=True,
        )
        return None
    rmse = np.load(rmse_path)
    acc = np.load(acc_path)
    if (
        rmse.ndim != 2
        or acc.ndim != 2
        or rmse.shape[0] < spec.steps
        or acc.shape[0] < spec.steps
        or rmse.shape[1] < TRUTH_TOTAL_VARS
        or acc.shape[1] < TRUTH_TOTAL_VARS
    ):
        print(
            f"[WARN] {spec.name} bad cache rmse={rmse.shape} acc={acc.shape}, "
            f"need >= ({spec.steps}, {TRUTH_TOTAL_VARS}); recomputing",
            flush=True,
        )
        return None
    count_path = output_dir / f"{spec.stub}_rmse_count.npy"
    rmse_count = np.load(count_path) if count_path.exists() else None

    return {
        "rmse": np.asarray(rmse[: spec.steps, :TRUTH_TOTAL_VARS], dtype=np.float64),
        "acc": np.asarray(acc[: spec.steps, :TRUTH_TOTAL_VARS], dtype=np.float64),
        "rmse_count": rmse_count,
        "stats": stats,
    }

def evaluate_aero_and_wrf(
    start_date: date,
    end_date: date,
    *,
    aero_root: Optional[Path] = None,
    wrf_root: Optional[Path] = None,
    truth_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    progress_every: int = 1,
    recompute_climate: bool = False,
    recompute: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    aero_root = Path(aero_root or pc.forecast_output_root())
    wrf_root = Path(wrf_root or pc.hrrr_forecast_root())
    truth_root = Path(truth_root or pc.hrrr_test_root())
    output_dir = Path(output_dir or OUTPUT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    init_dates = valid_00z_init_dates()
    n_00z = len(init_dates)

    specs = (
        ModelSpec(
            "WRF", "wrf", WRF_STEPS, "wrf",
            WRF_RMSE_PRED, WRF_RMSE_TRUTH,
        ),
        ModelSpec(
            "AERO-ODE", "aero", AERO_STEPS, "aero",
            AERO_PRED, AERO_TRUTH,
        ),
    )
    results: Dict[str, Dict[str, np.ndarray]] = {}
    pending: list[ModelSpec] = []
    for spec in specs:
        cached = None if recompute else _load_cached_scores(
            output_dir, spec, start_date, end_date, n_00z,
        )
        if cached is not None:
            results[spec.stub] = cached
            print_lead_table(f"{spec.name} RMSE (cached)", cached["rmse"])
            print_lead_table(f"{spec.name} ACC (cached)", cached["acc"])
        else:
            pending.append(spec)

    if pending:
        mean_path = output_dir / "acc_mean_field.npz"
        mean_field = None
        if mean_path.exists() and not recompute_climate:
            packed = np.load(mean_path)
            if int(packed["start_ordinal"]) == start_date.toordinal() and int(packed["end_ordinal"]) == end_date.toordinal():
                mean_field = packed["mean_field"]

        if mean_field is None:
            print("[INFO] computing ACC climate mean", flush=True)
            mean_field = compute_truth_mean_field(truth_root, start_date, end_date, progress_every)
            np.savez(
                mean_path,
                mean_field=mean_field,
                start_ordinal=np.array(start_date.toordinal()),
                end_ordinal=np.array(end_date.toordinal()),
            )

        accessor = TruthAccessor([truth_root])
        loaders = {
            "wrf": lambda d: load_wrf_forecast(wrf_root, d),
            "aero": lambda d: load_aero_forecast(aero_root, d),
        }
        for spec in pending:
            print("=" * 70, flush=True)
            print(
                f"eval {spec.name}: {n_00z} 00Z, "
                f"{init_dates[0]} ~ {init_dates[-1]}, steps={spec.steps}",
                flush=True,
            )
            out = evaluate_model(
                spec,
                init_dates,
                load_pred=loaders[spec.loader],
                truth_accessor=accessor,
                mean_field=mean_field,
                progress_every=progress_every,
            )
            results[spec.stub] = out
            out["stats"]["start"] = start_date.isoformat()
            out["stats"]["end"] = end_date.isoformat()
            np.save(output_dir / f"{spec.stub}_rmse.npy", out["rmse"].astype(np.float32))
            np.save(output_dir / f"{spec.stub}_acc.npy", out["acc"].astype(np.float32))
            np.save(output_dir / f"{spec.stub}_rmse_count.npy", out["rmse_count"])
            (output_dir / f"{spec.stub}_stats.json").write_text(
                json.dumps(out["stats"], indent=2, ensure_ascii=False) + "\n"
            )
            print_lead_table(f"{spec.name} RMSE", out["rmse"])
            print_lead_table(f"{spec.name} ACC", out["acc"])
            print(f"stats: {out['stats']}", flush=True)

    return results

def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()

def _default_range() -> Tuple[date, date]:
    start = date(YEAR, pc.DEMO_MONTH, 1)
    end = date(start.year, start.month + 1, 1) - timedelta(days=1) if start.month < 12 else date(start.year, 12, 31)
    return start, end

def main(argv: Optional[Sequence[str]] = None) -> Path:
    default_start, default_end = _default_range()
    parser = argparse.ArgumentParser(description="Compute AERO-ODE and WRF RMSE/ACC")
    parser.add_argument("--start", type=_parse_date, default=default_start)
    parser.add_argument("--end", type=_parse_date, default=default_end)
    parser.add_argument("--aero_root", type=Path, default=None)
    parser.add_argument("--wrf_root", type=Path, default=None)
    parser.add_argument("--truth_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument("--recompute_climate", action="store_true")
    parser.add_argument("--recompute", action="store_true", help="ignore existing RMSE/ACC files and recompute")
    args = parser.parse_args(argv)
    if args.end < args.start:
        raise ValueError(f"--end {args.end} before --start {args.start}")
    out = Path(args.output_dir or OUTPUT_ROOT)
    print("=" * 70)
    print("AERO-ODE / WRF  RMSE + ACC")
    print("=" * 70)
    print(f"AERO: {args.aero_root or pc.forecast_output_root()}")
    print