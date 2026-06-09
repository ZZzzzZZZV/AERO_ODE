"""
Compute daily RMSE samples and ESS-corrected paired t-tests for 2024 Rb forecasts.

Outputs:
  - nwp_rmse, aero_rmse, neuralgcm_rmse: (valid_days, 48, 24)
  - p/t/significance matrices for AERO-ODE vs NWP and AERO-ODE vs NeuralGCM
  - effective sample size (ESS), lag-1 autocorrelation, and variance inflation matrices

Run on the server, for example:
  python rmse_ttest_wb.py --workers 8
"""

from __future__ import annotations

import argparse
import calendar
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - depends on server environment
    tqdm = None


VARIABLE_NAMES = [
    "z50",
    "z500",
    "z850",
    "z1000",
    "t50",
    "t500",
    "t850",
    "t1000",
    "s50",
    "s500",
    "s850",
    "s1000",
    "u50",
    "u500",
    "u850",
    "u1000",
    "v50",
    "v500",
    "v850",
    "v1000",
    "mslp",
    "u10",
    "v10",
    "t2m",
]

N_LEADS = 48
N_VARS = 24
N_AIR_VARS = 20
GRID_SHAPE = (440, 408)
TRUTH_SHAPE = (24, N_VARS, *GRID_SHAPE)
NWP_SHAPE = (N_LEADS, N_VARS, *GRID_SHAPE)
AERO_AIR_SHAPE = (72, N_AIR_VARS, *GRID_SHAPE)
AERO_SURFACE_SHAPE = (72, 4, *GRID_SHAPE)
NEURALGCM_SHAPE = (72, N_AIR_VARS, *GRID_SHAPE)


@dataclass(frozen=True)
class Config:
    nwp_root: Path
    aero_root: Path
    neuralgcm_root: Path
    truth_root: Path
    output: Path
    year: int


@dataclass(frozen=True)
class DayResult:
    init_date: str
    ok: bool
    nwp_rmse: np.ndarray | None = None
    aero_rmse: np.ndarray | None = None
    neuralgcm_rmse: np.ndarray | None = None
    skip_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RMSE samples and paired t-tests.")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--nwp-root",
        type=Path,
        default=Path("/nfs/samba/数据聚变/气象数据/hefang_09gpu/hrrr_nwp_west_h5"),
    )
    parser.add_argument(
        "--aero-root",
        type=Path,
        default=Path("/nfs/samba/数据聚变/气象数据/NeuralGCM_LAM_predicted_hrrr_wb"),
    )
    parser.add_argument(
        "--neuralgcm-root",
        type=Path,
        default=Path("/nfs/samba/数据聚变/气象数据/zhangjing/NeuralGCM_wb"),
    )
    parser.add_argument(
        "--truth-root",
        type=Path,
        default=Path("/nfs/samba/数据聚变/气象数据/hefang/upper_hrrr+sfc_hrrr+same_resolution_west/test_dataset/2024"),
    )
    parser.add_argument("--output", type=Path, default=Path("rmse_ttest_results_2024_wb.npz"))
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--serial", action="store_true", help="Run without multiprocessing.")
    parser.add_argument("--report-every", type=int, default=10)
    return parser.parse_args()


def forecast_path(root: Path, init_date: date, suffix: str = ".h5") -> Path:
    return root / f"{init_date.year:04d}" / f"{init_date.month:02d}" / f"{init_date.day:02d}{suffix}"


def truth_path(root: Path, valid_time: datetime) -> Path:
    return root / f"{valid_time.month:02d}" / f"{valid_time.day:02d}.h5"


def list_init_dates(year: int) -> list[date]:
    n_days = 366 if calendar.isleap(year) else 365
    first = date(year, 1, 1)
    return [first + timedelta(days=i) for i in range(n_days)]


def find_unique_dataset(h5_file: h5py.File, file_path: Path) -> h5py.Dataset:
    datasets: list[h5py.Dataset] = []

    def visitor(_: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append(obj)

    h5_file.visititems(visitor)
    if len(datasets) != 1:
        raise ValueError(f"{file_path}: expected exactly 1 dataset, found {len(datasets)}")
    return datasets[0]


def open_unique_dataset(file_path: Path, expected_shape: tuple[int, ...]) -> tuple[h5py.File, h5py.Dataset]:
    if not file_path.exists():
        raise FileNotFoundError(str(file_path))
    h5_file = h5py.File(file_path, "r")
    try:
        dataset = find_unique_dataset(h5_file, file_path)
        if tuple(dataset.shape) != expected_shape:
            raise ValueError(f"{file_path}: shape {dataset.shape}, expected {expected_shape}")
        return h5_file, dataset
    except Exception:
        h5_file.close()
        raise


def rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    diff = pred.astype(np.float32, copy=False) - truth.astype(np.float32, copy=False)
    return np.sqrt(np.nanmean(diff * diff, axis=(-2, -1), dtype=np.float64)).astype(np.float32)


def validate_truth_files(config: Config, init_date: date) -> dict[date, Path]:
    truth_files: dict[date, Path] = {}
    init_dt = datetime(init_date.year, init_date.month, init_date.day)
    for lead in range(1, N_LEADS + 1):
        valid_time = init_dt + timedelta(hours=lead)
        if valid_time.year != config.year:
            raise FileNotFoundError(f"truth_out_of_year:{valid_time:%Y-%m-%d %H:%M}")
        valid_day = valid_time.date()
        if valid_day in truth_files:
            continue
        path = truth_path(config.truth_root, valid_time)
        with h5py.File(path, "r") as h5_file:
            dataset = find_unique_dataset(h5_file, path)
            if tuple(dataset.shape) != TRUTH_SHAPE:
                raise ValueError(f"{path}: shape {dataset.shape}, expected {TRUTH_SHAPE}")
        truth_files[valid_day] = path
    return truth_files


def process_one_day(config: Config, init_date: date) -> DayResult:
    init_label = init_date.isoformat()

    nwp_file = forecast_path(config.nwp_root, init_date)
    aero_air_file = forecast_path(config.aero_root, init_date)
    aero_surface_file = forecast_path(config.aero_root, init_date, "_surface.h5")
    neuralgcm_file = forecast_path(config.neuralgcm_root, init_date)

    nwp_h5 = aero_air_h5 = aero_surface_h5 = neuralgcm_h5 = None
    truth_h5_files: dict[date, h5py.File] = {}
    try:
        truth_files = validate_truth_files(config, init_date)
        nwp_h5, nwp_ds = open_unique_dataset(nwp_file, NWP_SHAPE)
        aero_air_h5, aero_air_ds = open_unique_dataset(aero_air_file, AERO_AIR_SHAPE)
        aero_surface_h5, aero_surface_ds = open_unique_dataset(aero_surface_file, AERO_SURFACE_SHAPE)
        neuralgcm_h5, neuralgcm_ds = open_unique_dataset(neuralgcm_file, NEURALGCM_SHAPE)

        truth_ds_by_day: dict[date, h5py.Dataset] = {}
        for valid_day, path in truth_files.items():
            h5_file, dataset = open_unique_dataset(path, TRUTH_SHAPE)
            truth_h5_files[valid_day] = h5_file
            truth_ds_by_day[valid_day] = dataset

        nwp_rmse_day = np.empty((N_LEADS, N_VARS), dtype=np.float32)
        aero_rmse_day = np.empty((N_LEADS, N_VARS), dtype=np.float32)
        neuralgcm_rmse_day = np.full((N_LEADS, N_VARS), np.nan, dtype=np.float32)

        init_dt = datetime(init_date.year, init_date.month, init_date.day)
        for lead_idx in range(N_LEADS):
            valid_time = init_dt + timedelta(hours=lead_idx + 1)
            truth_ds = truth_ds_by_day[valid_time.date()]
            truth_slice = truth_ds[valid_time.hour, :, :, :]

            nwp_rmse_day[lead_idx, :] = rmse(nwp_ds[lead_idx, :, :, :], truth_slice)

            aero_pred = np.empty((N_VARS, *GRID_SHAPE), dtype=np.float32)
            aero_pred[:N_AIR_VARS] = aero_air_ds[lead_idx, :, :, :]
            aero_pred[N_AIR_VARS:] = aero_surface_ds[lead_idx, :, :, :]
            aero_rmse_day[lead_idx, :] = rmse(aero_pred, truth_slice)

            neural_pred = neuralgcm_ds[lead_idx, :, :, :]
            neural_truth = truth_slice[:N_AIR_VARS, :, :]
            neuralgcm_rmse_day[lead_idx, :N_AIR_VARS] = rmse(neural_pred, neural_truth)

        return DayResult(
            init_label,
            True,
            nwp_rmse=nwp_rmse_day,
            aero_rmse=aero_rmse_day,
            neuralgcm_rmse=neuralgcm_rmse_day,
        )
    except FileNotFoundError as exc:
        return DayResult(init_label, False, skip_reason=f"missing_file:{Path(str(exc)).name}")
    except Exception as exc:
        return DayResult(init_label, False, skip_reason=f"processing_error:{exc}")
    finally:
        for h5_file in [nwp_h5, aero_air_h5, aero_surface_h5, neuralgcm_h5]:
            if h5_file is not None:
                h5_file.close()
        for h5_file in truth_h5_files.values():
            h5_file.close()


def progress_iter(futures: Iterable, total: int, report_every: int):
    if tqdm is not None:
        yield from tqdm(futures, total=total, desc="Processing init days", unit="day")
        return

    for idx, item in enumerate(futures, start=1):
        if idx == 1 or idx % report_every == 0 or idx == total:
            print(f"[progress] completed {idx}/{total} init days")
        yield item


def collect_results(config: Config, workers: int, serial: bool, report_every: int) -> tuple[np.ndarray, ...]:
    init_dates = list_init_dates(config.year)
    valid_results: list[DayResult] = []
    skip_counter: Counter[str] = Counter()

    if serial:
        for init_date in progress_iter(init_dates, len(init_dates), report_every):
            result = process_one_day(config, init_date)
            if result.ok:
                valid_results.append(result)
            else:
                skip_counter[result.skip_reason or "unknown"] += 1
                print(f"[skip] {result.init_date}: {result.skip_reason}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_date = {executor.submit(process_one_day, config, init_date): init_date for init_date in init_dates}
            for future in progress_iter(as_completed(future_to_date), len(future_to_date), report_every):
                init_date = future_to_date[future]
                try:
                    result = future.result()
                except Exception as exc:
                    skip_counter[f"worker_error:{type(exc).__name__}"] += 1
                    print(f"[skip] {init_date.isoformat()}: worker_error:{exc}")
                    continue
                if result.ok:
                    valid_results.append(result)
                else:
                    skip_counter[result.skip_reason or "unknown"] += 1
                    print(f"[skip] {result.init_date}: {result.skip_reason}")

    valid_results.sort(key=lambda item: item.init_date)
    if not valid_results:
        raise RuntimeError("No valid init days found; cannot compute t-tests.")

    valid_dates = np.array([item.init_date for item in valid_results])
    nwp_rmse = np.stack([item.nwp_rmse for item in valid_results], axis=0)
    aero_rmse = np.stack([item.aero_rmse for item in valid_results], axis=0)
    neuralgcm_rmse = np.stack([item.neuralgcm_rmse for item in valid_results], axis=0)

    print(f"[summary] valid days: {len(valid_results)} / {len(init_dates)}")
    print("[summary] skip reasons:")
    for reason, count in skip_counter.most_common():
        print(f"  {reason}: {count}")

    return valid_dates, nwp_rmse, aero_rmse, neuralgcm_rmse


def lag1_autocorrelation(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    x0 = values[:-1]
    x1 = values[1:]
    if np.nanstd(x0) == 0.0 or np.nanstd(x1) == 0.0:
        return 0.0
    rho = float(np.corrcoef(x0, x1)[0, 1])
    return rho if np.isfinite(rho) else 0.0


def effective_sample_size(n_samples: int, rho: float) -> tuple[float, float, float]:
    # Positive temporal autocorrelation inflates the standard error.
    # Negative rho is not used to claim more independent information.
    rho_used = min(max(rho, 0.0), 0.99)
    inflation = (1.0 + rho_used) / (1.0 - rho_used)
    ess = n_samples / inflation
    return max(2.0, min(float(n_samples), ess)), rho_used, inflation


def paired_ttest(
    data1: np.ndarray,
    data2: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_stat_matrix = np.full((N_LEADS, N_VARS), np.nan, dtype=np.float64)
    p_value_matrix = np.full((N_LEADS, N_VARS), np.nan, dtype=np.float64)
    ess_matrix = np.full((N_LEADS, N_VARS), np.nan, dtype=np.float64)
    rho1_matrix = np.full((N_LEADS, N_VARS), np.nan, dtype=np.float64)
    inflation_matrix = np.full((N_LEADS, N_VARS), np.nan, dtype=np.float64)
    significant = np.zeros((N_LEADS, N_VARS), dtype=bool)

    for time_idx in range(N_LEADS):
        for var_idx in range(N_VARS):
            var_time_data1 = data1[:, time_idx, var_idx]
            var_time_data2 = data2[:, time_idx, var_idx]
            valid_mask = np.isfinite(var_time_data1) & np.isfinite(var_time_data2)
            n_valid = int(valid_mask.sum())
            if n_valid < 2:
                continue
            diff = var_time_data1[valid_mask].astype(np.float64) - var_time_data2[valid_mask].astype(np.float64)
            rho1 = lag1_autocorrelation(diff)
            ess, rho_used, inflation = effective_sample_size(n_valid, rho1)
            mean_diff = float(np.mean(diff))
            std_diff = float(np.std(diff, ddof=1))

            if std_diff == 0.0:
                if mean_diff == 0.0:
                    t_stat, p_value = 0.0, 1.0
                else:
                    t_stat = float(np.sign(mean_diff) * 1.0e12)
                    p_value = 0.0
            else:
                t_stat = mean_diff / (std_diff / np.sqrt(ess))
                p_value = 2.0 * stats.t.sf(abs(t_stat), df=ess - 1.0)

            t_stat_matrix[time_idx, var_idx] = t_stat
            p_value_matrix[time_idx, var_idx] = p_value
            ess_matrix[time_idx, var_idx] = ess
            rho1_matrix[time_idx, var_idx] = rho_used
            inflation_matrix[time_idx, var_idx] = inflation
            significant[time_idx, var_idx] = bool(np.isfinite(p_value) and p_value < alpha)

    return t_stat_matrix, p_value_matrix, significant, ess_matrix, rho1_matrix, inflation_matrix


def main() -> None:
    args = parse_args()
    config = Config(
        nwp_root=args.nwp_root,
        aero_root=args.aero_root,
        neuralgcm_root=args.neuralgcm_root,
        truth_root=args.truth_root,
        output=args.output,
        year=args.year,
    )

    workers = max(1, args.workers)
    print(f"[config] year={config.year}, workers={workers}, serial={args.serial}")
    print(f"[config] output={config.output}")

    valid_dates, nwp_rmse, aero_rmse, neuralgcm_rmse = collect_results(
        config=config,
        workers=workers,
        serial=args.serial,
        report_every=args.report_every,
    )

    (
        aero_vs_nwp_t_stat,
        aero_vs_nwp_p_value,
        aero_vs_nwp_significant,
        aero_vs_nwp_ess,
        aero_vs_nwp_rho1,
        aero_vs_nwp_inflation,
    ) = paired_ttest(aero_rmse, nwp_rmse)
    (
        aero_vs_neuralgcm_t_stat,
        aero_vs_neuralgcm_p_value,
        aero_vs_neuralgcm_significant,
        aero_vs_neuralgcm_ess,
        aero_vs_neuralgcm_rho1,
        aero_vs_neuralgcm_inflation,
    ) = paired_ttest(
        aero_rmse, neuralgcm_rmse
    )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output,
        valid_dates=valid_dates,
        variable_names=np.array(VARIABLE_NAMES),
        nwp_rmse=nwp_rmse,
        aero_rmse=aero_rmse,
        neuralgcm_rmse=neuralgcm_rmse,
        aero_vs_nwp_t_stat=aero_vs_nwp_t_stat,
        aero_vs_nwp_p_value=aero_vs_nwp_p_value,
        aero_vs_nwp_significant=aero_vs_nwp_significant,
        aero_vs_nwp_ess=aero_vs_nwp_ess,
        aero_vs_nwp_rho1=aero_vs_nwp_rho1,
        aero_vs_nwp_inflation=aero_vs_nwp_inflation,
        aero_vs_neuralgcm_t_stat=aero_vs_neuralgcm_t_stat,
        aero_vs_neuralgcm_p_value=aero_vs_neuralgcm_p_value,
        aero_vs_neuralgcm_significant=aero_vs_neuralgcm_significant,
        aero_vs_neuralgcm_ess=aero_vs_neuralgcm_ess,
        aero_vs_neuralgcm_rho1=aero_vs_neuralgcm_rho1,
        aero_vs_neuralgcm_inflation=aero_vs_neuralgcm_inflation,
    )

    print(f"[done] saved: {config.output}")
    print(f"[done] nwp_rmse shape: {nwp_rmse.shape}")
    print(f"[done] aero_rmse shape: {aero_rmse.shape}")
    print(f"[done] neuralgcm_rmse shape: {neuralgcm_rmse.shape}")
    print(f"[done] aero_vs_nwp_significant shape: {aero_vs_nwp_significant.shape}")
    print(f"[done] aero_vs_neuralgcm_significant shape: {aero_vs_neuralgcm_significant.shape}")


if __name__ == "__main__":
    main()
