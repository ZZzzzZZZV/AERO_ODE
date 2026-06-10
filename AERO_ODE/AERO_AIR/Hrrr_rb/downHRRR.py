#!/usr/bin/env python3
"""
Download HRRR 3 km data for a region (Oct–Dec 2024 by default),
interpolate to a lat/lon grid, and save as (24, 24, 440, 408) .h5 files.

Usage (on server):
  python downHRRR.py
  python downHRRR.py --start 2024-11-01 --end 2024-11-30
  python downHRRR.py --out-dir /path/to/output
  # If Permission denied: '/home/xxx/data', set Herbie cache to a writable dir:
  python downHRRR.py --herbie-cache /path/you/can/write/herbie_cache
  # Slow in China: try Azure/Google first (default aws is often slow):
  python downHRRR.py --priority azure,google,aws,nomads
  # China mirror: OpenXLab OpenScienceLab/HRRR points to AWS only (404 direct download); use Herbie:
  python downHRRR.py --priority azure,google,aws
  # If opendatalab is configured: --source opendatalab --mirror-path-fmt "..." (files must exist on platform)

Requires: pip install herbie-data numpy scipy h5py
  Note: OpenXLab OpenScienceLab/HRRR references NOAA only (no direct download on site).
  Use Herbie from AWS/Google/Azure; in China try --priority azure,google,aws.
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import h5py
from scipy.interpolate import griddata

# Download retry (infinite until success; Ctrl+C to exit)
DOWNLOAD_RETRY_DELAY = 10

# OpenDataLab mirror (odl login first; set OPENDATALAB_HOME or --opendatalab-home if home not writable)
OPENDATALAB_HRRR_REPO = "OpenScienceLab/HRRR"
OPENDATALAB_HOME_DEFAULT = "/home/zhangjing/Code/AERO_AIR/Hrrr_rb"  # writable dir for odl config


def _merge_hypercubes(ds_or_list):
    """Merge cfgrib list of Datasets (multiple hypercubes) into one Dataset.
    Avoids merge(compat='override'): use first as base, keep its coords to avoid heightAboveGround conflicts.
    If the same variable appears in multiple cubes with a pressure level dim, concat along that dim.
    """
    if not isinstance(ds_or_list, list):
        return ds_or_list
    if len(ds_or_list) == 1:
        return ds_or_list[0]
    base = ds_or_list[0].copy(deep=True)
    level_coords = [GRIB_LEVEL_COORD_HPA, GRIB_LEVEL_COORD_PA]
    for ds in ds_or_list[1:]:
        for v in ds.data_vars:
            dv = ds[v]
            if v not in base:
                base[v] = dv
                continue
            bv = base[v]
            for coord in level_coords:
                if coord in bv.dims and coord in dv.dims:
                    base[v] = xr.concat([bv, dv], dim=coord)
                    break
            else:
                base[v] = dv
    return base


def _find_file_in_dir(directory, filename):
    """Recursively find filename under directory; return first Path or None."""
    directory = Path(directory)
    if not directory.exists():
        return None
    for p in directory.rglob(filename):
        if p.is_file():
            return p
    return None


def _download_grib_from_opendatalab(dt, product, save_dir, path_fmt=None):
    """Download one-hour prs or sfc GRIB2 from OpenDataLab/OpenXLab; return local path or None.
    path_fmt: optional platform path template with {ymd},{y},{m},{d},{h},{fname}. If set, only that format is tried.
    """
    download_fn = None
    try:
        from openxlab.dataset import download as download_fn
    except ImportError:
        try:
            from opendatalab.dataset import download as download_fn
        except ImportError:
            return None
    if download_fn is None:
        return None

    ymd, h = dt.strftime("%Y%m%d"), dt.strftime("%H")
    y, m, d = dt.year, dt.month, dt.day
    if product == "prs":
        fname = f"hrrr.t{h}z.wrfprsf00.grib2"
    else:
        fname = f"hrrr.t{h}z.wrfsfcf00.grib2"
    save_dir = Path(save_dir)

    if path_fmt:
        source_paths_to_try = [
            path_fmt.format(ymd=ymd, y=y, m=f"{m:02d}", d=f"{d:02d}", h=h, fname=fname)
        ]
    else:
        source_paths_to_try = [
            f"hrrr.{ymd}/conus/{fname}",
            f"{ymd}/conus/{fname}",
            f"{y}/{m:02d}/{d:02d}/{fname}",
            f"{y}/{m:02d}/{d:02d}/conus/{fname}",
            f"{ymd}/{fname}",
            fname,
            f"conus/{fname}",
        ]
    attempt = 0
    _logged_no_file = [False]

    while True:
        try:
            for source_path in source_paths_to_try:
                try:
                    download_fn(
                        dataset_repo=OPENDATALAB_HRRR_REPO,
                        source_path=source_path,
                        target_path=str(save_dir),
                    )
                except Exception as sub_e:
                    err_str = str(sub_e).lower()
                    if "not found" in err_str or "404" in err_str or "no such" in err_str or "exist" in err_str:
                        continue
                    raise
                for candidate in [
                    save_dir / source_path,
                    save_dir / fname,
                    save_dir / f"hrrr.{ymd}" / "conus" / fname,
                    save_dir / ymd / fname,
                ]:
                    if candidate.exists():
                        return candidate
                found = _find_file_in_dir(save_dir, fname)
                if found is not None:
                    return found
            if not _logged_no_file[0]:
                _logged_no_file[0] = True
                try:
                    contents = list(Path(save_dir).iterdir())[:10]
                    print(f"  Mirror {product}: {fname} not found after download; save_dir sample: {[str(p.name) for p in contents]}")
                except Exception:
                    print(f"  Mirror {product}: {fname} not found; check HRRR dataset path layout on platform")
            return None
        except KeyboardInterrupt:
            raise
        except Exception as e:
            attempt += 1
            print(f"  Mirror {product} {dt} download failed, retry #{attempt} in {DOWNLOAD_RETRY_DELAY}s: {e}")
            time.sleep(DOWNLOAD_RETRY_DELAY)


def _open_local_grib(path):
    """Open local GRIB2 with cfgrib; merge if list of hypercubes."""
    ds = xr.open_dataset(path, engine="cfgrib")
    return _merge_hypercubes(ds) if isinstance(ds, list) else ds


def _get_hrrr_lat_lon(ds):
    """Get 2D lat/lon from Dataset; supports latitude/longitude or lat/lon."""
    for lat_name, lon_name in [("latitude", "longitude"), ("lat", "lon")]:
        if lat_name in ds and lon_name in ds:
            lat = np.asarray(ds[lat_name]).squeeze()
            lon = np.asarray(ds[lon_name]).squeeze()
            if lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            return lat, lon
    raise KeyError("Dataset has no latitude/longitude or lat/lon")

# Lat/lon grid paths (server)
LATS_NPY = "/home/zhangjing/Code/AERO_AIR/Hrrr_rb/lats.npy"
LONS_NPY = "/home/zhangjing/Code/AERO_AIR/Hrrr_rb/lons.npy"
OUT_BASE = "./test"  # output root, e.g. ./test/2024/10/01.h5

# Variable order: matches second dim of (24, 24, 440, 408)
VAR_ORDER = [
    "z50", "z500", "z850", "z1000",
    "t50", "t500", "t850", "t1000",
    "s50", "s500", "s850", "s1000",
    "u50", "u500", "u850", "u1000",
    "v50", "v500", "v850", "v1000",
    "mslp", "u10", "v10", "t2m",
]

# HRRR pressure levels (mb), first 20 vars in VAR_ORDER
PRESSURE_LEVELS = [50, 500, 850, 1000]
PRS_VARS = ["z", "t", "s", "u", "v"]  # geopotential height, T(K), specific humidity, u/v wind

# GRIB/cfgrib names (pressure coord may be isobaricInhPa or isobaricInPa)
GRIB_LEVEL_COORD_HPA = "isobaricInhPa"
GRIB_LEVEL_COORD_PA = "isobaricInPa"
GRIB_VAR_MAP = {
    "z": ["gh", "z", "HGT", "hgt"],
    "t": ["t", "TMP", "tmp"],
    "s": ["q", "spfh", "SPFH", "q"],
    "u": ["u", "UGRD", "u"],
    "v": ["v", "VGRD", "v"],
}


def load_target_grid(lats_path: str, lons_path: str):
    """Load target lat/lon grid, shape (440, 408)."""
    lats = np.load(lats_path)
    lons = np.load(lons_path)
    assert lats.shape == lons.shape, "lats and lons must have the same shape"
    return lats, lons


def interpolate_to_grid(lat_src, lon_src, values_2d, lat_tgt, lon_tgt, fill=np.nan):
    """Interpolate HRRR 2D field onto target (lat_tgt, lon_tgt) grid."""
    points = np.column_stack([lon_src.ravel(), lat_src.ravel()])
    values_flat = values_2d.ravel()
    valid = np.isfinite(values_flat)
    if not np.any(valid):
        return np.full_like(lat_tgt, fill, dtype=np.float32)
    points = points[valid]
    values_flat = values_flat[valid]
    target_points = np.column_stack([lon_tgt.ravel(), lat_tgt.ravel()])
    out_flat = griddata(
        points, values_flat, target_points,
        method="linear", fill_value=fill
    )
    return np.asarray(out_flat, dtype=np.float32).reshape(lat_tgt.shape)


def _level_coord_and_val(ds, level_hpa):
    """Return (coord name, value for sel). Pressure may be hPa or Pa."""
    for coord in [GRIB_LEVEL_COORD_HPA, GRIB_LEVEL_COORD_PA]:
        if coord in ds.coords:
            vals = ds.coords[coord].values
            if np.issubdtype(vals.dtype, np.floating) or np.max(vals) > 2000:
                sel_val = level_hpa * 100.0
            else:
                sel_val = level_hpa
            return coord, sel_val
    return None, None


def find_var_in_ds(ds, candidates, level=None):
    """Find variable in xarray Dataset by candidate names; optional level filter."""
    for c in candidates:
        if c in ds:
            v = ds[c]
            if level is not None:
                coord_name, sel_val = _level_coord_and_val(ds, level)
                if coord_name and coord_name in v.dims:
                    v = v.sel({coord_name: sel_val}, method="nearest")
            return v
    return None


def extract_prs_vars(ds_prs, lat_hrrr, lon_hrrr, lat_tgt, lon_tgt):
    """Extract z/t/s/u/v at 50,500,850,1000 mb from HRRR pressure-level xarray and interpolate."""
    out = {}
    for lev in PRESSURE_LEVELS:
        for vname, grib_names in GRIB_VAR_MAP.items():
            key = f"{vname}{lev}"
            var = find_var_in_ds(ds_prs, grib_names, level=lev)
            if var is None:
                continue
            data = np.asarray(var).squeeze()
            if data.ndim != 2:
                continue
            out[key] = interpolate_to_grid(
                lat_hrrr, lon_hrrr, data.astype(np.float32),
                lat_tgt, lon_tgt
            )
    return out


def _sfc_var_2d(ds, name, prefer_height_m=None):
    """Surface variable as 2D array; if heightAboveGround dim, pick prefer_height_m (e.g. 10 or 2)."""
    if name not in ds:
        return None
    v = ds[name]
    if "heightAboveGround" in v.dims:
        coords = np.asarray(v.coords["heightAboveGround"].values)
        if coords.size > 1 and prefer_height_m is not None:
            idx = int(np.argmin(np.abs(coords - prefer_height_m)))
            v = v.isel(heightAboveGround=idx)
        else:
            v = v.isel(heightAboveGround=0)
    v = v.squeeze()
    return np.asarray(v).astype(np.float32) if v.ndim == 2 else None

def extract_sfc_vars(ds_sfc, lat_hrrr, lon_hrrr, lat_tgt, lon_tgt):
    """Extract mslp, u10, v10, t2m from HRRR surface xarray and interpolate."""
    out = {}
    for name in ["prmsl", "msl", "mslp", "PRMSL"]:
        data = _sfc_var_2d(ds_sfc, name)
        if data is not None:
            out["mslp"] = interpolate_to_grid(
                lat_hrrr, lon_hrrr, data, lat_tgt, lon_tgt
            )
            break
    for ukey in ["u10", "10u", "UGRD_10m", "UGRD"]:
        data = _sfc_var_2d(ds_sfc, ukey, prefer_height_m=10.0)
        if data is not None:
            out["u10"] = interpolate_to_grid(
                lat_hrrr, lon_hrrr, data, lat_tgt, lon_tgt
            )
            break
    for vkey in ["v10", "10v", "VGRD_10m", "VGRD"]:
        data = _sfc_var_2d(ds_sfc, vkey, prefer_height_m=10.0)
        if data is not None:
            out["v10"] = interpolate_to_grid(
                lat_hrrr, lon_hrrr, data, lat_tgt, lon_tgt
            )
            break
    for tkey in ["t2m", "2t", "TMP_2m", "TMP"]:
        data = _sfc_var_2d(ds_sfc, tkey, prefer_height_m=2.0)
        if data is not None:
            out["t2m"] = interpolate_to_grid(
                lat_hrrr, lon_hrrr, data, lat_tgt, lon_tgt
            )
            break
    return out


def build_one_hour_data(dt, lats_tgt, lons_tgt, out_base: Path, save_dir=None, priority=None, source="herbie", mirror_path_fmt=None):
    """Download and process one HRRR hour; return dict of (24,) variable arrays or None.
    mirror_path_fmt: used when --source opendatalab; placeholders {ymd} {y} {m} {d} {h} {fname}.
    """
    date_str = dt.strftime("%Y-%m-%d %H:%M")

    if source == "opendatalab":
        if save_dir is None:
            save_dir = Path(out_base) / "herbie_cache"
        save_dir = Path(save_dir)
        path_prs = _download_grib_from_opendatalab(dt, "prs", save_dir, path_fmt=mirror_path_fmt)
        path_sfc = _download_grib_from_opendatalab(dt, "sfc", save_dir, path_fmt=mirror_path_fmt)
        if path_prs is None or path_sfc is None:
            return None
        try:
            ds_prs = _open_local_grib(path_prs)
            ds_sfc = _open_local_grib(path_sfc)
        except Exception as e:
            print(f"  {date_str} failed to open GRIB: {e}")
            return None
    else:
        try:
            from herbie import Herbie
        except ImportError:
            raise ImportError("Install: pip install herbie-data")
        kwargs = {}
        if save_dir is not None:
            kwargs["save_dir"] = str(save_dir)
        if priority is not None:
            kwargs["priority"] = priority

        search_prs = r"(?:HGT|TMP|SPFH|UGRD|VGRD):(?:1000|850|500|50) mb"
        H_prs = Herbie(date_str, model="hrrr", product="prs", fxx=0, **kwargs)
        ds_prs = None
        attempt = 0
        while True:
            try:
                ds_prs = H_prs.xarray(search=search_prs)
                ds_prs = _merge_hypercubes(ds_prs)
                break
            except Exception as e:
                attempt += 1
                print(f"  prs {date_str} download failed, retry #{attempt} in {DOWNLOAD_RETRY_DELAY}s: {e}")
                time.sleep(DOWNLOAD_RETRY_DELAY)

        search_sfc = r"(?:PRMSL:mean sea level|:UGRD:10 m|:VGRD:10 m|:TMP:2 m)"
        H_sfc = Herbie(date_str, model="hrrr", product="sfc", fxx=0, **kwargs)
        ds_sfc = None
        attempt = 0
        while True:
            try:
                ds_sfc = H_sfc.xarray(search=search_sfc)
                ds_sfc = _merge_hypercubes(ds_sfc)
                break
            except Exception as e:
                attempt += 1
                print(f"  sfc {date_str} download failed, retry #{attempt} in {DOWNLOAD_RETRY_DELAY}s: {e}")
                time.sleep(DOWNLOAD_RETRY_DELAY)

    try:
        lat_hrrr, lon_hrrr = _get_hrrr_lat_lon(ds_prs)
    except KeyError as e:
        print(f"  {date_str} cannot get lat/lon: {e}")
        return None

    prs_vars = extract_prs_vars(ds_prs, lat_hrrr, lon_hrrr, lats_tgt, lons_tgt)
    sfc_vars = extract_sfc_vars(ds_sfc, lat_hrrr, lon_hrrr, lats_tgt, lons_tgt)
    all_vars = {**prs_vars, **sfc_vars}

    ny, nx = lats_tgt.shape
    one_hour = np.zeros((24, ny, nx), dtype=np.float32)
    one_hour[:] = np.nan
    for i, key in enumerate(VAR_ORDER):
        if key in all_vars:
            one_hour[i] = all_vars[key]
    return one_hour


def process_one_day(date, lats_tgt, lons_tgt, out_base: Path, save_dir=None, priority=None, source="herbie", mirror_path_fmt=None, skip_existing: bool = True):
    """Process one day: 24 hours, save (24, 24, 440, 408) .h5."""
    out_dir = out_base / str(date.year) / f"{date.month:02d}"
    out_file = out_dir / f"{date.day:02d}.h5"
    if skip_existing and out_file.exists():
        print(f"Exists, skip {out_file}")
        return out_file

    day_data = []
    for hour in range(24):
        dt = datetime(date.year, date.month, date.day, hour, 0, 0)
        try:
            one = build_one_hour_data(dt, lats_tgt, lons_tgt, out_base, save_dir=save_dir, priority=priority, source=source, mirror_path_fmt=mirror_path_fmt)
            if one is None:
                print(f"  Skip {dt} (no data)")
                one = np.full((24, *lats_tgt.shape), np.nan, dtype=np.float32)
            day_data.append(one)
        except Exception as e:
            print(f"  {dt} failed: {e}")
            day_data.append(np.full((24, *lats_tgt.shape), np.nan, dtype=np.float32))

    stack = np.stack(day_data, axis=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        f.create_dataset("data", data=stack, compression="gzip")
        f.attrs["description"] = "hour, variable, lat, lon"
        f.attrs["variables"] = ",".join(VAR_ORDER)
        f.attrs["units_z"] = "gpm"
        f.attrs["units_t"] = "K"
    print(f"Saved {out_file} shape={stack.shape}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download HRRR Oct–Dec 2024 and save as .h5")
    parser.add_argument("--lats", default=LATS_NPY, help="Latitude grid .npy path")
    parser.add_argument("--lons", default=LONS_NPY, help="Longitude grid .npy path")
    parser.add_argument("--out-dir", default=OUT_BASE, help="Output root, e.g. ./test")
    parser.add_argument("--start", default="2024-10-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-download and overwrite existing .h5")
    parser.add_argument("--herbie-cache", default=None, help="Herbie cache dir (writable); default out-dir/herbie_cache")
    parser.add_argument("--priority", default=None, help="Herbie source priority, comma-separated, e.g. azure,google,aws,nomads")
    parser.add_argument("--source", default="herbie", choices=("herbie", "opendatalab"), help="Source: herbie (default) or opendatalab mirror (openxlab login required)")
    parser.add_argument("--opendatalab-home", default=None, help="Mirror config dir (writable), default: " + OPENDATALAB_HOME_DEFAULT)
    parser.add_argument("--mirror-path-fmt", default=None, help='Mirror path template: {ymd}=20241001 {y},{m},{d},{h} {fname}. E.g. 2024/{m}/{d}/{fname}')
    args = parser.parse_args()

    if getattr(args, "source", "herbie") == "opendatalab":
        odl_home = getattr(args, "opendatalab_home", None) or os.environ.get("OPENDATALAB_HOME") or OPENDATALAB_HOME_DEFAULT
        odl_home = Path(odl_home).resolve()
        if odl_home.exists() or not odl_home.exists():
            try:
                odl_home.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        os.environ["HOME"] = str(odl_home)
        print(f"OpenDataLab config dir (HOME): {odl_home}")

    if not os.path.isfile(args.lats) or not os.path.isfile(args.lons):
        raise FileNotFoundError(
            f"Lat/lon grid files not found: {args.lats}, {args.lons}\n"
            "Place lats.npy / lons.npy at the paths above, or use --lats / --lons."
        )

    lats_tgt, lons_tgt = load_target_grid(args.lats, args.lons)
    if lats_tgt.shape != (440, 408):
        print(f"Warning: grid shape {lats_tgt.shape}, expected (440, 408)")

    out_base = Path(args.out_dir)
    save_dir = Path(args.herbie_cache) if args.herbie_cache else (out_base / "herbie_cache")
    save_dir = save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    priority = None
    if args.priority:
        priority = [s.strip().lower() for s in args.priority.split(",") if s.strip()]
        print(f"Herbie source priority: {priority}")
    print(f"Herbie cache dir: {save_dir}")

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    skip_existing = not args.no_skip_existing
    current = start
    source = getattr(args, "source", "herbie") or "herbie"
    mirror_path_fmt = getattr(args, "mirror_path_fmt", None)
    if source == "opendatalab":
        print("Using mirror (openxlab or opendatalab; login required)")
        if mirror_path_fmt:
            print(f"Mirror path format: {mirror_path_fmt}")

    while current <= end:
        print(f"Processing {current} ...")
        process_one_day(current, lats_tgt, lons_tgt, out_base, save_dir=save_dir, priority=priority, source=source, mirror_path_fmt=mirror_path_fmt, skip_existing=skip_existing)
        current += timedelta(days=1)

    print("Done.")


if __name__ == "__main__":
    main()
