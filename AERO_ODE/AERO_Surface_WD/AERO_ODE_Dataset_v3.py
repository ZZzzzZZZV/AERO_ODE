import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(next(_p for _p in _pathlib.Path(__file__).resolve().parents
                             if (_p / "common" / "paths.py").is_file())))
from common.paths import ROOT, find_hrrr_daily_file, iter_era5_daily_files

import os
import time
import json
import hashlib
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List, Optional

import h5py
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

DEFAULT_INPUT_VARS = [
    'geopotential', 'specific_humidity', 'temperature',
    'u_component_of_wind', 'v_component_of_wind',
    'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    '2m_temperature', 'mean_sea_level_pressure',
    'sea_surface_temperature', 'sea_ice_cover'
]

HRRR_SHAPE = (24, 440, 408)

ERA5_CACHE_SIZE = 32
HRRR_CACHE_SIZE = 64

class ERA5XarrayDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        hrrr_root_dirs: Optional[List[str]] = None,
        input_vars: List[str] = None,
        predict_lead_time: int = 48,
        filter_invalid_hrrr: bool = True,
        valid_index_cache_path: Optional[str] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.hrrr_root_dirs = hrrr_root_dirs or []
        self.input_vars = input_vars or DEFAULT_INPUT_VARS
        self.predict_lead_time = predict_lead_time
        self.filter_invalid_hrrr = filter_invalid_hrrr
        self.valid_index_cache_path = valid_index_cache_path

        self.timeline = []
        self.file_paths = []
        self.path_to_idx = {}

        self.hrrr_file_paths = []
        self.hrrr_path_to_idx = {}

        self._era5_cache: OrderedDict[int, xr.Dataset] = None
        self._hrrr_cache: OrderedDict[int, h5py.File] = None
        self._worker_pid = None

        self.coords_cache = {}

        self._hrrr_fields_shape_cache = {}

        self._scan_era5_files()

        if self.hrrr_root_dirs:
            self._scan_hrrr_files()

        self._init_coords_cache()

        cache_meta = self._make_valid_index_cache_meta()
        cache_path = self._resolve_valid_index_cache_path(cache_meta)
        cached = self._try_load_valid_indices_cache(cache_path, cache_meta)
        if cached is not None:
            self.valid_indices = cached
        else:
            self.valid_indices = self._build_valid_indices()
            self._save_valid_indices_cache(cache_path, cache_meta, self.valid_indices)

    def _make_valid_index_cache_meta(self) -> dict:
        if self.timeline:
            t0 = self.timeline[0]["timestamp"]
            t1 = self.timeline[-1]["timestamp"]
            t0s = t0.strftime("%Y-%m-%d %H:%M:%S")
            t1s = t1.strftime("%Y-%m-%d %H:%M:%S")
        else:
            t0s, t1s = None, None

        meta = {
            "root_dir": os.path.normpath(self.root_dir),
            "hrrr_root_dirs": [os.path.normpath(p) for p in (self.hrrr_root_dirs or [])],
            "predict_lead_time": int(self.predict_lead_time),
            "filter_invalid_hrrr": bool(self.filter_invalid_hrrr),
            "input_vars": list(self.input_vars),
            "timeline_len": int(len(self.timeline)),
            "era5_file_count": int(len(self.file_paths)),
            "hrrr_file_count": int(len(self.hrrr_file_paths)),
            "timeline_start": t0s,
            "timeline_end": t1s,
        }
        return meta

    def _meta_signature(self, meta: dict) -> str:
        payload = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _resolve_valid_index_cache_path(self, meta: dict) -> Optional[str]:
        if self.valid_index_cache_path:
            return self.valid_index_cache_path

        sig = self._meta_signature(meta)
        cache_dir = str(ROOT / "AERO_Surface_WD/.dataset_cache")
        fname = f"valid_indices_{sig}.npz"
        return os.path.join(cache_dir, fname)

    def _try_load_valid_indices_cache(self, cache_path: Optional[str], meta: dict):
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                meta_json = z["meta_json"].item()
                saved_meta = json.loads(meta_json)
                if self._meta_signature(saved_meta) != self._meta_signature(meta):
                    return None
                indices = z["indices"].astype(np.int64).tolist()
                return indices
        except Exception as e:
            print(f"failed to read cache {cache_path}, err={e}")
            return None

    def _save_valid_indices_cache(self, cache_path: Optional[str], meta: dict, indices: list):
        if not cache_path:
            return
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            pid = os.getpid()
            tmp_path = f"{cache_path}.tmp.{pid}.{time.time_ns()}.npz"
            np.savez_compressed(
                tmp_path,
                indices=np.asarray(indices, dtype=np.int64),

                meta_json=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True), dtype=np.str_),
            )

            try:
                os.replace(tmp_path, cache_path)
            except FileNotFoundError:
                if not os.path.exists(cache_path):
                    raise
        except Exception as e:

            print(f"failed to write cache {cache_path}, err={e}")

    def _get_hrrr_fields_shape(self, file_idx: int):
        if file_idx in self._hrrr_fields_shape_cache:
            return self._hrrr_fields_shape_cache[file_idx]

        try:
            path = self.hrrr_file_paths[file_idx]
            with h5py.File(path, "r") as f:
                key = "data" if "data" in f else "fields"
                shape = tuple(f[key].shape)
        except Exception:
            shape = None

        self._hrrr_fields_shape_cache[file_idx] = shape
        return shape

    def _is_hrrr_file_valid(self, file_idx: int) -> bool:
        shape = self._get_hrrr_fields_shape(file_idx)

        if shape is None or len(shape) != 4:
            return False
        t, c, _, _ = shape

        if t < 24 or c < 20:
            return False
        return True

    def _build_valid_indices(self):
        max_start = len(self.timeline) - self.predict_lead_time
        if max_start <= 0:
            return []

        if not self.hrrr_root_dirs or not self.hrrr_file_paths:
            return list(range(max_start))

        bad_files = set()
        if self.filter_invalid_hrrr:
            for fi in range(len(self.hrrr_file_paths)):
                if not self._is_hrrr_file_valid(fi):
                    bad_files.add(fi)

        valid = []

        for start in range(max_start):
            ok = True
            for k in range(start, start + self.predict_lead_time):
                fi = self.timeline[k].get("hrrr_file_idx")
                if fi is None or fi in bad_files:
                    ok = False
                    break
            if ok:
                valid.append(start)

        return valid

    def _scan_era5_files(self):
        for nc_path, dt in iter_era5_daily_files(self.root_dir):
            fpath = os.fspath(nc_path)
            if fpath not in self.path_to_idx:
                self.path_to_idx[fpath] = len(self.file_paths)
                self.file_paths.append(fpath)

            file_idx = self.path_to_idx[fpath]

            for h in range(24):
                self.timeline.append({
                    'file_idx': file_idx,
                    'hour_index': h,
                    'timestamp': dt + timedelta(hours=h)
                })

        self.timeline.sort(key=lambda x: x['timestamp'])

    def _scan_hrrr_files(self):
        for item in self.timeline:
            found = find_hrrr_daily_file(self.hrrr_root_dirs, item['timestamp'])
            if found is None:
                continue
            found_path = str(found)
            if found_path not in self.hrrr_path_to_idx:
                self.hrrr_path_to_idx[found_path] = len(self.hrrr_file_paths)
                self.hrrr_file_paths.append(found_path)
            item['hrrr_file_idx'] = self.hrrr_path_to_idx[found_path]

    def _parse_date_from_path(self, root: str, fname: str) -> Optional[datetime]:
        date_str = fname.split('.')[0]

        if len(date_str) == 8 and date_str.isdigit():
            try:
                return datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                pass

        if len(date_str) == 2 and date_str.isdigit():
            parts = os.path.normpath(root).split(os.sep)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].isdigit() and len(parts[i]) == 4:
                    year = int(parts[i])
                    if i + 1 < len(parts) and parts[i+1].isdigit() and len(parts[i+1]) == 2:
                        try:
                            return datetime(year, int(parts[i+1]), int(date_str))
                        except ValueError:
                            pass

        if len(date_str) == 4 and date_str.isdigit():
            parts = os.path.normpath(root).split(os.sep)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].isdigit() and len(parts[i]) == 4:
                    try:
                        year = int(parts[i])
                        return datetime(year, int(date_str[:2]), int(date_str[2:]))
                    except ValueError:
                        pass

        return None

    def _init_coords_cache(self):
        if not self.file_paths:
            return

        try:
            ds = xr.open_dataset(self.file_paths[0])
            self.coords_cache = {
                'level': ds.coords.get('level', ds.get('level')),
                'latitude': ds.coords.get('latitude', ds.coords.get('lat')),
                'longitude': ds.coords.get('longitude', ds.coords.get('lon'))
            }
            self.coords_cache = {
                k: v.values if v is not None else None
                for k, v in self.coords_cache.items()
            }
            ds.close()
        except Exception as e:
            print(f"failed to init coordinate cache: {e}")
            self.coords_cache = {'level': None, 'latitude': None, 'longitude': None}

    def _ensure_cache(self):
        pid = os.getpid()
        if self._worker_pid != pid:

            self._era5_cache = OrderedDict()
            self._hrrr_cache = OrderedDict()
            self._worker_pid = pid

    def _get_era5_dataset(self, file_idx: int) -> Optional[xr.Dataset]:
        self._ensure_cache()

        if file_idx in self._era5_cache:
            self._era5_cache.move_to_end(file_idx)
            return self._era5_cache[file_idx]

        if len(self._era5_cache) >= ERA5_CACHE_SIZE:
            oldest_idx, oldest_ds = self._era5_cache.popitem(last=False)
            try:
                oldest_ds.close()
            except:
                pass

        file_path = self.file_paths[file_idx]
        for attempt in range(3):
            try:
                ds = xr.open_dataset(file_path)
                self._era5_cache[file_idx] = ds
                return ds
            except Exception as e:
                if attempt == 2:
                    print(f"cannot open ERA5 file {file_path}: {e}")
                    return None
                time.sleep(0.1 * (2 ** attempt))

        return None

    def _get_hrrr_handle(self, file_idx: int) -> Optional[h5py.File]:
        self._ensure_cache()

        if file_idx in self._hrrr_cache:
            f = self._hrrr_cache[file_idx]
            try:
                if f.id.valid:
                    self._hrrr_cache.move_to_end(file_idx)
                    return f
            except:
                pass

            del self._hrrr_cache[file_idx]

        if len(self._hrrr_cache) >= HRRR_CACHE_SIZE:
            oldest_idx, oldest_f = self._hrrr_cache.popitem(last=False)
            try:
                oldest_f.close()
            except:
                pass

        path = self.hrrr_file_paths[file_idx]
        for attempt in range(3):
            try:
                f = h5py.File(path, 'r', libver='latest')
                self._hrrr_cache[file_idx] = f
                return f
            except Exception as e:
                if attempt == 2:
                    print(f"cannot open HRRR file {path}: {e}")
                    return None
                time.sleep(0.1 * (2 ** attempt))

        return None

    def _read_era5_data(self, start_idx: int, length: int) -> dict:
        slice_info = self.timeline[start_idx:start_idx + length]
        if not slice_info:
            return {}

        tasks = []
        curr_file = slice_info[0]['file_idx']
        curr_start = slice_info[0]['hour_index']
        count = 0

        for item in slice_info:
            if item['file_idx'] == curr_file:
                count += 1
            else:
                tasks.append((curr_file, curr_start, curr_start + count))
                curr_file = item['file_idx']
                curr_start = item['hour_index']
                count = 1
        tasks.append((curr_file, curr_start, curr_start + count))

        buffer = {v: [] for v in self.input_vars}

        for file_idx, start, end in tasks:
            ds = self._get_era5_dataset(file_idx)
            if ds is None:
                continue

            try:
                for var in self.input_vars:
                    if var in ds:

                        data = ds[var].isel(time=slice(start, end)).values
                        buffer[var].append(data)
            except Exception as e:
                print(f"failed to read ERA5 (file_idx={file_idx}): {e}")
                continue

        result = {}
        for var, data_list in buffer.items():
            if data_list:
                result[var] = np.concatenate(data_list, axis=0)

        return result

    def _read_hrrr_data(self, start_idx: int, length: int) -> Optional[np.ndarray]:
        slice_info = self.timeline[start_idx:start_idx + length]
        if not slice_info:
            return None

        tasks = []
        curr_file = slice_info[0].get('hrrr_file_idx')
        curr_start = slice_info[0]['hour_index']
        count = 0

        for item in slice_info:
            idx = item.get('hrrr_file_idx')
            if idx == curr_file:
                count += 1
            else:
                tasks.append((curr_file, curr_start, curr_start + count))
                curr_file = idx
                curr_start = item['hour_index']
                count = 1
        tasks.append((curr_file, curr_start, curr_start + count))

        data_list = []
        task_info = []

        for file_idx, start, end in tasks:
            if file_idx is None:

                ts = slice_info[0]['timestamp']
                raise ValueError(f"missing HRRR file: no HRRR near time {ts}")

            file_path = self.hrrr_file_paths[file_idx]
            f = self._get_hrrr_handle(file_idx)
            if f is None:
                raise ValueError(f"cannot open HRRR file: {file_path}")

            try:
                hkey = "data" if "data" in f else "fields"
                data = f[hkey][start:end]

                if data.ndim != 4:
                    raise ValueError(
                        f"unexpected HRRR shape!\n"
                        f"  file: {file_path}\n"
                        f"  slice: [{start}:{end}]\n"
                        f"  returned shape: {data.shape} (expected 4-D)\n"
                        f"  full {hkey} shape: {f[hkey].shape}"
                    )

                data_list.append(data)
                task_info.append(f"[read] {os.path.basename(file_path)}, hours={start}:{end}, shape={data.shape}")

            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"failed to read HRRR file: {file_path}, error: {e}")

        if data_list:
            dims = [arr.ndim for arr in data_list]
            if len(set(dims)) > 1:
                print(f"\nHRRR chunk dimensions do not match, cannot concatenate.")
                print(f"   start_idx={start_idx}, length={length}")
                print(f"   chunk dims: {dims}")
                for i, (arr, info) in enumerate(zip(data_list, task_info)):
                    print(f"   [{i}] shape={arr.shape}, {info}")
                raise ValueError(f"HRRR dimension mismatch: {dims}")

        return np.concatenate(data_list, axis=0) if data_list else None

    def _wrap_as_xarray(self, data_dict: dict, start_time: datetime, length: int) -> xr.Dataset:
        if not data_dict:
            raise ValueError("data dict is empty")

        sample = next(iter(data_dict.values()))
        actual_length = sample.shape[0]

        if actual_length != length:
            raise ValueError(f"length mismatch: expected {length}, got {actual_length}")

        for name, data in data_dict.items():
            if data.shape[0] != actual_length:
                raise ValueError(f"variable {name} time mismatch: {data.shape[0]} vs {actual_length}")

        times = [start_time + timedelta(hours=h) for h in range(actual_length)]

        coords = {'time': times}
        if self.coords_cache.get('level') is not None:
            coords['level'] = self.coords_cache['level']

        lat = self.coords_cache.get('latitude')
        lon = self.coords_cache.get('longitude')

        if lat is not None and lon is not None:
            if len(lat) == sample.shape[-2]:
                coords['latitude'] = lat
                coords['longitude'] = lon
                dim_names = ['latitude', 'longitude']
            else:
                coords['latitude'] = lat
                coords['longitude'] = lon
                dim_names = ['longitude', 'latitude']
        else:
            coords['latitude'] = np.arange(sample.shape[-2], dtype=np.float32)
            coords['longitude'] = np.arange(sample.shape[-1], dtype=np.float32)
            dim_names = ['latitude', 'longitude']

        data_vars = {}
        for name, data in data_dict.items():
            if data.ndim == 4:
                dims = ('time', 'level', *dim_names)
            else:
                dims = ('time', *dim_names)
            data_vars[name] = (dims, data)

        return xr.Dataset(data_vars, coords=coords)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        try:
            global_idx = self.valid_indices[idx]
            start_time = self.timeline[global_idx]['timestamp']

            era5_data = self._read_era5_data(global_idx, self.predict_lead_time)
            era5_xr = self._wrap_as_xarray(era5_data, start_time, self.predict_lead_time)

            hrrr_data = self._read_hrrr_data(global_idx, self.predict_lead_time)

            time_str = start_time.strftime("%Y/%m/%d/%H")
            return era5_xr, time_str, hrrr_data

        except Exception as e:

            print(f"skipping sample {idx} (time: {self.timeline[self.valid_indices[idx]]['timestamp']}): {e}")
            return None

    def __del__(self):

        if self._era5_cache:
            for ds in self._era5_cache.values():
                try:
                    ds.close()
                except:
                    pass

        if self._hrrr_cache:
            for f in self._hrrr_cache.values():
                try:
                    f.close()
                except:
                    pass

def xarray_collate_fn(batch):

    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None

    era5_list, time_list, hrrr_list = zip(*batch)

    try:
        hrrr_stack = np.stack(hrrr_list, axis=0)
        hrrr_tensor = torch.from_numpy(hrrr_stack)
    except Exception as e:
        print(f"HRRR Collate Error: {e}")
        hrrr_tensor = None

    return list(era5_list), list(time_list), hrrr_tensor

def verify_time_alignment(dataset: ERA5XarrayDataset, samples_per_year: int = 3):
    import random
    from collections import defaultdict

    print("\n" + "=" * 60)
    print("time-alignment check")
    print("=" * 60)

    samples_by_year = defaultdict(list)
    for idx, item in enumerate(dataset.timeline):
        year = item['timestamp'].year
        samples_by_year[year].append(idx)

    print(f"\nsamples by year:")
    for year in sorted(samples_by_year.keys()):
        print(f"  {year}: {len(samples_by_year[year])} timeline records")

    target_years = [2021, 2022, 2023]
    errors = []
    warnings = []

    for year in target_years:
        if year not in samples_by_year:
            print(f"\n[skip] no data for {year}")
            continue

        year_samples = samples_by_year[year]
        sample_indices = random.sample(year_samples, min(samples_per_year, len(year_samples)))

        print(f"\n--- {year} check ({len(sample_indices)} samples) ---")

        for timeline_idx in sample_indices:
            item = dataset.timeline[timeline_idx]
            file_idx = item['file_idx']
            hour_index = item['hour_index']
            expected_time = item['timestamp']
            file_path = dataset.file_paths[file_idx]

            print(f"\n  sample timeline[{timeline_idx}]:")
            print(f"    file: {os.path.basename(file_path)}")
            print(f"    expected time: {expected_time} (hour_index={hour_index})")

            try:
                ds = xr.open_dataset(file_path)
                actual_times = ds.time.values
                num_hours = len(actual_times)

                print(f"    hours in file: {num_hours}")

                if num_hours != 24:
                    msg = f"file {os.path.basename(file_path)} has {num_hours} hours (expected 24)"
                    warnings.append(msg)
                    print(f"    warning: {msg}")

                if hour_index < num_hours:

                    actual_time = np.datetime64(actual_times[hour_index], 'us').astype('datetime64[s]')
                    actual_dt = datetime.utcfromtimestamp(actual_time.astype('int64'))

                    print(f"    actual time[{hour_index}]: {actual_dt}")

                    if actual_dt != expected_time:
                        msg = f"time mismatch: expected={expected_time}, actual={actual_dt}"
                        errors.append(msg)
                        print(f"    error: {msg}")
                    else:
                        print(f"    time matches")
                else:
                    msg = f"hour_index={hour_index} out of range (file has {num_hours} hours)"
                    errors.append(msg)
                    print(f"    error: {msg}")

                first_time = np.datetime64(actual_times[0], 'us').astype('datetime64[s]')
                last_time = np.datetime64(actual_times[-1], 'us').astype('datetime64[s]')
                first_dt = datetime.utcfromtimestamp(first_time.astype('int64'))
                last_dt = datetime.utcfromtimestamp(last_time.astype('int64'))
                print(f"    file time range: {first_dt} ~ {last_dt}")

                ds.close()

            except Exception as e:
                errors.append(f"cannot read file {file_path}: {e}")
                print(f"    read failed: {e}")

            hrrr_file_idx = item.get('hrrr_file_idx')
            if hrrr_file_idx is not None:
                hrrr_path = dataset.hrrr_file_paths[hrrr_file_idx]
                try:
                    with h5py.File(hrrr_path, 'r') as f:
                        hrrr_shape = f['fields'].shape
                        print(f"    HRRR file: {os.path.basename(hrrr_path)}")
                        print(f"    HRRR shape: {hrrr_shape}")
                        if hrrr_shape[0] != 24:
                            msg = f"HRRR file {os.path.basename(hrrr_path)} time dim={hrrr_shape[0]} (expected 24)"
                            warnings.append(msg)
                            print(f"    warning: {msg}")
                except Exception as e:
                    print(f"    HRRR read failed: {e}")

    print("\n" + "=" * 60)
    print("check summary")
    print("=" * 60)
    print(f"  errors: {len(errors)}")
    print(f"  warnings: {len(warnings)}")

    if errors:
        print("\nerrors:")
        for e in errors[:10]:
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    if warnings:
        print("\nwarnings:")
        for w in warnings[:10]:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")

    if not errors and not warnings:
        print("\nall sampled checks passed; times align.")

    return len(errors) == 0
