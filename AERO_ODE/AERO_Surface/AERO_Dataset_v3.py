"""
    ERA5 + HRRR dataset loader
"""

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
from torch.utils.data import Dataset, DataLoader


# ============================================
# Constants
# ============================================
DEFAULT_INPUT_VARS = [
    'geopotential', 'specific_humidity', 'temperature',
    'u_component_of_wind', 'v_component_of_wind',
    'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    '2m_temperature', 'mean_sea_level_pressure',
    'sea_surface_temperature', 'sea_ice_cover'
]

# Default HRRR data shape
HRRR_SHAPE = (24, 440, 408)  # (vars, lat, lon)

# Cache size limits
ERA5_CACHE_SIZE = 32   # Cache up to 32 ERA5 files
HRRR_CACHE_SIZE = 64   # Cache up to 64 HRRR files


# ============================================
# Main dataset class
# ============================================
class ERA5XarrayDataset(Dataset):
    """
    Joint ERA5 + HRRR dataset
    
    Args:
        root_dir: ERA5 data root directory
        hrrr_root_dirs: HRRR data root directories (may span years)
        input_vars: Input variable list
        predict_lead_time: Forecast length (hours)
    """
    
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
        
        # Timeline and file index
        self.timeline = []
        self.file_paths = []
        self.path_to_idx = {}
        
        # HRRR-related fields
        self.hrrr_file_paths = []
        self.hrrr_path_to_idx = {}
        
        # Multiprocess-safe: lazy file cache initialization
        # Each worker process owns its own cache
        # Use OrderedDict for true LRU caching
        self._era5_cache: OrderedDict[int, xr.Dataset] = None   # ERA5 xarray cache
        self._hrrr_cache: OrderedDict[int, h5py.File] = None    # HRRR h5py cache
        self._worker_pid = None
        
        # Coordinate cache
        self.coords_cache = {}

        # HRRR shape cache (init phase only; handles not reused across workers)
        self._hrrr_fields_shape_cache = {}
        
        # Initialization
        print(f"Scanning ERA5 directory: {root_dir}")
        self._scan_era5_files()
        
        if self.hrrr_root_dirs:
            print(f"Scanning HRRR directory: {self.hrrr_root_dirs}")
            self._scan_hrrr_files()
        
        self._init_coords_cache()

        # Valid sample indices (avoids dynamic skip during training that breaks batch size / DDP sync)
        cache_meta = self._make_valid_index_cache_meta()
        cache_path = self._resolve_valid_index_cache_path(cache_meta)
        cached = self._try_load_valid_indices_cache(cache_path, cache_meta)
        if cached is not None:
            self.valid_indices = cached
        else:
            self.valid_indices = self._build_valid_indices()
            self._save_valid_indices_cache(cache_path, cache_meta, self.valid_indices)

        print(
            f"Dataset ready: {len(self.file_paths)}  ERA5 files, "
            f"{len(self.hrrr_file_paths)}  HRRR files, "
            f"{len(self.valid_indices)}  valid samples (lead_time={predict_lead_time})"
        )

    def _make_valid_index_cache_meta(self) -> dict:
        """
        Build cache signature metadata to decide if on-disk cache is reusable.

        Notes:
        - We do not store every file path (too large); only key params, timeline range, and file counts.
        - If underlying data changes but metadata is unchanged, cache may be stale; delete cache files manually.
        """
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
        """Resolve cache path: use valid_index_cache_path if given, else .dataset_cache/ under cwd."""
        if self.valid_index_cache_path:
            return self.valid_index_cache_path

        sig = self._meta_signature(meta)
        cache_dir = os.path.join(os.getcwd(), ".dataset_cache")
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
                    print(f"Cache signature mismatch, ignoring cache: {cache_path}")
                    return None
                indices = z["indices"].astype(np.int64).tolist()
                print(f"Loaded valid-sample cache: {cache_path} (n={len(indices)})")
                return indices
        except Exception as e:
            print(f"Cache read failed, ignoring cache: {cache_path}, err={e}")
            return None

    def _save_valid_indices_cache(self, cache_path: Optional[str], meta: dict, indices: list):
        if not cache_path:
            return
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            # Under multiprocess/DDP, multiple processes may write the same cache_path:
            # - Do not share the same tmp filename (would overwrite/move each other)
            # - Use pid + timestamp for unique tmp, then atomic os.replace
            pid = os.getpid()
            tmp_path = f"{cache_path}.tmp.{pid}.{time.time_ns()}.npz"
            np.savez_compressed(
                tmp_path,
                indices=np.asarray(indices, dtype=np.int64),
                # NumPy 2.0 removed np.unicode_; store strings with np.str_
                meta_json=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True), dtype=np.str_),
            )
            # Use os.replace for atomic replace on Windows
            try:
                os.replace(tmp_path, cache_path)
                print(f"Wrote valid-sample cache: {cache_path} (n={len(indices)})")
            except FileNotFoundError:
                # Another process may have finished writing; treat existing target as success
                if os.path.exists(cache_path):
                    print(f"Valid-sample cache already exists (written by another process): {cache_path}")
                else:
                    raise
        except Exception as e:
            # Cache write failure does not block training
            print(f"Cache write failed (ignored): {cache_path}, err={e}")

    def _get_hrrr_fields_shape(self, file_idx: int):
        """Read and cache HRRR fields shape (init phase only)"""
        if file_idx in self._hrrr_fields_shape_cache:
            return self._hrrr_fields_shape_cache[file_idx]

        try:
            path = self.hrrr_file_paths[file_idx]
            with h5py.File(path, "r") as f:
                shape = tuple(f["fields"].shape)
        except Exception:
            shape = None

        self._hrrr_fields_shape_cache[file_idx] = shape
        return shape

    def _is_hrrr_file_valid(self, file_idx: int) -> bool:
        """Coarse check whether an HRRR daily file is usable (avoid failures during training)"""
        shape = self._get_hrrr_fields_shape(file_idx)
        # Expected fields: (T, C, H, W)
        if shape is None or len(shape) != 4:
            return False
        t, c, _, _ = shape
        # Training needs at least 24 hours and 20 variables
        if t < 24 or c < 20:
            return False
        return True

    def _build_valid_indices(self):
        """
        Build list of valid sample start indices.

        Goals:
        - Do not skip dynamically in __getitem__ (breaks batch size / DDP sync)
        - Ensure HRRR exists with valid dimensions for each window [start, start+lead)
        """
        max_start = len(self.timeline) - self.predict_lead_time
        if max_start <= 0:
            return []

        # Fall back to legacy logic when HRRR is absent
        if not self.hrrr_root_dirs or not self.hrrr_file_paths:
            return list(range(max_start))

        # Pre-scan bad files (wrong shape / insufficient dims)
        bad_files = set()
        if self.filter_invalid_hrrr:
            for fi in range(len(self.hrrr_file_paths)):
                if not self._is_hrrr_file_valid(fi):
                    bad_files.add(fi)

        valid = []
        # Check HRRR availability per window
        for start in range(max_start):
            ok = True
            for k in range(start, start + self.predict_lead_time):
                fi = self.timeline[k].get("hrrr_file_idx")
                if fi is None or fi in bad_files:
                    ok = False
                    break
            if ok:
                valid.append(start)

        if self.filter_invalid_hrrr and bad_files:
            print(f"Pre-filtered bad HRRR files: {len(bad_files)}/{len(self.hrrr_file_paths)}")
        skipped = max_start - len(valid)
        if skipped > 0:
            print(f"Pre-filtered bad time windows: {skipped}/{max_start} (reason: missing/bad HRRR)")

        return valid

    # ----------------------------------------
    # file scanning
    # ----------------------------------------
    def _scan_era5_files(self):
        """Scan ERA5 directory and parse file dates"""
        for root, _, files in os.walk(self.root_dir):
            for fname in sorted(files):
                if not fname.endswith('.nc'):
                    continue
                
                dt = self._parse_date_from_path(root, fname)
                if dt is None:
                    continue
                
                fpath = os.path.join(root, fname)
                if fpath not in self.path_to_idx:
                    self.path_to_idx[fpath] = len(self.file_paths)
                    self.file_paths.append(fpath)
                
                file_idx = self.path_to_idx[fpath]
                
                # Each file contains 24 hours
                for h in range(24):
                    self.timeline.append({
                        'file_idx': file_idx,
                        'hour_index': h,
                        'timestamp': dt + timedelta(hours=h)
                    })
        
        self.timeline.sort(key=lambda x: x['timestamp'])

    def _scan_hrrr_files(self):
        """Scan HRRR files and link to timeline"""
        for item in self.timeline:
            ts = item['timestamp']
            year_str = str(ts.year)
            
            # Try different path combinations
            candidates = [
                f"{ts.year}/{ts.month:02d}/{ts.day:02d}.h5",
                f"{ts.month:02d}/{ts.day:02d}.h5",
                f"{ts.day:02d}.h5"
            ]
            
            found_path = None
            for root in self.hrrr_root_dirs:
                # Search only roots containing the target year
                if year_str not in root:
                    continue
                    
                for rel in candidates:
                    p = os.path.join(root, rel)
                    if os.path.exists(p):
                        found_path = p
                        break
                if found_path:
                    break
            
            if found_path:
                if found_path not in self.hrrr_path_to_idx:
                    self.hrrr_path_to_idx[found_path] = len(self.hrrr_file_paths)
                    self.hrrr_file_paths.append(found_path)
                item['hrrr_file_idx'] = self.hrrr_path_to_idx[found_path]
        
        print(f"Linked to {len(self.hrrr_file_paths)}  HRRR files")

    def _parse_date_from_path(self, root: str, fname: str) -> Optional[datetime]:
        """Parse date from file path"""
        date_str = fname.split('.')[0]
        
        # Try YYYYMMDD.nc
        if len(date_str) == 8 and date_str.isdigit():
            try:
                return datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                pass
        
        # Try year/month from directories + DD.nc
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
        
        # Try year from directory + MMDD.nc
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
        """Initialize coordinate cache"""
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
            print(f"Coordinate cache initialization failed: {e}")
            self.coords_cache = {'level': None, 'latitude': None, 'longitude': None}

    # ----------------------------------------
    # Multiprocess-safe file cache management
    # ----------------------------------------
    def _ensure_cache(self):
        """Ensure file cache is initialized in the current process"""
        pid = os.getpid()
        if self._worker_pid != pid:
            # New process: re-init cache (OrderedDict LRU)
            self._era5_cache = OrderedDict()
            self._hrrr_cache = OrderedDict()
            self._worker_pid = pid

    def _get_era5_dataset(self, file_idx: int) -> Optional[xr.Dataset]:
        """
        Get or open ERA5 Dataset (LRU cache)
        
        First access opens file; later reads use cache
        True LRU via OrderedDict.move_to_end()
        """
        self._ensure_cache()
        
        # On cache hit, move entry to end (most recently used)
        if file_idx in self._era5_cache:
            self._era5_cache.move_to_end(file_idx)
            return self._era5_cache[file_idx]
        
        # LRU: evict front item when cache is full
        if len(self._era5_cache) >= ERA5_CACHE_SIZE:
            oldest_idx, oldest_ds = self._era5_cache.popitem(last=False)
            try:
                oldest_ds.close()
            except:
                pass
        
        # Open new file
        file_path = self.file_paths[file_idx]
        for attempt in range(3):
            try:
                ds = xr.open_dataset(file_path)
                self._era5_cache[file_idx] = ds  # New entry appended at end
                return ds
            except Exception as e:
                if attempt == 2:
                    print(f"Cannot open ERA5 file {file_path}: {e}")
                    return None
                time.sleep(0.1 * (2 ** attempt))
        
        return None

    def _get_hrrr_handle(self, file_idx: int) -> Optional[h5py.File]:
        """
        Get or open HRRR file handle (LRU cache)
        
        True LRU via OrderedDict.move_to_end()
        """
        self._ensure_cache()
        
        # On cache hit, move entry to end (most recently used)
        if file_idx in self._hrrr_cache:
            f = self._hrrr_cache[file_idx]
            try:
                if f.id.valid:
                    self._hrrr_cache.move_to_end(file_idx)
                    return f
            except:
                pass
            # Invalid handle, removing
            del self._hrrr_cache[file_idx]
        
        # LRU: evict front item when cache is full
        if len(self._hrrr_cache) >= HRRR_CACHE_SIZE:
            oldest_idx, oldest_f = self._hrrr_cache.popitem(last=False)
            try:
                oldest_f.close()
            except:
                pass
        
        # Open new file
        path = self.hrrr_file_paths[file_idx]
        for attempt in range(3):
            try:
                f = h5py.File(path, 'r', libver='latest')
                self._hrrr_cache[file_idx] = f  # New entry appended at end
                return f
            except Exception as e:
                if attempt == 2:
                    print(f"Cannot open HRRR file {path}: {e}")
                    return None
                time.sleep(0.1 * (2 ** attempt))
        
        return None

    # ----------------------------------------
    # Data reading
    # ----------------------------------------
    def _read_era5_data(self, start_idx: int, length: int) -> dict:
        """
        Read ERA5 data (xarray + cache)
        """
        slice_info = self.timeline[start_idx:start_idx + length]
        if not slice_info:
            return {}
        
        # Group by file: [(file_idx, start_hour, end_hour), ...]
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
        
        # Read data through cache
        buffer = {v: [] for v in self.input_vars}
        
        for file_idx, start, end in tasks:
            ds = self._get_era5_dataset(file_idx)
            if ds is None:
                continue
            
            try:
                for var in self.input_vars:
                    if var in ds:
                        # Slice by time index with isel, then convert to numpy
                        data = ds[var].isel(time=slice(start, end)).values
                        buffer[var].append(data)
            except Exception as e:
                print(f"Failed to read ERA5 data (file_idx={file_idx}): {e}")
                continue
        
        # Concatenate all chunks
        result = {}
        for var, data_list in buffer.items():
            if data_list:
                result[var] = np.concatenate(data_list, axis=0)
        
        return result

    def _read_hrrr_data(self, start_idx: int, length: int) -> Optional[np.ndarray]:
        """Read HRRR data (h5py + cache)"""
        slice_info = self.timeline[start_idx:start_idx + length]
        if not slice_info:
            return None
        
        # Group by file
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
        
        # Read data
        data_list = []
        task_info = []  # Record provenance for each chunk
        
        for file_idx, start, end in tasks:
            if file_idx is None:
                # Raise when data is missing
                ts = slice_info[0]['timestamp']
                raise ValueError(f"Missing HRRR file near time {ts} no matching HRRR data")
            
            file_path = self.hrrr_file_paths[file_idx]
            f = self._get_hrrr_handle(file_idx)
            if f is None:
                raise ValueError(f"Cannot open HRRR file: {file_path}")
            
            try:
                data = f['fields'][start:end]
                
                # Check for abnormal shape
                if data.ndim != 4:
                    raise ValueError(
                        f"Abnormal HRRR shape!\n"
                        f"  file: {file_path}\n"
                        f"  Slice: [{start}:{end}]\n"
                        f"  Returned shape: {data.shape} (expected 4D)\n"
                        f"  Full fields shape: {f['fields'].shape}"
                    )
                
                data_list.append(data)
                task_info.append(f"[read] {os.path.basename(file_path)}, hours={start}:{end}, shape={data.shape}")
                    
            except ValueError:
                raise  # Re-raise ValueError
            except Exception as e:
                raise ValueError(f"Failed to read HRRR file: {file_path}, Error: {e}")
        
        # Check dimension consistency before concat
        if data_list:
            dims = [arr.ndim for arr in data_list]
            if len(set(dims)) > 1:
                print(f"\nHRRR chunk dimensions inconsistent; cannot concatenate!")
                print(f"   start_idx={start_idx}, length={length}")
                print(f"   Chunk dimensions: {dims}")
                for i, (arr, info) in enumerate(zip(data_list, task_info)):
                    print(f"   [{i}] shape={arr.shape}, {info}")
                raise ValueError(f"Inconsistent HRRR dimensions: {dims}")
        
        return np.concatenate(data_list, axis=0) if data_list else None

    def _wrap_as_xarray(self, data_dict: dict, start_time: datetime, length: int) -> xr.Dataset:
        """Wrap NumPy arrays as xarray Dataset"""
        if not data_dict:
            raise ValueError("Data dictionary is empty")
        
        # Check actual data length
        sample = next(iter(data_dict.values()))
        actual_length = sample.shape[0]
        
        # Raise if data is incomplete for upstream handling
        if actual_length != length:
            raise ValueError(f"Data length mismatch: expected {length}, got {actual_length}")
        
        # Ensure all variables share the same time dimension
        for name, data in data_dict.items():
            if data.shape[0] != actual_length:
                raise ValueError(f"Variable {name} time dimension mismatch: {data.shape[0]} vs {actual_length}")
        
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

    # ----------------------------------------
    # Dataset interface
    # ----------------------------------------
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Fetch one sample; return None on corrupt data"""
        try:
            global_idx = self.valid_indices[idx]
            start_time = self.timeline[global_idx]['timestamp']
            
            # Read ERA5 (xarray + cache)
            era5_data = self._read_era5_data(global_idx, self.predict_lead_time)
            era5_xr = self._wrap_as_xarray(era5_data, start_time, self.predict_lead_time)
            
            # Read HRRR (h5py + cache; raises if missing)
            hrrr_data = self._read_hrrr_data(global_idx, self.predict_lead_time)
            
            time_str = start_time.strftime("%Y/%m/%d/%H")
            return era5_xr, time_str, hrrr_data
            
        except Exception as e:
            # On corrupt data, warn and return None (filtered by collate_fn)
            print(f"Skipping sample {idx} (time: {self.timeline[self.valid_indices[idx]]['timestamp']}): {e}")
            return None

    def __del__(self):
        """Close file caches"""
        # Close ERA5 xarray datasets
        if self._era5_cache:
            for ds in self._era5_cache.values():
                try:
                    ds.close()
                except:
                    pass
        
        # Close HRRR h5py files
        if self._hrrr_cache:
            for f in self._hrrr_cache.values():
                try:
                    f.close()
                except:
                    pass


# ============================================
# Collate function
# ============================================
def xarray_collate_fn(batch):
    """
    Custom collate function
    
    Returns:
        input_list: List[xarray.Dataset]
        time_list: List[str]
        hrrr_tensor: torch.Tensor, shape (B, T, C, H, W)
    """
    # Filter None entries
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


# ============================================
# Time alignment verification
# ============================================
def verify_time_alignment(dataset: ERA5XarrayDataset, samples_per_year: int = 3):
    """
    Verify time alignment
    
    Checks:
    1. ERA5 file times vs timeline assumption
    2. Actual hour count per file
    3. HRRR file dimensions
    """
    import random
    from collections import defaultdict
    
    print("\n" + "=" * 60)
    print("Time alignment verification")
    print("=" * 60)
    
    # Group samples by year
    samples_by_year = defaultdict(list)
    for idx, item in enumerate(dataset.timeline):
        year = item['timestamp'].year
        samples_by_year[year].append(idx)
    
    print(f"\nTimeline records by year:")
    for year in sorted(samples_by_year.keys()):
        print(f"  {year}: {len(samples_by_year[year])}  records")
    
    # Sample from years 2021, 2022, 2023
    target_years = [2021, 2022, 2023]
    errors = []
    warnings = []
    
    for year in target_years:
        if year not in samples_by_year:
            print(f"\n[skip] {year}  has no data")
            continue
        
        # Random samples from that year
        year_samples = samples_by_year[year]
        sample_indices = random.sample(year_samples, min(samples_per_year, len(year_samples)))
        
        print(f"\n--- {year}  validation ( {len(sample_indices)}  samples) ---")
        
        for timeline_idx in sample_indices:
            item = dataset.timeline[timeline_idx]
            file_idx = item['file_idx']
            hour_index = item['hour_index']
            expected_time = item['timestamp']
            file_path = dataset.file_paths[file_idx]
            
            print(f"\n  Sample timeline[{timeline_idx}]:")
            print(f"    file: {os.path.basename(file_path)}")
            print(f"    Expected time: {expected_time} (hour_index={hour_index})")
            
            # Check actual ERA5 file times
            try:
                ds = xr.open_dataset(file_path)
                actual_times = ds.time.values
                num_hours = len(actual_times)
                
                print(f"    Actual time steps in file: {num_hours}")
                
                if num_hours != 24:
                    msg = f"file {os.path.basename(file_path)} time steps={num_hours} (expected 24)"
                    warnings.append(msg)
                    print(f"    Warning: {msg}")
                
                if hour_index < num_hours:
                    # Actual time at index
                    actual_time = np.datetime64(actual_times[hour_index], 'us').astype('datetime64[s]')
                    actual_dt = datetime.utcfromtimestamp(actual_time.astype('int64'))
                    
                    print(f"    Actual time[{hour_index}]: {actual_dt}")
                    
                    # Compare
                    if actual_dt != expected_time:
                        msg = f"Time mismatch! expected={expected_time}, got={actual_dt}"
                        errors.append(msg)
                        print(f"    Error: {msg}")
                    else:
                        print(f"    Time aligned")
                else:
                    msg = f"hour_index={hour_index} out of range (file has only{num_hours} hours)"
                    errors.append(msg)
                    print(f"    Error: {msg}")
                
                # file time range
                first_time = np.datetime64(actual_times[0], 'us').astype('datetime64[s]')
                last_time = np.datetime64(actual_times[-1], 'us').astype('datetime64[s]')
                first_dt = datetime.utcfromtimestamp(first_time.astype('int64'))
                last_dt = datetime.utcfromtimestamp(last_time.astype('int64'))
                print(f"    file time range: {first_dt} ~ {last_dt}")
                
                ds.close()
                
            except Exception as e:
                errors.append(f"Cannot read file {file_path}: {e}")
                print(f"    Read failed: {e}")
            
            # Check HRRR file (if present)
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
                            print(f"    Warning: {msg}")
                except Exception as e:
                    print(f"    HRRR read failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification summary")
    print("=" * 60)
    print(f"  Error: {len(errors)}")
    print(f"  Warning: {len(warnings)}")
    
    if errors:
        print("\nError list:")
        for e in errors[:10]:  # show at most 10
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... more {len(errors) - 10}  errors")
    
    if warnings:
        print("\nWarning list:")
        for w in warnings[:10]:
            print(f"  - {w}")
        if len(warnings) > 10:
            print(f"  ... more {len(warnings) - 10}  warnings")
    
    if not errors and not warnings:
        print("\nAll sampled checks passed; time alignment OK!")
    
    return len(errors) == 0


# ============================================
# Test
# ============================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent
    AERO_ODE_ROOT = PROJECT_ROOT.parent
    if os.fspath(AERO_ODE_ROOT) not in sys.path:
        sys.path.insert(0, os.fspath(AERO_ODE_ROOT))
    import paths_config as pc

    ROOT_DIR = os.fspath(pc.era5_root())
    HRRR_DIRS = [os.fspath(pc.hrrr_truth_root())]
    
    if os.path.exists(ROOT_DIR):
        dataset = ERA5XarrayDataset(
            ROOT_DIR,
            hrrr_root_dirs=HRRR_DIRS if os.path.exists(HRRR_DIRS[0]) else None,
            predict_lead_time=48
        )
        
        # Time alignment verification
        verify_time_alignment(dataset, samples_per_year=5)
        
        # Data loading test
        print("\n" + "=" * 60)
        print("Data loading test")
        print("=" * 60)
        
        loader = DataLoader(
            dataset, batch_size=2, num_workers=2,
            persistent_workers=True, collate_fn=xarray_collate_fn
        )
        
        for i, (era5, times, hrrr) in enumerate(loader):
            if era5 is None:
                continue
            print(f"Batch {i}: ERA5 vars={list(era5[0].data_vars.keys())[:3]}, "
                  f"Times={times}, HRRR shape={hrrr.shape if hrrr is not None else 'None'}")
            if i >= 2:
                break
    else:
        print(f"Path does not exist: {ROOT_DIR}")
