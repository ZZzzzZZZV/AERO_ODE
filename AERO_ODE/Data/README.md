# Data

This folder holds the small auxiliary files needed at runtime (already committed):

- `HRRR/lats.npy`, `HRRR/lons.npy`, `HRRR/geo.h5` — grid / geolocation files
- `HRRR/stat/` — normalization statistics
- `Code_for_processing_data/` — preprocessing scripts

The larger sample data (ERA5 `.nc` and HRRR `.h5`) is **not** stored here.

## Where to get the data

- **Minimal runtime dataset** (`Data.zip`, split into two parts):
  [`data-v1` release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/data-v1).
  Download both parts, reassemble, and unzip into `AERO_ODE/`:

  ```bash
  BASE=https://github.com/ZZzzzZZZV/AERO_ODE/releases/download/data-v1
  curl -L "$BASE/Data.zip.part00" -o Data.zip.part00
  curl -L "$BASE/Data.zip.part01" -o Data.zip.part01
  cat Data.zip.part00 Data.zip.part01 > Data.zip
  unzip Data.zip -d AERO_ODE/
  ```

- **One-month full dataset**:
  [AeroODE Case Data on Zenodo](https://doi.org/10.5281/zenodo.20602695).

See the repository root `README.md` (Data Preparation / Weights Preparation) for full details.
