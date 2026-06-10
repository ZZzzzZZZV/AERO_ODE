# AERO-ODE

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-0.4-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey?logo=linux&logoColor=white)

## A Physics Guided Integrated Global to Regional Method for High Resolution Weather Forecasting

`AERO-ODE` is a physics-guided global-to-regional weather forecasting framework for fast, high-resolution regional prediction. Starting from global initial conditions, the model generates hourly 72 h regional forecasts at 3 km resolution, covering pressure-level variables and near-surface variables without requiring separate global surface lateral-boundary inputs.

## Model Architecture

**AERO-AIR: pressure-level variable prediction framework**

![AERO-AIR pressure-level variable prediction framework](./assets/AERO_ODE_AIR.png)

**AERO-Surface: surface-variable prediction framework**

![AERO-Surface surface-variable prediction framework](./assets/AERO_ODE_Surface.png)

## Visualization

**MSLP forecast comparison (initialized at 00 UTC on 1 January 2024)**

https://github.com/user-attachments/assets/885a7942-2d5e-4a63-a549-377119f7c7c7

**T2m forecast comparison (initialized at 00 UTC on 1 May 2024)**

https://github.com/user-attachments/assets/9661c7b1-1434-4647-b24f-9b25afdaa272

<sub>Note: These visualizations do not imply that AERO-ODE outperforms global models overall; they are intended only to illustrate its additional regional gains over the target domain.</sub>

## Environment Setup

### Method 1: Use the packed environment (preferred)

Because `AERO-ODE` depends on both JAX and PyTorch, which have strict version compatibility requirements, we provide a packed backup of the authors' virtual environment. Follow the steps below to restore the runtime environment.

#### Option A: Download from GitHub Release

The packed environment (`NeuralGCM_LAM_Env.tar.gz`, ~2.98 GB) is published as a GitHub Release asset. Because a single release asset is limited to 2 GB, it is split into two parts. Download both parts from the [`env-v1` release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/env-v1) and reassemble them:

```bash
cat NeuralGCM_LAM_Env.tar.gz.part00 NeuralGCM_LAM_Env.tar.gz.part01 > NeuralGCM_LAM_Env.tar.gz
# Optional integrity check (SHA-256):
# 375604987cbb299130e5eefb317dd4985ca730155837b06be9faefb49ad51552
sha256sum NeuralGCM_LAM_Env.tar.gz
```

#### Option B: Download from Zenodo

The same packed environment is also included in the [AeroODE Case Data release on Zenodo](https://doi.org/10.5281/zenodo.20602695). Download the archive from Zenodo and extract `NeuralGCM_LAM_Env.tar.gz` before continuing with the steps below. File access may require logging in to Zenodo if the record is restricted.

#### Step 1: Find your conda envs directory

```bash
conda env list
```

For example:

```bash
/home/zhangjing09/miniconda3/envs/
```

#### Step 2: Create the target directory for the unpacked environment

```bash
mkdir -p /home/zhangjing09/miniconda3/envs/NeuralGCM_LAM
```

#### Step 3: Extract the archive

Download the packed environment using **Option A** or **Option B** above, then extract it:

```bash
tar -xzvf NeuralGCM_LAM_Env.tar.gz -C /home/zhangjing09/miniconda3/envs/NeuralGCM_LAM
```

Please replace `/home/zhangjing09/miniconda3/envs/` with your own conda environment path.

#### Step 4: Activate and fix paths

Run `conda-unpack` once after the first activation to fix hard-coded paths in the packed environment.

```bash
source /home/zhangjing09/miniconda3/envs/NeuralGCM_LAM/bin/activate
conda-unpack
```

#### Step 5: Verify the installation

```bash
python -c "import jax; print('jax:', jax.__version__)"      # jax: 0.4.29
python -c "import torch; print('torch:', torch.__version__)"  # torch: 2.4.0+cu121
```

#### Daily activation

```bash
conda activate NeuralGCM_LAM
```

### Notes

| Topic | Details |
|---|---|
| conda-unpack | Run once after the first activation to fix hard-coded paths. |
| Cross-platform | Not supported. A Linux pack only works on Linux. |
| tar warnings | A few file warnings are acceptable if key packages import correctly. |


### Method 2: Create the environment manually

Install the required versions manually from `environment.yml`. This path is not recommended unless the packed environment cannot be used.

```bash
conda env create -f environment.yml -n NeuralGCM_LAM
```

## Downloads (GitHub Releases)

Large files (sample data, model checkpoints, and the packed environment) are **not** stored directly in the git tree. They are published as **GitHub Release** assets. Small auxiliary files (grid files, statistics, processing scripts) are already included in the repository. The table below summarizes what each asset is for, where to get it, and where to put it.

| Asset (release) | What it is / used for | Download | Place at |
|---|---|---|---|
| `ERA5_2024_08_0X.nc` ([`data-v1`](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/data-v1)) | ERA5 global reanalysis on pressure levels, used as the **global initial conditions / forcing** input. One file per day (2024-08-01 … 04), ~795 MB each. | [data-v1 release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/data-v1) | `AERO_ODE/Data/ERA5/2024/08/0X/2024080X.nc` |
| `HRRR_2024_08_0X.h5` ([`data-v1`](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/data-v1)) | HRRR high-resolution (3 km) regional analysis, used as the **regional target / ground truth** for training and verification. One file per day (2024-08-01 … 04), ~395 MB each. | [data-v1 release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/data-v1) | `AERO_ODE/Data/HRRR/2024/08/0X.h5` |
| `AERO_AIR_model_ep5.pth` ([`checkpoints-v1`](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/checkpoints-v1)) | Trained **AERO-AIR** checkpoint (pressure-level prediction), needed for inference with `AERO_AIR/test_film_rb.py` etc. ~837 MB. | [checkpoints-v1 release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/checkpoints-v1) | `AERO_ODE/AERO_AIR/checkpoints_film/model_ep5.pth` |
| `AERO_Surface_model_ep5.pth` ([`checkpoints-v1`](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/checkpoints-v1)) | Trained **AERO-Surface** checkpoint (near-surface prediction), needed for inference with `AERO_Surface/test_film_00z_RMSE.py` etc. ~837 MB. | [checkpoints-v1 release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/checkpoints-v1) | `AERO_ODE/AERO_Surface/checkpoints_film_v2/model_ep5.pth` |
| `NeuralGCM_LAM_Env.tar.gz.part0X` ([`env-v1`](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/env-v1)) | Packed conda runtime environment (JAX + PyTorch), ~2.98 GB split into 2 parts. See [Environment Setup](#environment-setup). | [env-v1 release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/env-v1) | reassemble, then unpack into your conda envs dir |

> Each empty data/checkpoint directory in the repo contains a `DOWNLOAD_FROM_RELEASE.txt` file noting exactly which asset to download and the target filename. See also [`AERO_ODE/Data/README.md`](AERO_ODE/Data/README.md) for copy-paste download commands.

### Quick download (ERA5 / HRRR sample data)

```bash
BASE=https://github.com/ZZzzzZZZV/AERO_ODE/releases/download/data-v1
curl -L "$BASE/ERA5_2024_08_01.nc" -o AERO_ODE/Data/ERA5/2024/08/01/20240801.nc
curl -L "$BASE/HRRR_2024_08_01.h5" -o AERO_ODE/Data/HRRR/2024/08/01.h5
# ... repeat for 02, 03, 04
```

### Quick download (model checkpoints)

```bash
CK=https://github.com/ZZzzzZZZV/AERO_ODE/releases/download/checkpoints-v1
curl -L "$CK/AERO_AIR_model_ep5.pth"     -o AERO_ODE/AERO_AIR/checkpoints_film/model_ep5.pth
curl -L "$CK/AERO_Surface_model_ep5.pth" -o AERO_ODE/AERO_Surface/checkpoints_film_v2/model_ep5.pth
```

## Data Preparation

This repository provides two types of datasets:

- **Minimal runtime dataset**: small auxiliary files (HRRR grid files `lats.npy` / `lons.npy` / `geo.h5`, normalization statistics under `HRRR/stat/`, and the preprocessing scripts in `Data/Code_for_processing_data/`) are bundled directly in the repository; no extra download is required for these.
- **One-month sample dataset (ERA5 + HRRR, 2024-08-01 … 04)**: the large `.nc` / `.h5` files are distributed via the [`data-v1` release](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/data-v1) (see the download table above). The same data and weights are also mirrored on Zenodo — [AeroODE Case Data](https://doi.org/10.5281/zenodo.20602695) — together with paper figures and the packed environment.

To expand the dataset, raw data can be obtained from the following official sources:

- HRRR: [https://rapidrefresh.noaa.gov/hrrr/](https://rapidrefresh.noaa.gov/hrrr/)
- ERA5: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

After downloading, apply bilinear interpolation using the latitude/longitude grid files provided in the `HRRR` folder to generate the model input format.

## Model Inference

### AERO-AIR

Run the following under the `AERO_AIR` directory:

```bash
python test_film_rb.py
```

Runs pressure-level variable prediction and computes RMSE; outputs and error curves are saved to `Test_Data_Rmse_00z`.

```bash
python Generate_Yearly_Forecast_rb.py
```

Generates full-year forecast data.

### AERO-Surface

Run the following under the `AERO_Surface` directory:

```bash
python test_film_00z_RMSE.py
```

Runs near-surface variable prediction and computes RMSE; outputs and error curves are saved to `Test_Surface_Rmse_00z`.

```bash
python Generate_Yearly_Forecast_Surface.py
```

Generates full-year near-surface forecast data.

### Result Visualization

Run the following under `AERO_ODE_Draw/Draw_New`:

```bash
python run_all_drawings.py
```

Automatically generates all curve figures used in the paper and saves them to the `Draw_New` directory.

## License

AERO-ODE code and model weights are released by Shanghai Zhangjiang Institute of Mathematics.

Commercial use of these AERO-ODE models is prohibited.

## Third-Party Components

Modified NeuralGCM source is included under Apache-2.0. See `AERO_ODE/AERO_Surface/neuralgcm-main/LICENSE`.

NeuralGCM pretrained weights (if distributed) are under CC-BY-SA-4.0.

Dinosaur — Apache-2.0 (if included).
