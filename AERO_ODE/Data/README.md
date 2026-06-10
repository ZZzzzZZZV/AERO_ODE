# Data

由于 ERA5 / HRRR 原始数据文件较大(单个 0.4~0.8 GB,超过 GitHub 仓库文件限制),
大文件不直接存放在仓库中,而是作为 **GitHub Release 资源** 提供下载。

Large ERA5 / HRRR data files (0.4–0.8 GB each) are not stored in the git tree.
They are published as **GitHub Release assets** and must be downloaded separately.

## 下载地址 / Download

Release: [`data-v1`](https://github.com/ZZzzzZZZV/AERO_ODE/releases/tag/data-v1)

下载后请按下表把文件放到对应路径(注意需要重命名)。
After downloading, place each file at the path below (rename as indicated).

| Release 资源名 (asset) | 放置路径 (target path) |
| --- | --- |
| `ERA5_2024_08_01.nc` | `Data/ERA5/2024/08/01/20240801.nc` |
| `ERA5_2024_08_02.nc` | `Data/ERA5/2024/08/02/20240802.nc` |
| `ERA5_2024_08_03.nc` | `Data/ERA5/2024/08/03/20240803.nc` |
| `ERA5_2024_08_04.nc` | `Data/ERA5/2024/08/04/20240804.nc` |
| `HRRR_2024_08_01.h5` | `Data/HRRR/2024/08/01.h5` |
| `HRRR_2024_08_02.h5` | `Data/HRRR/2024/08/02.h5` |
| `HRRR_2024_08_03.h5` | `Data/HRRR/2024/08/03.h5` |
| `HRRR_2024_08_04.h5` | `Data/HRRR/2024/08/04.h5` |

## 命令行下载示例 / Download via CLI

```bash
BASE=https://github.com/ZZzzzZZZV/AERO_ODE/releases/download/data-v1

# ERA5
curl -L "$BASE/ERA5_2024_08_01.nc" -o Data/ERA5/2024/08/01/20240801.nc
curl -L "$BASE/ERA5_2024_08_02.nc" -o Data/ERA5/2024/08/02/20240802.nc
curl -L "$BASE/ERA5_2024_08_03.nc" -o Data/ERA5/2024/08/03/20240803.nc
curl -L "$BASE/ERA5_2024_08_04.nc" -o Data/ERA5/2024/08/04/20240804.nc

# HRRR
curl -L "$BASE/HRRR_2024_08_01.h5" -o Data/HRRR/2024/08/01.h5
curl -L "$BASE/HRRR_2024_08_02.h5" -o Data/HRRR/2024/08/02.h5
curl -L "$BASE/HRRR_2024_08_03.h5" -o Data/HRRR/2024/08/03.h5
curl -L "$BASE/HRRR_2024_08_04.h5" -o Data/HRRR/2024/08/04.h5
```

> 仓库中已包含的小文件(HRRR 的 `geo.h5`、`lats.npy`、`lons.npy`、`stat/` 等,
> 以及 `Code_for_processing_data/` 数据处理脚本)无需另外下载。
