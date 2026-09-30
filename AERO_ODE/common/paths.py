from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def hrrr_daily_relpaths(when: datetime) -> list[str]:
    return [
        f"{when.year}/{when.month:02d}/{when.day:02d}.h5",
        f"{when.month:02d}/{when.day:02d}.h5",
        f"{when.day:02d}.h5",
    ]

def _year_tokens(path: Path) -> set[str]:
    return {part for part in path.parts if part.isdigit() and len(part) == 4}

def hrrr_root_covers_year(root, year) -> bool:
    year_str = str(year)
    raw = Path(root)
    tokens = _year_tokens(raw)
    try:
        tokens |= _year_tokens(raw.resolve())
    except OSError:
        pass
    if tokens:
        return year_str in tokens
    return True

def find_hrrr_daily_file(roots, when: datetime):
    for root in roots:
        if not hrrr_root_covers_year(root, when.year):
            continue
        base = Path(root)
        for rel in hrrr_daily_relpaths(when):
            candidate = base / rel
            if candidate.exists():
                return candidate
    return None

def era5_daily_nc(root, year: int, month: int, day: int) -> Path:
    return (
        Path(root)
        / f"{year:04d}"
        / f"{month:02d}"
        / f"{day:02d}"
        / f"{year:04d}{month:02d}{day:02d}.nc"
    )

def iter_era5_daily_files(root):
    root = Path(root)
    if not root.exists():
        return
    try:
        year_dirs = sorted(root.iterdir())
    except OSError:
        return
    for year_dir in year_dirs:
        if not year_dir.name.isdigit() or len(year_dir.name) != 4:
            continue
        year = int(year_dir.name)
        try:
            month_dirs = sorted(year_dir.iterdir())
        except OSError:
            continue
        for month_dir in month_dirs:
            if not month_dir.name.isdigit() or len(month_dir.name) != 2:
                continue
            month = int(month_dir.name)
            for day in range(1, 32):
                nc = era5_daily_nc(root, year, month, day)
                if nc.is_file():
                    yield nc, datetime(year, month, day)

def add_repo_to_syspath() -> Path:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return ROOT
