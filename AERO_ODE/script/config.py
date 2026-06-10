from pathlib import Path

import paths_config as pc

AERO_ODE_ROOT = pc.AERO_ODE_ROOT
AIR_ROOT = AERO_ODE_ROOT / "AERO_AIR"
SURFACE_ROOT = AERO_ODE_ROOT / "AERO_Surface"
OUTPUT_ROOT = AERO_ODE_ROOT / "quickstart_outputs"

FORECAST_HOURS = 72
TIME_STEPS = FORECAST_HOURS + 1
YEAR = pc.DEMO_YEAR
MONTH = pc.DEMO_MONTH
DAY = 1

PLOT_LEADS = [6, 12, 24, 48, 72]
SURFACE_VARS = {"mslp": 0, "t2m": 3}
