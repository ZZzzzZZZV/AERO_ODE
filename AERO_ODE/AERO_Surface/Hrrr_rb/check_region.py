"""Simple check of HRRR regional grid extent."""
import numpy as np

lats = np.load('./lats.npy')
lons = np.load('./lons.npy')

print(f"Grid shape: {lats.shape}")
print(f"Latitude range: {lats.min():.4f} ~ {lats.max():.4f}")
print(f"Longitude range: {lons.min():.4f} ~ {lons.max():.4f}")

# Bounding box
lat_min, lat_max = lats.min() - 0.1, lats.max() + 0.1
lon_min, lon_max = lons.min() - 0.1, lons.max() + 0.1

print(f"\nBounding box for download:")
print(f"  lon_min={lon_min:.4f}, lat_min={lat_min:.4f}")
print(f"  lon_max={lon_max:.4f}, lat_max={lat_max:.4f}")
