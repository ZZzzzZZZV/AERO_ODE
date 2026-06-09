import numpy as np

lats = np.load("lats.npy")
lons = np.load("lons.npy")

print(f"Grid shape: {lats.shape}")
print(f"Latitude range: {lats.min():.4f} ~ {lats.max():.4f}")
print(f"Longitude range: {lons.min():.4f} ~ {lons.max():.4f}")

# Bounding box for downloads
lat_min, lat_max = lats.min(), lats.max()
lon_min, lon_max = lons.min(), lons.max()
print(f"\nDownload bounding box:")
print(f"  lat: [{lat_min:.2f}, {lat_max:.2f}]")
print(f"  lon: [{lon_min:.2f}, {lon_max:.2f}]")
