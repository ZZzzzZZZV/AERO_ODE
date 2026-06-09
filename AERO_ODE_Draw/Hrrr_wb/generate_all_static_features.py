"""
AERO-ODE surface static feature batch generator

Features:
    Generate all surface static features needed for training/inference from raw inputs

Input files:
    1. geo.h5              - DEM Elevation, shape (1, H, W)
    2. MODIS_LandCover_HRRR.tif - MODIS Land-cover GeoTIFF
    3. lats.npy            - Latitude grid, shape (H, W)
    4. lons.npy            - Longitude grid, shape (H, W)

Output files:
    Terrain features:
        - slope.npy              - Slope (normalized)
        - aspect_sin.npy         - Aspect sin component [-1, 1]
        - aspect_cos.npy         - Aspect cos component [-1, 1]
        - curvature.npy          - Curvature (normalized)
        - tpi.npy                - TPI (normalized)
        - terrain_roughness.npy  - Terrain roughness (normalized)
    
    Land-cover features:
        - landcover.npy          - Land-cover class (0-17, IGBP)
        - roughness_z0.npy       - Surface roughness length (m)
        - roughness_log_z0.npy   - log(roughness length)
        - roughness_log_z0_norm.npy - Normalized log(roughness length)
    
    Mask layers:
        - water_mask.npy         - Water mask (class 0)
        - urban_mask.npy         - Urban mask (class 13)
        - forest_mask.npy        - Forest mask (classes 1-5)
        - cropland_mask.npy      - Cropland mask (classes 12, 14)
        - grassland_mask.npy     - Grassland mask (classes 6-10)
    
    Combined file:
        - static_features.h5     - Bundle of all features above

Usage:
    python generate_all_static_features.py

    Or specify paths:
    python generate_all_static_features.py --input_dir ./data --output_dir ./data

"""

import os
import sys
import argparse
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import h5py

# ============================================
# Config
# ============================================

# IGBP land-cover classes
IGBP_CLASSES = {
    0: "Water Bodies",
    1: "Evergreen Needleleaf Forest",
    2: "Evergreen Broadleaf Forest", 
    3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest",
    5: "Mixed Forests",
    6: "Closed Shrublands",
    7: "Open Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-up",
    14: "Cropland/Natural Mosaic",
    15: "Snow and Ice",
    16: "Barren",
    17: "Unclassified"
}

# Roughness length lookup (meters)
Z0_TABLE = {
    0: 0.0001,   # Water Bodies - very smooth
    1: 1.0,      # Evergreen Needleleaf Forest
    2: 1.0,      # Evergreen Broadleaf Forest
    3: 0.8,      # Deciduous Needleleaf Forest
    4: 0.8,      # Deciduous Broadleaf Forest
    5: 0.9,      # Mixed Forests
    6: 0.1,      # Closed Shrublands
    7: 0.05,     # Open Shrublands
    8: 0.5,      # Woody Savannas
    9: 0.15,     # Savannas
    10: 0.03,    # Grasslands
    11: 0.05,    # Permanent Wetlands
    12: 0.1,     # Croplands
    13: 1.5,     # Urban and Built-up - roughest
    14: 0.1,     # Cropland/Natural Vegetation Mosaic
    15: 0.001,   # Snow and Ice
    16: 0.01,    # Barren
    17: 0.03,    # Unclassified
}


# ============================================
# Data loading
# ============================================

def load_input_data(input_dir, geo_file, landcover_file, lats_file, lons_file):
    """Load all input data"""
    
    # 1. Load lat/lon
    lats_path = os.path.join(input_dir, lats_file)
    lons_path = os.path.join(input_dir, lons_file)
    
    if not os.path.exists(lats_path) or not os.path.exists(lons_path):
        raise FileNotFoundError(f"Lat/lon files not found: {lats_path} or {lons_path}")
    
    lats = np.load(lats_path)
    lons = np.load(lons_path)
    print(f"  Lat/lon shape: {lats.shape}")
    print(f"  Latitude range: [{lats.min():.4f}, {lats.max():.4f}]")
    print(f"  Longitude range: [{lons.min():.4f}, {lons.max():.4f}]")
    
    # 2. Load DEM
    geo_path = os.path.join(input_dir, geo_file)
    if not os.path.exists(geo_path):
        raise FileNotFoundError(f"DEM file not found: {geo_path}")
    
    with h5py.File(geo_path, 'r') as f:
        dem = f['fields'][0]  # (H, W)
    print(f"  DEM shape: {dem.shape}")
    print(f"  DEM range: [{dem.min():.2f}, {dem.max():.2f}] m")
    
    # 3. Check MODIS GeoTIFF
    landcover_path = os.path.join(input_dir, landcover_file)
    if not os.path.exists(landcover_path):
        raise FileNotFoundError(f"MODIS land-cover file not found: {landcover_path}")
    
    return lats, lons, dem, landcover_path


# ============================================
# Terrain features
# ============================================

def compute_cell_size(lats, lons):
    """Compute grid cell size (meters)"""
    lat_center = lats.mean()
    
    dlat = np.abs(np.diff(lats, axis=0)).mean()
    dlon = np.abs(np.diff(lons, axis=1)).mean()
    
    dy = dlat * 111000
    dx = dlon * 111000 * np.cos(np.radians(lat_center))
    
    print(f"  Grid resolution: dx={dx:.1f}m, dy={dy:.1f}m")
    return dx, dy


def compute_slope_aspect(dem, dx, dy):
    """Compute slope and aspect"""
    dz_dx = np.zeros_like(dem)
    dz_dy = np.zeros_like(dem)
    
    # x gradient (longitude)
    dz_dx[:, 1:-1] = (dem[:, 2:] - dem[:, :-2]) / (2 * dx)
    dz_dx[:, 0] = (dem[:, 1] - dem[:, 0]) / dx
    dz_dx[:, -1] = (dem[:, -1] - dem[:, -2]) / dx
    
    # y gradient (latitude)
    dz_dy[1:-1, :] = (dem[2:, :] - dem[:-2, :]) / (2 * dy)
    dz_dy[0, :] = (dem[1, :] - dem[0, :]) / dy
    dz_dy[-1, :] = (dem[-1, :] - dem[-2, :]) / dy
    
    # Slope (radians)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    
    # Aspect (radians, 0=north, clockwise)
    aspect = np.arctan2(-dz_dx, dz_dy)
    aspect = np.where(aspect < 0, aspect + 2*np.pi, aspect)
    
    # Return sin/cos components
    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)
    
    print(f"  Slope range: [{np.degrees(slope.min()):.2f}°, {np.degrees(slope.max()):.2f}°]")
    return slope, aspect_sin, aspect_cos


def compute_curvature(dem, dx, dy):
    """Compute terrain curvature"""
    d2z_dx2 = np.zeros_like(dem)
    d2z_dy2 = np.zeros_like(dem)
    
    d2z_dx2[:, 1:-1] = (dem[:, 2:] - 2*dem[:, 1:-1] + dem[:, :-2]) / (dx**2)
    d2z_dy2[1:-1, :] = (dem[2:, :] - 2*dem[1:-1, :] + dem[:-2, :]) / (dy**2)
    
    curvature = d2z_dx2 + d2z_dy2
    curvature = np.clip(curvature, -0.01, 0.01)
    
    print(f"  Curvature range: [{curvature.min():.6f}, {curvature.max():.6f}]")
    return curvature


def compute_tpi(dem, window_size=5):
    """Compute terrain position index (TPI)"""
    kernel = np.ones((window_size, window_size)) / (window_size**2)
    dem_smooth = ndimage.convolve(dem, kernel, mode='reflect')
    tpi = dem - dem_smooth
    
    print(f"  TPI range: [{tpi.min():.2f}, {tpi.max():.2f}]")
    return tpi


def compute_terrain_roughness(dem, window_size=3):
    """Compute terrain roughness"""
    dem_mean = ndimage.uniform_filter(dem, size=window_size, mode='reflect')
    dem_sq_mean = ndimage.uniform_filter(dem**2, size=window_size, mode='reflect')
    roughness = np.sqrt(np.maximum(dem_sq_mean - dem_mean**2, 0))
    
    print(f"  Terrain roughness range: [{roughness.min():.4f}, {roughness.max():.4f}]")
    return roughness


def normalize_zscore(data):
    """Z-score normalization"""
    mean = data.mean()
    std = data.std()
    return (data - mean) / (std + 1e-6)


def compute_all_terrain_features(dem, lats, lons):
    """Compute all terrain features"""
    print("\n  Computing grid resolution...")
    dx, dy = compute_cell_size(lats, lons)
    
    print("  Compute slope and aspect...")
    slope, aspect_sin, aspect_cos = compute_slope_aspect(dem, dx, dy)
    
    print("  Computing curvature...")
    curvature = compute_curvature(dem, dx, dy)
    
    print("  Compute terrain position index (TPI)...")
    tpi = compute_tpi(dem, window_size=5)
    
    print("  Compute terrain roughness...")
    terrain_roughness = compute_terrain_roughness(dem, window_size=3)
    
    # Normalization
    print("  Normalizing...")
    terrain_features = {
        'slope': normalize_zscore(slope).astype(np.float32),
        'aspect_sin': aspect_sin.astype(np.float32),
        'aspect_cos': aspect_cos.astype(np.float32),
        'curvature': normalize_zscore(curvature).astype(np.float32),
        'tpi': normalize_zscore(tpi).astype(np.float32),
        'terrain_roughness': normalize_zscore(terrain_roughness).astype(np.float32),
    }
    
    return terrain_features


# ============================================
# Land-cover processing
# ============================================

def process_landcover_geotiff(tiff_path, hrrr_lats, hrrr_lons):
    """Process MODIS GeoTIFF and regrid to HRRR"""
    try:
        import rasterio
    except ImportError:
        print("Error: Install rasterio first: pip install rasterio")
        sys.exit(1)
    
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        transform = src.transform
        
        # Build source grid coordinates
        rows, cols = data.shape
        src_lons = np.array([transform.c + transform.a * i for i in range(cols)])
        src_lats = np.array([transform.f + transform.e * i for i in range(rows)])
        
        print(f"  Source shape: {data.shape}")
        print(f"  Source latitude range: [{src_lats.min():.4f}, {src_lats.max():.4f}]")
        print(f"  source longitude range: [{src_lons.min():.4f}, {src_lons.max():.4f}]")
        
        # Ensure latitude ascending
        if src_lats[0] > src_lats[-1]:
            src_lats = src_lats[::-1]
            data = data[::-1, :]
        
        # Nearest-neighbor interpolator (categorical)
        interpolator = RegularGridInterpolator(
            (src_lats, src_lons), 
            data.astype(np.float32),
            method='nearest',
            bounds_error=False,
            fill_value=0
        )
        
        # Regrid to HRRR mesh
        # Lon CRS mismatch: HRRR 0-360 vs MODIS -180-180
        query_lons = hrrr_lons.copy()
        if hrrr_lons.max() > 180 and src_lons.max() <= 180:
            query_lons = np.where(query_lons > 180, query_lons - 360, query_lons)
            print(f"  Lon convert 0-360 -> -180-180 (query [{query_lons.min():.4f}, {query_lons.max():.4f}])")
        elif hrrr_lons.min() < 0 and src_lons.min() >= 0:
            query_lons = np.where(query_lons < 0, query_lons + 360, query_lons)
            print(f"  Lon convert -180-180 -> 0-360 (query [{query_lons.min():.4f}, {query_lons.max():.4f}])")
        points = np.stack([hrrr_lats.ravel(), query_lons.ravel()], axis=-1)
        landcover_hrrr = interpolator(points).reshape(hrrr_lats.shape)
        
        return landcover_hrrr.astype(np.uint8)


def create_roughness_length(landcover_data):
    """Build roughness length from land cover"""
    z0 = np.zeros_like(landcover_data, dtype=np.float32)
    for cls, value in Z0_TABLE.items():
        z0[landcover_data == cls] = value
    return z0


def create_all_masks(landcover_data):
    """Create all masks"""
    masks = {
        'water_mask': (landcover_data == 0).astype(np.float32),
        'urban_mask': (landcover_data == 13).astype(np.float32),
        'forest_mask': ((landcover_data >= 1) & (landcover_data <= 5)).astype(np.float32),
        'cropland_mask': ((landcover_data == 12) | (landcover_data == 14)).astype(np.float32),
        'grassland_mask': ((landcover_data >= 6) & (landcover_data <= 10)).astype(np.float32),
    }
    return masks


def compute_all_landcover_features(landcover_path, lats, lons):
    """Compute all land-cover features"""
    
    print("  Processing GeoTIFF...")
    landcover = process_landcover_geotiff(landcover_path, lats, lons)
    
    print("  Compute roughness length...")
    z0 = create_roughness_length(landcover)
    log_z0 = np.log(z0 + 1e-6)
    log_z0_norm = normalize_zscore(log_z0)
    
    print("  Creating masks...")
    masks = create_all_masks(landcover)
    
    landcover_features = {
        'landcover': landcover,
        'roughness_z0': z0,
        'roughness_log_z0': log_z0.astype(np.float32),
        'roughness_log_z0_norm': log_z0_norm.astype(np.float32),
        **masks
    }
    
    return landcover_features


# ============================================
# Save data
# ============================================

def save_all_features(features, output_dir):
    """Save all features"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nSaving individual files:")
    for name, data in features.items():
        path = os.path.join(output_dir, f'{name}.npy')
        np.save(path, data)
        print(f"  {name}.npy: shape={data.shape}, dtype={data.dtype}")
    
    # Save combined bundle to H5
    print("\nSaving combined file:")
    h5_path = os.path.join(output_dir, 'static_features.h5')
    with h5py.File(h5_path, 'w') as f:
        for name, data in features.items():
            f.create_dataset(name, data=data, compression='gzip')
    print(f"  static_features.h5")


def print_landcover_statistics(landcover):
    """Print land-cover statistics"""
    print("\nLand-cover class distribution:")
    print("-" * 50)
    total = landcover.size
    for cls in sorted(np.unique(landcover)):
        count = (landcover == cls).sum()
        pct = count / total * 100
        name = IGBP_CLASSES.get(cls, "Unknown")
        print(f"  {cls:2d} {name:30s}: {pct:5.2f}%")


def print_mask_statistics(features):
    """Print mask statistics"""
    print("\nMask coverage:")
    print("-" * 50)
    mask_names = ['water_mask', 'forest_mask', 'urban_mask', 'cropland_mask', 'grassland_mask']
    for name in mask_names:
        if name in features:
            pct = features[name].mean() * 100
            print(f"  {name:20s}: {pct:5.2f}%")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='AERO-ODE surface static feature batch generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input files:
  west_geo_interpolate.h5        DEM elevation (HDF5, dataset 'fields')
  MODIS_LandCover_HRRR_West.tif  MODIS land cover (GeoTIFF)
  hrrr_west_lat.npy              Latitude grid
  hrrr_west_lon.npy              Longitude grid

Examples:
  python generate_all_static_features.py
  python generate_all_static_features.py --input_dir ./data --output_dir ./data
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='./data',
                        help='Input directory (default: ./data)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--geo_file', type=str, default='west_geo_interpolate.h5',
                        help='DEM filename (default: geo.h5)')
    parser.add_argument('--landcover_file', type=str, default='MODIS_LandCover_HRRR_West.tif',
                        help='MODIS land-cover filename (default: MODIS_LandCover_HRRR.tif)')
    parser.add_argument('--lats_file', type=str, default='hrrr_west_lat.npy',
                        help='Latitude filename (default: lats.npy)')
    parser.add_argument('--lons_file', type=str, default='hrrr_west_lon.npy',
                        help='Longitude filename (default: lons.npy)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print("=" * 70)
    print("AERO-ODE surface static feature generator")
    print("=" * 70)
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"DEM file: {args.geo_file}")
    print(f"Land-cover file: {args.landcover_file}")
    print(f"Lat/lon files: {args.lats_file}, {args.lons_file}")
    print("=" * 70)
    
    # 1. Load inputs
    print("\n[1/4] Loading inputs...")
    try:
        lats, lons, dem, landcover_path = load_input_data(
            args.input_dir, 
            args.geo_file, 
            args.landcover_file,
            args.lats_file, 
            args.lons_file
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # 2. Compute terrain features
    print("\n[2/4] Computing terrain features...")
    terrain_features = compute_all_terrain_features(dem, lats, lons)
    
    # 3. Compute land-cover features
    print("\n[3/4] Computing land-cover features...")
    landcover_features = compute_all_landcover_features(landcover_path, lats, lons)
    
    # 4. Merge and save
    print("\n[4/4] Saving data...")
    all_features = {**terrain_features, **landcover_features}
    save_all_features(all_features, args.output_dir)
    
    # Statistics
    print_landcover_statistics(landcover_features['landcover'])
    print_mask_statistics(all_features)
    
    # Done
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  Terrain features:")
    print("    - slope.npy              Slope (normalized)")
    print("    - aspect_sin.npy         Aspect sin [-1, 1]")
    print("    - aspect_cos.npy         Aspect cos [-1, 1]")
    print("    - curvature.npy          Curvature (normalized)")
    print("    - tpi.npy                TPI (normalized)")
    print("    - terrain_roughness.npy  Terrain roughness (normalized)")
    print("  Land cover:")
    print("    - landcover.npy          Land class (0-17)")
    print("    - roughness_z0.npy       Roughness length (m)")
    print("    - roughness_log_z0.npy   log(roughness)")
    print("    - roughness_log_z0_norm.npy  Normalized log(roughness)")
    print("  Masks:")
    print("    - water_mask.npy         Water")
    print("    - urban_mask.npy         Urban")
    print("    - forest_mask.npy        Forest")
    print("    - cropland_mask.npy      Cropland")
    print("    - grassland_mask.npy     Grassland")
    print("  Combined:")
    print("    - static_features.h5     All features bundle")


if __name__ == "__main__":
    main()
