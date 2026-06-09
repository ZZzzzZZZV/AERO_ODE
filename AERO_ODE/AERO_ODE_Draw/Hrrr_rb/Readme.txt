check_region.py sets land-cover download bounds from the data grid lat/lon extent.

generate_all_static_features.py 
From files under data/:

Terrain: geo.h5
Longitude: lon.npy
Latitude: lat.npy
MODIS_LandCover_HRRR.tif land cover,

generate all static features needed for the surface model and save under data/
Run as-is; no edits required.  



Download MODIS_LandCover_HRRR.tif via GEE: 
https://code.earthengine.google.com/ paste and run:

// HRRR domain - US central/east
var region = ee.Geometry.Rectangle([-95.4196, 29.9204, -80.1078, 42.9374]);

// 2021 MODIS land cover (IGBP)
var landcover = ee.ImageCollection('MODIS/061/MCD12Q1')
  .filterDate('2021-01-01', '2021-12-31')
  .first()
  .select('LC_Type1');

// Preview map
Map.centerObject(region, 6);
Map.addLayer(landcover.clip(region), {
  min: 0, max: 17, 
  palette: [
    '05450a','086a10','54a708','78d203','009900',
    'c6b044','dcd159','dade48','fbff13','27ff87',
    'c24f44','a5a5a5','ff6d4c','69fff8','f9ffa4',
    '1c0dff','ffffff'
  ]
}, 'Land Cover');

// Export to Google Drive
Export.image.toDrive({
  image: landcover,
  description: 'MODIS_LandCover_HRRR',
  folder: 'GEE_Export',
  region: region,
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e10
});