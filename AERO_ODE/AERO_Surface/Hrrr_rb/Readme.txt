check_region.py determines the lat/lon extent for downloading land-cover data based on the data grid bounds.

generate_all_static_features.py
Uses the following files in data/:

terrain data, geo.h5
longitude data, lon.npy
latitude data, lat.npy
MODIS_LandCover_HRRR.tif — land-cover data

Generates all static feature variables required by the surface model and saves them to data/.
No configuration changes needed; run directly.



MODIS_LandCover_HRRR.tif download code — paste and run at https://code.earthengine.google.com/:

// HRRR region - US Midwest / Southeast
var region = ee.Geometry.Rectangle([-95.4196, 29.9204, -80.1078, 42.9374]);

// MODIS land use for 2021 (IGBP classification)
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
