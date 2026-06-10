HRRR data used in this project:

geo.h5           Terrain for this region, shape (1, 440, 408) — lat points, lon points
lats.npy         Latitude grid, shape (440, 408) — lat points, lon points
lons.npy         Longitude grid, shape (440, 408) — lat points, lon points
hrrr_rb_2021.log Training data file layout
hrrr_rb_2022.log Training data file layout
hrrr_rb_2023.log Training data file layout
hrrr_rb_2021.log Test data file layout
Each data file is .h5, shape (24, 24, 440, 408): 24 hours, 24 variables, lat, lon
Variable order:
'z50', 'z500', 'z850', 'z1000',
't50', 't500', 't850', 't1000',
's50', 's500', 's850', 's1000',
'u50', 'u500', 'u850', 'u1000',
'v50', 'v500', 'v850', 'v1000',
'mslp', 'u10',  'v10',  't2m'.
