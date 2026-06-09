HRRR data used in this project:

geo.h5           Regional terrain, shape (1, 440, 408) — lat x lon grid
lats.npy         Latitude grid, shape (440, 408)
lons.npy         Longitude grid, shape (440, 408)
hrrr_rb_2021.log Training file layout log
hrrr_rb_2022.log Training file layout log
hrrr_rb_2023.log Training file layout log
hrrr_rb_2021.log Test file layout log

Each .h5 file has shape (24, 24, 440, 408): 24 hours, 24 variables, lat, lon.
Variable order:
