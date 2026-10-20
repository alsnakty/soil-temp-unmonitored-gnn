import datetime
import os

import netCDF4
import numpy as np

URL = ("https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_surface_elev_netcdf/"
       "ETOPO_2022_v1_60s_N90W180_surface.nc")
KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(KOK, "DATASET", "DEM", "etopo2022_60s_turkiye.nc")
LAT, LON = (35.5, 42.5), (25.0, 45.5)


def main():
    ds = netCDF4.Dataset(URL)
    lat, lon = ds["lat"][:].data, ds["lon"][:].data
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise RuntimeError("lat/lon artan degil")
    i = np.nonzero((lat >= LAT[0]) & (lat <= LAT[1]))[0]
    j = np.nonzero((lon >= LON[0]) & (lon <= LON[1]))[0]
    z = ds["z"][i[0]:i[-1] + 1, j[0]:j[-1] + 1]
    z = np.ma.filled(z.astype(np.float32), np.nan)
    if z.shape != (len(i), len(j)):
        raise RuntimeError("kesit boyutu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with netCDF4.Dataset(OUT, "w") as o:
        o.createDimension("lat", len(i))
        o.createDimension("lon", len(j))
        v = o.createVariable("lat", "f8", ("lat",)); v[:] = lat[i]; v.units = "degrees_north"
        v = o.createVariable("lon", "f8", ("lon",)); v[:] = lon[j]; v.units = "degrees_east"
        v = o.createVariable("z", "f4", ("lat", "lon"), zlib=True); v[:] = z
        v.units, v.long_name, v.vert_crs_name = "meters", "surface height (topography-bathymetry)", "EGM2008"
        o.source = URL
        o.kesit = f"lat indis {i[0]}-{i[-1]}, lon indis {j[0]}-{j[-1]}"
        o.indirme = datetime.datetime.now().isoformat(timespec="seconds")
        o.kaynak_adi = "NOAA NCEI ETOPO 2022 v1, 60 arc-second, surface elevation"
    print(f"{OUT}: {z.shape}, lat {lat[i[0]]:.4f}-{lat[i[-1]]:.4f}, lon {lon[j[0]]:.4f}-{lon[j[-1]]:.4f}, "
          f"z {np.nanmin(z):.0f}..{np.nanmax(z):.0f} m, NaN {int(np.isnan(z).sum())}, "
          f"{os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
