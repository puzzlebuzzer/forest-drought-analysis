import rasterio
import numpy as np

terrain_dir = "/home/rowan/win/LifeMgmt/Mind/School/UwGisProgram/Project_Appalachia/AOI/GWNF_cache/terrain"

with rasterio.open(f"{terrain_dir}/elevation.tif") as src:
    elev = src.read(1, masked=True)
    profile = src.profile.copy()
    profile.update(dtype='uint8', nodata=255)

    bands = {
        "mask_elev_low.tif":  (elev > 0)   & (elev < 175),
        "mask_elev_mid.tif":  (elev >= 175) & (elev < 300),
        "mask_elev_high.tif": (elev >= 300) & (elev.mask == False),
    }

    for filename, mask in bands.items():
        out = np.where(mask, 1, 0).astype('uint8')
        out[elev.mask] = 255  # preserve nodata
        with rasterio.open(f"{terrain_dir}/{filename}", 'w', **profile) as dst:
            dst.write(out, 1)
        
        pixel_count = mask.sum()
        area_km2 = pixel_count * 100 / 1_000_000  # 10m pixels
        print(f"{filename}: {pixel_count:,} pixels ({area_km2:.1f} km²)")