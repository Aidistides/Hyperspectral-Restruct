import numpy as np
import spectral as sp
import os
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin

def convert_npz_to_envi(npz_path, output_dir, format='envi'):
    data = np.load(npz_path)
    cube = data['data']          # shape: (rows, cols, bands) or similar
    mask = data.get('mask', None)  # optional validity mask
    
    # Clean: apply mask if present (set invalid to 0 or NaN)
    if mask is not None:
        cube = np.where(mask[..., None], cube, 0)  # or np.nan
    
    basename = os.path.basename(npz_path).replace('.npz', '')
    os.makedirs(output_dir, exist_ok=True)
    
    wavelengths = [462.08 + i*3.2 for i in range(150)]  # from dataset spec; adjust if you load wavelength.csv
    
    if format == 'envi':
        hdr_path = os.path.join(output_dir, f"{basename}.hdr")
        img_path = os.path.join(output_dir, f"{basename}.img")
        sp.envi.save_image(hdr_path, cube, dtype=np.float32, interleave='bil', 
                          metadata={'wavelength': wavelengths, 'wavelength units': 'nm'})
        print(f"✓ ENVI: {hdr_path} + {img_path}")
    
    elif format == 'tif':
        tif_path = os.path.join(output_dir, f"{basename}.tif")
        with rasterio.open(
            tif_path, 'w', driver='GTiff', height=cube.shape[0], width=cube.shape[1],
            count=cube.shape[2], dtype=cube.dtype, crs='EPSG:4326',  # update CRS if you have geo coords
            transform=from_origin(0, 0, 2, 2)  # placeholder 2m GSD; update if geo-referenced
        ) as dst:
            for i in range(cube.shape[2]):
                dst.write(cube[..., i], i+1)
        # Add wavelengths as metadata (optional)
        print(f"✓ GeoTIFF: {tif_path}")

# Batch process
input_dir = 'path/to/HYPERVIEW2-v2/train'  # or test
output_dir = 'data/hyperview_patches/envi'  # or /tif
for file in tqdm(os.listdir(input_dir)):
    if file.endswith('.npz'):
        convert_npz_to_envi(os.path.join(input_dir, file), output_dir, format='envi')  # or 'tif'
