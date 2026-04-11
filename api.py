# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import torch
import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.transform import from_origin
import tempfile
import shutil
from pathlib import Path
from typing import Dict

app = FastAPI(title="Hyperspectral-Restruct Nitrogen API", version="1.0")

# Load ONNX model (edge-optimized)
ort_session = ort.InferenceSession("models/nitrogen_model.onnx")

@app.post("/predict_nitrogen")
async def predict_nitrogen(
    hsi_cube: UploadFile = File(...),          # .npy file (bands x H x W)
    geotransform: str = None,                  # optional: "ulx,uly,resx,resy"
) -> Dict:
    """
    Predict soil nitrogen from hyperspectral cube.
    Returns JSON metrics + downloadable GeoTIFF false-color map.
    """
    if not hsi_cube.filename.endswith(".npy"):
        raise HTTPException(400, "Upload .npy hyperspectral cube")

    # Load cube
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
        shutil.copyfileobj(hsi_cube.file, tmp)
        cube_path = tmp.name

    try:
        cube = np.load(cube_path).astype(np.float32)          # shape: (bands, H, W) or (1, bands, H, W)
        if len(cube.shape) == 3:
            cube = cube[np.newaxis, ...]                     # add batch dim

        # Run ONNX inference
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: cube})
        n_map = outputs[0].squeeze()                         # (H, W) nitrogen map (% or mg/kg)

        # Simple stats
        mean_n = float(n_map.mean())
        max_n = float(n_map.max())
        min_n = float(n_map.min())

        # Generate GeoTIFF (false-color N map)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
            tif_path = tmp_tif.name

        # Default geotransform if none provided
        if geotransform:
            ulx, uly, resx, resy = map(float, geotransform.split(","))
        else:
            ulx, uly, resx, resy = 0.0, 0.0, 1.0, -1.0

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=n_map.shape[0],
            width=n_map.shape[1],
            count=1,
            dtype=n_map.dtype,
            crs="EPSG:4326",
            transform=from_origin(ulx, uly, resx, resy),
        ) as dst:
            dst.write(n_map, 1)

        return JSONResponse({
            "status": "success",
            "mean_nitrogen_percent": round(mean_n, 3),
            "min_nitrogen": round(min_n, 3),
            "max_nitrogen": round(max_n, 3),
            "map_shape": n_map.shape,
            "swir_bands_used": [1478, 1697, 2050, 2104, 2410],  # from NITROGEN_PREDICTIVE_BANDS.md
            "download_url": "/download_map"  # or return file directly
        })

    finally:
        Path(cube_path).unlink(missing_ok=True)

@app.get("/download_map")
async def download_map():
    """Download the latest generated GeoTIFF (in production you would store per-request)"""
    # For demo: return the last generated map (in real deployment use a temp store or DB)
    map_path = "/tmp/latest_n_map.tif"   # populated by predict endpoint in prod
    return FileResponse(map_path, media_type="image/tiff", filename="nitrogen_map.tif")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
