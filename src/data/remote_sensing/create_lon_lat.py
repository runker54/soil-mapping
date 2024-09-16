import rasterio
import numpy as np
import logging
import os
from pathlib import Path

# 设置日志
log_file = Path('logs/create_lon_lat.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_lon_lat(input_path, output_path):
    try:
        with rasterio.open(input_path) as src:
            # Get raster dimensions and transform
            height = src.height
            width = src.width
            transform = src.transform

            # Create coordinate arrays
            rows, cols = np.mgrid[0:height, 0:width]
            xs, ys = rasterio.transform.xy(transform, rows, cols)

            # Convert lists to arrays for raster writing
            lon_array = np.array(xs)
            lat_array = np.array(ys)

            # Update metadata for longitude and latitude
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "dtype": rasterio.float32,
                "count": 1,
            })

            # Write longitude to file
            with rasterio.open(os.path.join(output_path,'lon.tif'), "w", **out_meta) as dest:
                dest.write(lon_array.astype(rasterio.float32), 1)

            # Write latitude to file
            with rasterio.open(os.path.join(output_path,'lat.tif'), "w", **out_meta) as dest:
                dest.write(lat_array.astype(rasterio.float32), 1)
        logger.info(f"Longitude and latitude rasters created successfully. Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating longitude and latitude rasters: {str(e)}")
        raise

if __name__ == "__main__":
    # This allows the script to be run standalone for testing
    input_path = r"C:\Users\Runker\Desktop\GL\gltif\DEM.tif"
    output_path = r"C:\Users\Runker\Desktop\GL\gltif"
    generate_lon_lat(input_path, output_path)