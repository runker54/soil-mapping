import rasterio
import numpy as np
import logging
import os
from pathlib import Path

def generate_lon_lat(input_path, output_path,logger):
    try:
        with rasterio.open(input_path) as src:
            # Get raster dimensions and transform
            logger.info(f"Raster dimensions: {src.height} x {src.width}")
            height = src.height
            width = src.width
            transform = src.transform

            # Create coordinate arrays
            rows, cols = np.mgrid[0:height, 0:width]
            xs, ys = rasterio.transform.xy(transform, rows, cols)

            # Convert lists to arrays for raster writing
            logger.info(f"Creating longitude and latitude rasters")
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

def main(input_path, output_path,log_file):
    # 设置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    logger = logging.getLogger(__name__)
    logger.info("开始创建经纬度栅格")
    try:
        generate_lon_lat(input_path, output_path,logger)
    except Exception as e:
        logger.error(f"创建经纬度栅格时发生错误: {e}")
        raise
    finally:
        logger.info("经纬度栅格创建完成")
