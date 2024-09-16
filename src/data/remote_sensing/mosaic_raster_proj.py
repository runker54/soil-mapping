import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
from rasterio.enums import ColorInterp
import logging
import sys
from tqdm import tqdm
from pathlib import Path
# 设置日志
log_file = Path('logs/mosaic_raster_proj.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_rasters(raster_path, output_path, target_crs=None):
    raster_paths = [os.path.join(raster_path, f) for f in os.listdir(raster_path) if f.endswith('.tif')]
    
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                          GDAL_TIFF_INTERNAL_MASK=True,
                          GDAL_PAM_ENABLED=False,
                          CPL_DEBUG=False):
            
            # 打开所有栅格文件
            src_files = [rasterio.open(raster) for raster in raster_paths]
            
            # 获取第一个文件的元数据
            meta = src_files[0].meta.copy()
            
            # 执行合并（使用原始投影）
            logger.info(f"Merging {len(src_files)} rasters in original projection...")
            mosaic, out_trans = merge(src_files)
            
            logger.info(f"Merged raster shape: {mosaic.shape}")
            
            # 更新元数据
            meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": mosaic.shape[0],
                "compress": "DEFLATE",
                "predictor": 2,
                "zlevel": 6
            })
            
            # 如果目标CRS与原始CRS不同，则进行重投影
            if target_crs != meta['crs'].to_string():
                logger.info(f"Reprojecting from {meta['crs'].to_string()} to {target_crs}")
                
                # 计算新的变换和尺寸
                dst_crs = rasterio.crs.CRS.from_string(target_crs)
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    meta['crs'], dst_crs, meta['width'], meta['height'], *rasterio.transform.array_bounds(meta['height'], meta['width'], out_trans)
                )
                
                # 更新元数据
                meta.update({
                    'crs': dst_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height
                })
                
                # 创建目标数组
                dst_mosaic = np.zeros((meta['count'], dst_height, dst_width), dtype=meta['dtype'])
                
                # 进行重投影
                for i in tqdm(range(meta['count']), desc="Reprojecting bands"):
                    reproject(
                        source=mosaic[i],
                        destination=dst_mosaic[i],
                        src_transform=out_trans,
                        src_crs=src_files[0].crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
                
                mosaic = dst_mosaic
            
            # 写入最终的栅格
            logger.info(f"Writing output to {output_path}")
            with rasterio.open(output_path, "w", **meta) as dest:
                dest.write(mosaic)
                # 设置颜色解释
                color_interps = [ColorInterp.undefined] * meta['count']
                dest.colorinterp = color_interps
        
        logger.info(f"Rasters processed successfully. Output saved to {output_path}")
        logger.info(f"Number of bands in output: {mosaic.shape[0]}")
        logger.info(f"Output CRS: {meta['crs'].to_string()}")
        
    except Exception as e:
        logger.error(f"Error processing rasters: {str(e)}")
        raise
    finally:
        # 关闭所有打开的文件
        for src in src_files:
            src.close()

if __name__ == "__main__":
    try:
        raster_paths = r'C:\Users\Runker\Desktop\test\gl'
        output_path = r'F:\tif_features\temp\mosaic.tif'
        target_crs = "EPSG:4544"  # 目标坐标系统
        merge_rasters(raster_paths, output_path, target_crs)
        print("Raster processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logger.exception("An error occurred during raster processing:")