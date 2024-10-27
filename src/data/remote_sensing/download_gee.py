import ee
import geemap
import os
import math
import geopandas as gpd
from typing import List, Tuple
import logging
import time
import backoff
from pathlib import Path



@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def authenticate_gee(logger):
    """使用重试机制验证Google Earth Engine。"""
    logger.info("尝试验证Google Earth Engine")
    try:
        ee.Initialize()
        logger.info("验证成功")
    except Exception as e:
        logger.warning(f"验证失败：{str(e)}。尝试重新验证...")
        try:
            ee.Authenticate()
            ee.Initialize()
            logger.info("重新验证成功")
        except Exception as e:
            logger.error(f"重新验证失败：{str(e)}")
            raise

def split_region(geometry: gpd.GeoDataFrame, col_size: float, row_size: float,logger) -> List[Tuple[float, float, float, float]]:
    """将一个区域分割成更小的矩形。"""
    logger.info(f"将区域分割成大小为{col_size}x{row_size}的矩形")
    bounds = geometry.total_bounds
    min_x, min_y, max_x, max_y = bounds
    x_len = max_x - min_x
    y_len = max_y - min_y
    
    logger.info(f"Region bounds: min_x={min_x:.4f}, min_y={min_y:.4f}, max_x={max_x:.4f}, max_y={max_y:.4f}")
    logger.info(f"Region size: x_len={x_len:.4f}, y_len={y_len:.4f}")
    
    cols = max(1, math.ceil(x_len / col_size))
    rows = max(1, math.ceil(y_len / row_size))
    
    logger.info(f"Creating {cols}x{rows} = {cols*rows} rectangles")
    
    rectangles = [
        (min_x + i * col_size, min_y + j * row_size, 
         min(min_x + (i+1) * col_size, max_x), min(min_y + (j+1) * row_size, max_y))
        for i in range(cols) for j in range(rows)
    ]
    
    logger.info(f"Created {len(rectangles)} rectangles")
    return rectangles

def mask_s2_clouds(image):
    """在Sentinel-2影像中遮蔽云层。"""
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(
        qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

def get_sentinel2_collection(region: ee.Geometry, start_date: str, end_date: str, cloud_cover: int = 20) -> ee.ImageCollection:
    """获取给定区域和时间段的Sentinel-2影像集合。"""
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12','QA60','SCL']  # 选择要下载的波段
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select(bands)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
            .map(mask_s2_clouds))

@backoff.on_exception(backoff.expo, ee.EEException, max_tries=5)
def download_sentinel2_data(region: ee.Geometry, start_date: str, end_date: str, output_folder: str, logger, cloud_cover_threshold: int = 20):
    """下载给定区域和时间段的Sentinel-2数据。"""
    logger.info(f"下载{start_date}到{end_date}期间的Sentinel-2数据")
    
    os.makedirs(output_folder, exist_ok=True)
    
    rectangles = split_region(region, 0.1, 0.1, logger)  #默认  0.1 x 0.1 度的矩形，避免下载超出限制
    
    for i, rect in enumerate(rectangles):
        logger.info(f"Processing rectangle {i+1}/{len(rectangles)}")
        
        rect_geometry = ee.Geometry.Rectangle(rect)
        
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(rect_geometry) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_threshold)) \
            .map(mask_s2_clouds)
        
        if s2.size().getInfo() == 0:
            logger.warning(f"No images found for rectangle {i+1}")
            continue
        
        composite = s2.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']).median()
        
        export_params = {
            'image': composite,
            'description': f'sentinel2_composite_{i+1}',
            'folder': 'GEE_exports',
            'fileNamePrefix': f'sentinel2_{start_date}_{end_date}_{i+1}',
            'region': rect_geometry,
            'scale': 10, # 默认10米分辨率
            'crs': 'EPSG:4326', # 默认WGS84坐标系
            'maxPixels': 1e13 # 
        }
        
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()
        
        while task.active():
            logger.info(f"Task {task.id} for rectangle {i+1} is {task.state()}")
            time.sleep(30)
        
        if task.state() == 'COMPLETED':
            logger.info(f"Task {task.id} for rectangle {i+1} completed successfully")
        else:
            logger.error(f"Task {task.id} for rectangle {i+1} failed: {task.status()['error_message']}")

def main(region_path: str, output_folder: str, start_date: str, end_date: str,log_file:str):
    """下载Sentinel-2数据的主函数。"""
    # 配置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    
    logger = logging.getLogger(__name__)
    logger.info("开始下载Sentinel-2数据")
    try:
        # 验证Google Earth Engine
        logger.info("验证Google Earth Engine")
        authenticate_gee(logger)
        
        # 读取研究区域
        logger.info("读取研究区域")
        region = gpd.read_file(region_path)
        ee_region = geemap.geopandas_to_ee(region)
        
        # 下载数据
        logger.info("下载数据")
        download_sentinel2_data(ee_region, start_date, end_date, output_folder,logger)
        
        logger.info("数据下载成功完成")
    except Exception as e:
        logger.error(f"发生错误：{str(e)}")
        raise
# 测试
if __name__ == "__main__":
    region_path = r'F:\soil_mapping\dy\soil-mapping\data\raw\shapefile\DY_20230701_20231031.shp'
    output_folder = r'F:\GEEDOWNLOAD\sentinel2\DY_20230701_20231031'
    start_date = '2023-07-01'
    end_date = '2023-10-31'
    log_file = r'F:\soil_mapping\dy\soil-mapping\logs\download_gee.log'
    main(region_path, output_folder, start_date, end_date,log_file)
