import ee
import geemap
import os
import math
import geopandas as gpd
from typing import List, Optional, Dict, Any
import logging
import backoff
from pathlib import Path
import aiohttp
import asyncio
from shapely.geometry import box, Polygon, MultiPolygon
import json

# 常量定义
DEFAULT_MAX_SIZE = 40000000  # 40MB
DOWNLOAD_TIMEOUT = 600  # 10分钟
CHUNK_SIZE = 1024 * 1024  # 1MB

class GEEDownloadError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def authenticate_gee(logger: logging.Logger) -> None:
    logger.info("尝试验证Google Earth Engine")
    try:
        ee.Initialize()
        logger.info("验证成功")
    except Exception as e:
        error_msg = f"验证失败：{str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e

class GEEDatasetConfig:
    """简化的GEE数据集配置类"""
    def __init__(self, collection_name: str, bands: List[str], scale: int = 30):
        self.collection_name = collection_name
        self.bands = bands
        self.scale = scale
        self.crs = 'EPSG:4326'  # 使用默认值
        self.format = 'GEO_TIFF'  # 使用默认值
        self.max_size = 40000000  # 固定为40MB

    @classmethod
    def srtm_dem(cls):
        return cls(
            collection_name='USGS/SRTMGL1_003',
            bands=['elevation'],
            scale=30,
        )

def adaptive_split_geometry(geometry: Polygon, max_size: int, scale: int) -> List[Polygon]:
    """自适应分割几何体"""
    bounds = geometry.bounds
    width = (bounds[2] - bounds[0]) / scale
    height = (bounds[3] - bounds[1]) / scale
    area = width * height

    if area <= max_size:
        return [geometry]

    # 计算需要分割的份数
    split_factor = math.ceil(math.sqrt(area / max_size))
    dx = (bounds[2] - bounds[0]) / split_factor
    dy = (bounds[3] - bounds[1]) / split_factor

    sub_geometries = []
    for i in range(split_factor):
        for j in range(split_factor):
            minx = bounds[0] + i * dx
            miny = bounds[1] + j * dy
            maxx = bounds[0] + (i + 1) * dx
            maxy = bounds[1] + (j + 1) * dy
            sub_box = box(minx, miny, maxx, maxy)
            if sub_box.intersects(geometry):
                sub_geom = sub_box.intersection(geometry)
                if not sub_geom.is_empty:
                    sub_geometries.append(sub_geom)

    return sub_geometries

async def download_file(url: str, output_path: str, logger: logging.Logger) -> None:
    """异步下载文件"""
    try:
        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise GEEDownloadError(f"下载失败，HTTP状态码: {response.status}")

                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)

                with open(output_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)

        logger.info(f"文件已下载到: {output_path}")
    except Exception as e:
        raise GEEDownloadError(f"下载文件时出错: {str(e)}") from e

def get_download_url(image: ee.Image, region: Dict[str, Any], config: GEEDatasetConfig) -> str:
    """获取下载URL"""
    try:
        url = image.getDownloadURL({
            'region': region,
            'scale': config.scale,
            'crs': config.crs,
            'format': config.format,
            'bands': config.bands
        })
        return url
    except Exception as e:
        raise GEEDownloadError(f"获取下载URL时出错: {str(e)}") from e

def download_gee_data(region: gpd.GeoDataFrame,
                     dataset_config: GEEDatasetConfig,
                     output_folder: str,
                     logger: logging.Logger) -> None:
    """简化的下载函数"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        image = ee.Image(dataset_config.collection_name)
        
        for idx, row in region.iterrows():
            geom = row.geometry
            # ... 保留几何体处理逻辑 ...
            
            # 简化URL获取和下载逻辑
            coords = [[[x, y] for x, y in zip(*geom.exterior.coords.xy)]]
            ee_geometry = ee.Geometry.Polygon(coords)
            region_dict = ee_geometry.getInfo()
            
            output_path = os.path.join(output_folder, f"dem_part_{idx}.tif")
            
            if os.path.exists(output_path):
                logger.info(f"文件已存在，跳过: {output_path}")
                continue
                
            url = get_download_url(image, region_dict, dataset_config)
            asyncio.run(download_file(url, output_path, logger))

    except Exception as e:
        raise GEEDownloadError(f"下载GEE数据时出错: {str(e)}") from e

def download_dem(region_path: str, output_folder: str, log_file: Optional[str] = None) -> None:
    """简化的主函数"""
    # 配置日志
    log_dir = Path(log_file).parent if log_file else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("开始下载DEM数据")
    
    try:
        authenticate_gee(logger)
        region = gpd.read_file(region_path)
        dem_config = GEEDatasetConfig.srtm_dem()
        
        download_gee_data(
            region=region,
            dataset_config=dem_config,
            output_folder=output_folder,
            logger=logger
        )
        
        logger.info("DEM数据下载成功完成")
    except Exception as e:
        logger.error(f"下载过程发生错误：{str(e)}")
        raise GEEDownloadError(str(e)) from e

if __name__ == "__main__":
    region_path = r'F:\cache_data\shp_file\test\test_extent.shp'
    output_folder = r'F:\GEEDOWNLOAD\dem'
    log_file = r'F:\soil_mapping\ky\soil-mapping\logs\download_gee_dem.log'
    
    download_dem(region_path, output_folder, log_file)
