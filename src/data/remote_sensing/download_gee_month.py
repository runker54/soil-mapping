import ee
import geemap
import os
import math
import geopandas as gpd
from typing import List, Tuple, Optional
import logging
import time
import backoff
from pathlib import Path
import aiohttp
import asyncio
from urllib.parse import urlparse

# 常量定义
SENTINEL2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'] # 哨兵2波段
QA_BANDS = ['QA60', 'SCL'] # 质量控制波段
ALL_BANDS = SENTINEL2_BANDS + QA_BANDS # 所有波段
DEFAULT_SCALE = 10 # 默认像元大小
DEFAULT_MAX_SIZE = 40000000  # 40MB # 每个子区域的最大大小
DEFAULT_CLOUD_COVER = 20 # 云层覆盖率阈值
MAX_CONCURRENT_DOWNLOADS = 5 # 并发下载的最大数量
DOWNLOAD_TIMEOUT = 600  # 10分钟
CHUNK_SIZE = 1024 * 1024  # 1MB

class GEEDownloadError(Exception):
    """Google Earth Engine数据下载异常类。
    用于处理在下载GEE数据过程中可能出现的各种错误，包括但不限于：
    - 认证失败
    - 网络连接错误
    - 下载超时
    - 数据格式错误
    - 存储空间不足
    Attributes:
        message (str): 错误信息
        error_code (int, optional): 错误代码
        details (dict, optional): 详细错误信息
    """

    def __init__(self, message: str, error_code: int = None, details: dict = None):
        """初始化GEEDownloadError实例。

        Args:
            message: 错误描述信息
            error_code: 错误代码（可选）
            details: 详细错误信息字典（可选）
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """返回格式化的错误信息。"""
        error_msg = self.message
        if self.error_code:
            error_msg = f"[错误代码 {self.error_code}] {error_msg}"
        if self.details:
            error_msg = f"{error_msg}\n详细信息: {self.details}"
        return error_msg

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def authenticate_gee(logger: logging.Logger) -> None:
    """
    验证Google Earth Engine服务。
    
    Args:
        logger: 日志记录器实例
    
    Raises:
        GEEDownloadError: 当验证失败时抛出
    """
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
            error_msg = f"重新验证失败：{str(e)}"
            logger.error(error_msg)
            raise GEEDownloadError(error_msg) from e

def estimate_image_size(geometry: ee.Geometry, scale: int = 10) -> int:
    """估算影像大小（字节）"""
    # 获取区域面积（平方米）
    area = geometry.area().getInfo()
    # 计算像素数量（考虑所有波段）
    num_bands = 12  # B1-B12
    pixels = (area / (scale * scale)) * num_bands
    # 估算文件大小（每个像素4字节，增加50%安全余量）
    estimated_size = pixels * 4 * 1.5
    return estimated_size

def adaptive_split_geometry(geometry: ee.Geometry, max_size: int = 40000000, scale: int = 10) -> List[ee.Geometry]:
    """自适应分割几何体，确保每个子区域的预计大小不超过最大限制"""
    estimated_size = estimate_image_size(geometry, scale)
    
    if estimated_size <= max_size:
        return [geometry]
    
    # 计算需要的分割数（向上取整，并增加一个额外的分割以确保安全）
    split_factor = math.ceil(math.sqrt(estimated_size / max_size)) + 1
    
    bounds = geometry.bounds().getInfo()['coordinates'][0]
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[2]
    
    width = max_x - min_x
    height = max_y - min_y
    
    x_step = width / split_factor
    y_step = height / split_factor
    
    sub_geometries = []
    for i in range(split_factor):
        for j in range(split_factor):
            sub_geometry = ee.Geometry.Rectangle([
                min_x + i * x_step,
                min_y + j * y_step,
                min_x + (i + 1) * x_step,
                min_y + (j + 1) * y_step
            ])
            sub_geometries.append(sub_geometry)
    
    return sub_geometries

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

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_download_url(image: ee.Image, geometry: ee.Geometry) -> str:
    """获取下载链接，带重试机制"""
    return image.getDownloadURL({
        'scale': 10,
        'region': geometry,
        'format': 'GEO_TIFF',
        'crs': 'EPSG:4326'
    })

async def download_file(url: str, output_path: str, logger: logging.Logger) -> None:
    """
    异步下载文件。
    
    Args:
        url: 下载链接
        output_path: 输出文件路径
        logger: 日志记录器实例
    
    Raises:
        GEEDownloadError: 当下载失败时抛出
    """
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    try:
        conn = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(output_path, 'wb') as f:
                        while True:
                            chunk = await response.content.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            f.write(chunk)
                    logger.info(f"文件下载成功：{output_path}")
                else:
                    error_msg = f"下载失败，HTTP状态码：{response.status}"
                    logger.error(error_msg)
                    raise GEEDownloadError(error_msg)
    except asyncio.TimeoutError as e:
        error_msg = f"下载超时：{output_path}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e
    except Exception as e:
        error_msg = f"下载过程发生错误：{str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e

async def download_subregion(composite: ee.Image, sub_geometry: ee.Geometry, output_path: str, logger):
    """下载子区域的影像"""
    try:
        # 检查该子区域是否有有效数据
        stats = composite.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=sub_geometry,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        # 检查是否所有波段都是0或null
        has_data = any(value not in [0, None] for value in stats.values())
        
        if not has_data:
            logger.warning(f"子区域没有有效数据，跳过下载")
            return False
            
        url = await get_download_url(composite, sub_geometry)
        await download_file(url, output_path, logger)
        return True
    except Exception as e:
        logger.error(f"获取下载链接或下载失败：{str(e)}")
        raise

async def download_all_subregions(composite: ee.Image, sub_geometries: List[ee.Geometry], 
                                output_folder: str, start_date: str, end_date: str, logger,
                                max_retries: int = 3):
    """异步下载所有子区域，包含失败重试机制"""
    semaphore = asyncio.Semaphore(5)
    failed_regions = []
    skipped_regions = []
    
    async def download_with_semaphore(sub_geometry, i):
        async with semaphore:
            output_path = os.path.join(output_folder, f'sentinel2_part_{i+1}.tif')
            
            # 检查文件是否已存在
            if os.path.exists(output_path):
                logger.info(f"文件已存在，跳过下载：{output_path}")
                return True, i
                
            try:
                has_data = await download_subregion(composite, sub_geometry, output_path, logger)
                if not has_data:
                    skipped_regions.append(i)
                    return True, i  # 返回True因为这不是错误，只是没有数据
                return True, i
            except Exception as e:
                logger.error(f"子区域 {i+1} 下载失败: {str(e)}")
                return False, i

    # 初始下载
    tasks = [
        download_with_semaphore(sub_geometry, i) 
        for i, sub_geometry in enumerate(sub_geometries)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    # 收集失败的区域
    failed_regions = [(i, sub_geometries[i]) for success, i in results if not success]
    
    # 重试失败的区域
    retry_count = 0
    while failed_regions and retry_count < max_retries:
        retry_count += 1
        logger.info(f"第 {retry_count} 次重试，还有 {len(failed_regions)} 个区域需要下载")
        
        # 等待一段时间后重试
        await asyncio.sleep(5)  # 等待5秒
            
        retry_tasks = [
            download_with_semaphore(sub_geometry, i)
            for i, sub_geometry in failed_regions
        ]
        
        retry_results = await asyncio.gather(*retry_tasks, return_exceptions=False)
        
        # 更新失败列表
        failed_regions = [(i, sub_geometries[i]) for success, i in retry_results if not success]
    
    # 最终统计
    total_regions = len(sub_geometries)
    success_count = total_regions - len(failed_regions)
    
    logger.info(f"下载完成: {success_count} 成功, {len(failed_regions)} 失败")
    
    if failed_regions:
        logger.warning("以下区域下载失败：")
        for i, _ in failed_regions:
            logger.warning(f"区域 {i+1}")
        raise Exception(f"仍有 {len(failed_regions)} 个区域下载失败")
    else:
        logger.info("所有区域下载成功完成")

def get_monthly_date_ranges(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """生成每月的起止日期对。
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        List[Tuple[str, str]]: 每月的起止日期对列表
    """
    from datetime import datetime, timedelta
    import calendar
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_ranges = []
    current = start
    
    while current <= end:
        # 获取当月最后一天
        _, last_day = calendar.monthrange(current.year, current.month)
        month_end = datetime(current.year, current.month, last_day)
        
        # 如果当月结束日期超过了总的结束日期，使用总的结束日期
        if month_end > end:
            month_end = end
            
        date_ranges.append((
            current.strftime('%Y-%m-%d'),
            month_end.strftime('%Y-%m-%d')
        ))
        
        # 移动到下个月第一天
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return date_ranges

def download_sentinel2_data(region: gpd.GeoDataFrame, start_date: str, end_date: str, 
                          output_folder: str, logger, cloud_cover_threshold: int = 20):
    """下载给定区域和时间段的Sentinel-2数据。"""
    logger.info(f"下载{start_date}到{end_date}期间的Sentinel-2数据")
    
    if region.crs != 'EPSG:4326':
        logger.info(f"将坐标从{region.crs}转换为WGS84(EPSG:4326)")
        region = region.to_crs('EPSG:4326')
    
    # 获取每月的日期范围
    monthly_ranges = get_monthly_date_ranges(start_date, end_date)
    
    # 获取研究区域的ee.Geometry对象
    region_geometry = ee.Geometry(region.geometry.iloc[0].__geo_interface__)
    
    for month_start, month_end in monthly_ranges:
        logger.info(f"处理时间段: {month_start} 到 {month_end}")
        
        # 为每个月创建子文件夹
        month_folder = os.path.join(output_folder, f"{month_start[:7]}")
        os.makedirs(month_folder, exist_ok=True)
        
        try:
            # 获取影像集合
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(region_geometry) \
                .filterDate(month_start, month_end) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_threshold))
            
            image_count = s2.size().getInfo()
            if image_count == 0:
                logger.warning(f"{month_start[:7]}没有找到符合条件的影像")
                continue
                
            logger.info(f"{month_start[:7]}找到 {image_count} 张影像")
            
            # 创建合成影像
            composite = s2.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'QA60']) \
                         .map(mask_s2_clouds) \
                         .median()
            
            composite = composite.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
            
            # 分割区域
            sub_geometries = adaptive_split_geometry(region_geometry)
            logger.info(f"{month_start[:7]}研究区域已被分割为 {len(sub_geometries)} 个子区域")
            
            # 异步下载所有子区域
            asyncio.run(download_all_subregions(
                composite, sub_geometries, month_folder, month_start, month_end, logger
            ))
            
            logger.info(f"{month_start[:7]}数据下载成功完成")
                
        except Exception as e:
            logger.error(f"{month_start[:7]}处理过程发生错误: {str(e)}")
            raise

def main(
    region_path: str,
    output_folder: str,
    start_date: str,
    end_date: str,
    log_file: Optional[str] = None
) -> None:
    """
    下载Sentinel-2数据的主函数。

    Args:
        region_path: 研究区域矢量文件路径
        output_folder: 输出文件夹路径
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        log_file: 日志文件路径，可选

    Raises:
        GEEDownloadError: 当下载过程发生错误时抛出
    """
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
    logger.info("开始下载Sentinel-2数据")
    
    try:
        authenticate_gee(logger)
        region = gpd.read_file(region_path)
        download_sentinel2_data(region, start_date, end_date, output_folder, logger)
        logger.info("数据下载成功完成")
    except Exception as e:
        error_msg = f"下载过程发生错误{str(e)}"
        logger.error(error_msg)
        raise GEEDownloadError(error_msg) from e
# 测试
if __name__ == "__main__":
    region_path = r'E:\MONTH_TEST\shp\zn.shp'
    output_folder = r'E:\MONTH_TEST\tif'
    start_date = '2024-01-01'
    end_date = '2024-11-30'
    log_file = r'E:\MONTH_TEST\logs\download_gee.log'
    main(region_path, output_folder, start_date, end_date,log_file)

