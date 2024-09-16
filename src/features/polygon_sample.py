import os
import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from rasterstats import zonal_stats
import logging
from datetime import datetime
import sys
from shapely.geometry import shape
from rasterio.mask import mask
import multiprocessing
import psutil
from tqdm import tqdm
import time
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger('zonal_raster', Path('logs/zonal_raster.log'))

def preprocess_vector(vector_path, simplify=True, tolerance=0.001):
    gdf = gpd.read_file(vector_path)
    if simplify:
        original_area = gdf.area.sum()
        gdf['geometry'] = gdf.geometry.simplify(tolerance=tolerance)
        simplified_area = gdf.area.sum()
        area_change = (simplified_area - original_area) / original_area * 100
        logger.info(f"几何简化完成。面积变化: {area_change:.2f}%")
    return gdf

def process_single_raster(args):
    vector_data, raster_path = args
    logger.info(f"开始处理栅格文件: {Path(raster_path).name}")
    
    with rasterio.open(raster_path) as src:
        logger.info(f"栅格文件打开成功: {raster_path}")
        logger.info(f"栅格形状: {src.shape}, 数据类型: {src.dtypes[0]}")
        
        raster_data = src.read(1, masked=True)
        affine = src.transform
        
        stats_results = zonal_stats(vector_data, 
                                    raster_data, 
                                    affine=affine,
                                    stats=['mean', 'std', 'min', 'max', 'count', 'majority'],
                                    all_touched=True,
                                    nodata=src.nodata)
        
    logger.info(f"zonal_stats 计算完成，结果数量: {len(stats_results)}")

    result_df = pd.DataFrame(stats_results)
    logger.info(f"转换为DataFrame成功，形状: {result_df.shape}")
    
    raster_name = Path(raster_path).stem.replace(' ', '_').replace('-', '_')
    
    # 修改这里：确保列名格式正确
    new_columns = {col: f"{raster_name}_{col}" for col in result_df.columns}
    result_df.rename(columns=new_columns, inplace=True)
    
    logger.info(f"完成栅格文件处理: {Path(raster_path).name}")
    return result_df

def process_rasters(polygon_path, raster_folder, output_path, output_format='csv', simplify=True, simplify_tolerance=0.001, use_multiprocessing=True, num_workers=None):
    start_time = time.time()
    
    raster_list = list(Path(raster_folder).glob('*.tif'))
    logger.info(f"找到 {len(raster_list)} 个栅格文件")
    
    logger.info(f"读取并预处理矢量数据: {polygon_path}")
    gdf = preprocess_vector(polygon_path, simplify=simplify, tolerance=simplify_tolerance)
    
    if use_multiprocessing:
        if num_workers is None:
            available_memory = psutil.virtual_memory().available
            num_workers = min(multiprocessing.cpu_count(), max(1, int(available_memory / (4 * 1024 * 1024 * 1024))))
        logger.info(f"使用 {num_workers} 个工作进程进行并行处理")
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_raster, [(gdf, str(raster_path)) for raster_path in raster_list]), total=len(raster_list), desc="处理栅格文件"))
    else:
        logger.info("使用单进程处理")
        results = []
        for raster_path in tqdm(raster_list, desc="处理栅格文件"):
            result = process_single_raster((gdf, str(raster_path)))
            results.append(result)
    
    logger.info("合并所有结果")
    df_merged = pd.concat([gdf.reset_index(drop=True)] + results, axis=1)
    
    logger.info("添加中心点坐标和边界框信息")
    df_merged['Centroid_X'] = gdf.geometry.centroid.x
    df_merged['Centroid_Y'] = gdf.geometry.centroid.y
    bounds = gdf.bounds
    df_merged['XMin'], df_merged['YMin'], df_merged['XMax'], df_merged['YMax'] = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
    
    logger.info(f"保存数据到: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format.lower() == 'csv':
        df_merged.to_csv(output_path, index=False, encoding='utf-8')
        logger.info("数据已保存为CSV格式")
    elif output_format.lower() == 'shp':
        gdf_result = gpd.GeoDataFrame(df_merged, geometry=gdf.geometry)
        gdf_result.to_file(output_path, driver='ESRI Shapefile', encoding='utf-8')
        logger.info("数据已保存为Shapefile格式")
    else:
        logger.error(f"不支持的输出格式: {output_format}") 
        raise ValueError(f"不支持的输出格式: {output_format}")

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"处理完成，总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    polygon_path = Path(r"C:\Users\Runker\Desktop\polygon_test\shp\gl_sd_dltb.shp")
    raster_folder = Path(r'C:\Users\Runker\Desktop\polygon_test\tif')
    output_path = Path(r"C:\Users\Runker\Desktop\polygon_test\polygon_sample_not.csv")
    output_format = 'csv'
    
    # 使用多进程
    process_rasters(polygon_path, raster_folder, output_path, output_format, simplify=True, simplify_tolerance=0.001, use_multiprocessing=True, num_workers=2)
    # 不使用多进程
    # process_rasters(polygon_path, raster_folder, output_path, output_format, simplify=True, simplify_tolerance=0.001, use_multiprocessing=False)