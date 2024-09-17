import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
import logging
from tqdm import tqdm



def align_rasters(base_raster, input_vector, other_rasters_path, output_folder, output_crs,logger):
    """
    对齐所有栅格数据。

    参数:
    base_raster (str): 基准栅格文件路径
    input_vector (str): 输入矢量文件路径
    other_rasters_path (str): 栅格文件所在文件夹路径（包括基准栅格）
    output_folder (str): 输出文件夹路径
    output_crs (str, optional): 输出坐标系统，例如 "EPSG:4326"
    """
    logger.info("开始对齐所有栅格")

    # 读取矢量文件
    vector = gpd.read_file(input_vector)
    
    # 处理基准栅格以获取参考信息
    with rasterio.open(base_raster) as src:
        # 确保矢量和栅格使用相同的坐标系
        if vector.crs != src.crs:
            vector = vector.to_crs(src.crs)
        # 获取矢量的边界
        bounds = vector.total_bounds
        
        # 计算边界范围内的像素
        window = from_bounds(*bounds, src.transform)
        window_ceiled = window.round_offsets(op='ceil')
        # 获取参考信息
        ref_transform = src.window_transform(window_ceiled)
        ref_height = window_ceiled.height
        ref_width = window_ceiled.width
        ref_crs = src.crs

        # 如果指定了输出坐标系，计算新的变换
        if output_crs:
            dst_crs = rasterio.crs.CRS.from_string(output_crs)
            ref_transform, ref_width, ref_height = calculate_default_transform(
                ref_crs, dst_crs, ref_width, ref_height, 
                *rasterio.transform.array_bounds(ref_height, ref_width, ref_transform)
            )
            ref_crs = dst_crs

    # 处理所有栅格（包括基准栅格）
    rasters = [os.path.join(other_rasters_path, f) for f in os.listdir(other_rasters_path) if f.endswith('.tif')]
    for raster in tqdm(rasters, desc="处理栅格"):
        logger.info(f"开始处理栅格: {raster}")
        with rasterio.open(raster) as src:
            # 重投影和裁剪栅格以匹配参考信息
            image, transform = reproject(
                source=rasterio.band(src, 1),
                destination=np.zeros((ref_height, ref_width), dtype=src.dtypes[0]),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest
            )

            # 更新元数据
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": ref_height,
                "width": ref_width,
                "transform": ref_transform,
                "crs": ref_crs
            })

            # 保存栅格
            output = Path(output_folder) / f"a_{Path(raster).name}"
            with rasterio.open(output, "w", **meta) as dest:
                dest.write(image, 1)

        logger.info(f"栅格对齐完成: {output}")

    logger.info("所有栅格处理完成。")

def main(base_raster, input_vector, other_rasters_path, output_folder, output_crs, log_file):
    # 设置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    logger = logging.getLogger(__name__)

    try:
        align_rasters(base_raster, input_vector, other_rasters_path, output_folder, output_crs,logger)
        logger.info("所有栅格处理完成。")
    except Exception as e:
        logger.error(f"处理栅格时发生错误: {e}")

