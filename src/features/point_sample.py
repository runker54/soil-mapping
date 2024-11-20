import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
import logging
from tqdm import tqdm
from pathlib import Path


def is_large_integer(series):
    """检查序列是否包含应该被视为字符串的大整数或浮点数"""
    if series.dtype == 'float64':
        return series.apply(lambda x: x.is_integer() and abs(x) > 2**53 - 1).any()
    return series.dtype == 'int64' and series.abs().max() > 2**53 - 1

def sample_rasters(point_shp_path, raster_folder_path, output_csv_path, keep_out_of_bounds=False, fill_value=np.nan, logger=None):
    """
    使用来自shapefile的点对栅格文件进行采样，并将结果保存到CSV文件中。
    """
    logger.info("开始栅格采样过程")

    # 读取点shapefile
    points_gdf = gpd.read_file(point_shp_path, encoding='utf8')
    logger.info(f"已加载shapefile，共{len(points_gdf)}个点")
    
    # 初始化存储结果的字典
    results = {
        'point_id': range(len(points_gdf)),
        'longitude': points_gdf.geometry.x,
        'latitude': points_gdf.geometry.y
    }

    # 包括所有非几何列
    label_columns = [col for col in points_gdf.columns if col != 'geometry']
    logger.info(f"包括的非几何列: {label_columns}")

    # 添加原始矢量数据的属性列
    for col in label_columns:
        if is_large_integer(points_gdf[col]):
            results[col] = points_gdf[col].astype(str)
            logger.info(f"列 '{col}' 包含大整数，作为字符串处理")
        else:
            results[col] = points_gdf[col]
    
    # 从几何中提取坐标
    coords = [(point.x, point.y) for point in points_gdf.geometry]
    logger.info("已从几何中提取坐标")

    # 添加内存使用和点数量的日志
    logger.info(f"点数据内存使用: {points_gdf.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # 遍历文件夹中的栅格文件
    raster_files = [f for f in os.listdir(raster_folder_path) if f.endswith('.tif')]
    logger.info(f"找到{len(raster_files)}个TIF文件")

    # 在处理栅格文件之前添加总体信息
    total_raster_size = sum(os.path.getsize(Path(raster_folder_path) / f) for f in raster_files)
    logger.info(f"待处理栅格文件总大小: {total_raster_size / 1024 / 1024:.2f} MB")

    valid_points = np.ones(len(coords), dtype=bool)

    for raster_file in tqdm(raster_files, desc="处理栅格"):
        raster_path = Path(raster_folder_path) / raster_file
        raster_name = raster_path.stem
        logger.info(f"开始处理栅格: {raster_name} (大小: {os.path.getsize(raster_path) / 1024 / 1024:.2f} MB)")
        
        try:
            with rasterio.open(raster_path) as src:
                # 添加栅格元数据信息
                logger.info(f"栅格 {raster_name} 信息: 大小={src.shape}, CRS={src.crs}, 数据类型={src.dtypes[0]}")
                
                # 在每个点位置采样栅格
                sampled_values = [src.sample([coord]) for coord in coords]
                
                # 处理采样结果
                processed_values = []
                for i, val in enumerate(sampled_values):
                    value = next(val)[0] if val else fill_value  # 直接获取第一个元素
                    if value == src.nodata or np.isnan(value):
                        if keep_out_of_bounds:
                            processed_values.append(fill_value)
                        else:
                            processed_values.append(None)
                            valid_points[i] = False
                    else:
                        processed_values.append(value)
                
                # 将采样值添加到结果字典
                results[raster_name] = processed_values
                
                # 添加采样统计信息
                valid_count = sum(1 for v in processed_values if v is not None)
                invalid_count = len(processed_values) - valid_count
                logger.info(f"栅格 {raster_name} 采样统计: 有效点={valid_count}, 无效点={invalid_count}")
                
            logger.info(f"栅格 {raster_name} 处理完成")
        except Exception as e:
            logger.error(f"处理栅格 {raster_name} 时出错: {str(e)}", exc_info=True)

    # 从结果创建DataFrame
    df = pd.DataFrame(results)
    if not keep_out_of_bounds:
        df = df[valid_points]  # 删除无效的行
    logger.info("已从采样结果创建DataFrame")

    # 添加最终结果统计
    logger.info(f"最终数据集大小: {len(df)}行 x {len(df.columns)}列")
    logger.info(f"输出CSV预计大小: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # 检查保存路径
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将DataFrame保存到CSV文件，保留数据类型
    df.to_csv(output_csv_path, index=False, float_format='%.10g', encoding='utf8')
    logger.info(f"采样结果已保存到 {output_csv_path}")

def main(point_shp_path, raster_folder_path, output_csv_path, keep_out_of_bounds=False, fill_value=np.nan,log_file=None):
    """
    主函数
    """
    # 设置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    
    logger = logging.getLogger(__name__)
    logger.info("开始点采样过程")

    try:
        sample_rasters(point_shp_path, raster_folder_path, output_csv_path, keep_out_of_bounds, fill_value, logger)
        logger.info("点采样过程完成")
    except Exception as e:
        logger.error(f"点采样过程中发生错误: {str(e)}")
        raise


# 测试
if __name__ == "__main__":
    point_shp_path = r'D:\soil-mapping\data\raw\soil_property_point\soil_property_point.shp'
    raster_folder_path = r'D:\soil-mapping\data\soil_property'
    output_csv_path = r'D:\soil-mapping\data\soil_property_table\soil_property_point.csv'
    keep_out_of_bounds = False
    fill_value = 0
    log_file = r'D:\soil-mapping\logs\point_sample.log'
    main(point_shp_path, raster_folder_path, output_csv_path, keep_out_of_bounds, fill_value, log_file)
