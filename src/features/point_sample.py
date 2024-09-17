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

def sample_rasters(point_shp_path, raster_folder_path, output_csv_path, keep_out_of_bounds=False, fill_value=np.nan,logger=None):
    """
    使用来自shapefile的点对栅格文件进行采样，并将结果保存到CSV文件中。

    参数:
    point_shp_path (str): 输入点shapefile的路径。
    raster_folder_path (str): 包含栅格文件的文件夹路径。
    output_csv_path (str): 保存输出CSV文件的路径。
    keep_out_of_bounds (bool, optional): 是否保留超出栅格范围的点数据。默认为False。
    fill_value (optional): 超出范围点的填充值。默认为np.nan。
    logger (logging.Logger, optional): 用于记录消息的Logger对象。
    """
    logger.info("开始栅格采样过程")

    # 读取点shapefile
    logger.info(f"读取shapefile: {point_shp_path}")
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

    # 包括标签列，保留数据类型
    for col in label_columns:
        if is_large_integer(points_gdf[col]):
            results[col] = points_gdf[col].astype(str)
            logger.info(f"列 '{col}' 包含大整数。作为字符串处理。")
        else:
            results[col] = points_gdf[col]

    # 从几何中提取坐标
    coords = [(point.x, point.y) for point in points_gdf.geometry]
    logger.info("已从几何中提取坐标")

    # 遍历文件夹中的栅格文件
    logger.info(f"处理文件夹中的栅格文件: {raster_folder_path}")
    raster_files = [f for f in os.listdir(raster_folder_path) if f.endswith('.tif')]
    logger.info(f"找到{len(raster_files)}个TIF文件")

    valid_points = np.ones(len(coords), dtype=bool)

    for raster_file in tqdm(raster_files, desc="处理栅格"):
        raster_path = Path(raster_folder_path) / raster_file
        raster_name = raster_path.stem
        logger.info(f"处理栅格: {raster_name}")
        
        try:
            with rasterio.open(raster_path) as src:
                # 在每个点位置采样栅格
                sampled_values = list(sample_gen(src, coords))
                
                # 处理采样结果
                processed_values = []
                for i, val in enumerate(sampled_values):
                    if val and len(val) > 0 and not np.isnan(val[0]) and val[0] != src.nodata:
                        processed_values.append(val[0])
                    elif keep_out_of_bounds:
                        processed_values.append(fill_value)
                    else:
                        processed_values.append(None)
                        valid_points[i] = False
                
                # 将采样值添加到结果字典
                results[raster_name] = processed_values
            logger.info(f"栅格 {raster_name} 处理成功")
        except Exception as e:
            logger.error(f"处理栅格 {raster_name} 时出错: {str(e)}")

    # 从结果创建DataFrame
    df = pd.DataFrame(results)
    logger.info("已从采样结果创建DataFrame")

    # 如果不保留超出范围的点，则删除无效的行
    if not keep_out_of_bounds:
        df = df[valid_points]
        logger.info(f"删除了超出范围的点后，剩余{len(df)}个点")

    # 检查保存路径
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将DataFrame保存到CSV文件，保留数据类型
    df.to_csv(output_csv_path, index=False, float_format='%.10g', encoding='utf8')
    logger.info(f"采样结果已保存到 {output_csv_path}")

    # 验证输出
    df_check = pd.read_csv(output_csv_path)
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('-', '').isdigit())).all():
                if not (df[col] == df_check[col]).all():
                    logger.warning(f"在列 {col} 中检测到差异")
                    logger.warning(f"原始: {df[col].iloc[0]}")
                    logger.warning(f"保存: {df_check[col].iloc[0]}")

    # 检查填充值是否正确应用
    if keep_out_of_bounds:
        for col in df.columns:
            if col not in label_columns + ['point_id', 'longitude', 'latitude']:
                fill_count = (df[col] == fill_value).sum()
                logger.info(f"列 {col} 中填充值 {fill_value} 的数量: {fill_count}")

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

