import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import fiona
import logging
from pathlib import Path


def table_to_shp(input_file, output_file, lon_col, lat_col, input_crs=4326, output_crs=4545):
    # 设置fiona支持utf-8编码
    fiona.supported_drivers['ESRI Shapefile'] = 'rw'

    # 读取输入文件
    _, file_extension = os.path.splitext(input_file)
    if file_extension.lower() == '.xlsx':
        df = pd.read_excel(input_file)
    elif file_extension.lower() == '.csv':
        df = pd.read_csv(input_file, encoding='utf-8')
    else:
        raise ValueError("不支持的文件格式。请使用.xlsx或.csv文件。")

    # 创建几何列
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    
    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=f"EPSG:{input_crs}")
    
    # 如果需要,转换坐标系
    if input_crs != output_crs:
        gdf = gdf.to_crs(epsg=output_crs)
    
    # 保存为shapefile,使用utf-8编码
    gdf.to_file(output_file, driver="ESRI Shapefile", encoding='utf-8')

    print(f"Shapefile已保存至: {output_file}")

def main(input_file, output_file, lon_col, lat_col,log_file, input_crs=4326, output_crs=4545):
    # 设置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    
    logger = logging.getLogger(__name__)

    logger.info("开始表格转shapefile过程")
    try:
        table_to_shp(input_file, output_file, lon_col, lat_col, input_crs, output_crs)
        logger.info("表格转shapefile过程完成")
    except Exception as e:
        logger.error(f"表格转shapefile过程发生错误: {e}")
