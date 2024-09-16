import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import fiona
import logging
from pathlib import Path

# 设置日志
log_file = Path('logs/table_to_shp.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# 使用示例
if __name__ == "__main__":
    # input_file = r"F:\collection_spb_info\sp_float_data\GL\marked_data.csv"  # 或 .csv
    input_file = r"C:\Users\Runker\Desktop\土壤培肥区域(1)\table\fg_result.csv"  # 或 .csv
    # output_file = r"F:\cache_data\shp_file\gl\gl_sp_chemical_info.shp"
    output_file = r"C:\Users\Runker\Desktop\土壤培肥区域(1)\table\fg_result.shp"
    lon_col = "dwjd"
    lat_col = "dwwd"
    input_crs = 4490  # cgcs2000
    output_crs = 4545  # 用户指定的输出坐标系

    table_to_shp(input_file, output_file, lon_col, lat_col, input_crs, output_crs)
