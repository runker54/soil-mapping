import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.windows import Window
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import shape, mapping
from pathlib import Path
from tqdm import tqdm

# 设置日志
log_file = Path('logs/predict_soil_type.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_labels(model_path):
    logger.info(f"正在加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data and 'label_encoder' in model_data:
        model = model_data['model']
        feature_names = model_data['feature_names']
        label_encoder = model_data['label_encoder']
        categorical_features = model_data.get('categorical_features', [])
        
        logger.info(f"加载的模型期望 {len(feature_names)} 个特征")
        logger.info(f"特征名称: {feature_names}")
        logger.info(f"加载的标签编码器类别: {label_encoder.classes_}")
        logger.info(f"加载的标签编码器类别数量: {len(label_encoder.classes_)}")
        logger.info(f"类别特征: {categorical_features}")
        
        return model, feature_names, label_encoder, categorical_features
    else:
        raise ValueError("模型文件格式不正确，缺少必要信息")

def get_memmap_array(raster_path, window=None):
    with rasterio.open(raster_path) as src:
        if window is None:
            window = Window(0, 0, src.width, src.height)
        return src.read(1, window=window, out_shape=(window.height, window.width), masked=True)

def process_raster_chunk(model, feature_files, window, feature_names, label_encoder):
    chunk_data = {Path(file).stem: get_memmap_array(file, window) for file in feature_files}
    
    rows, cols = next(iter(chunk_data.values())).shape
    feature_array = np.stack([chunk_data[f] for f in feature_names], axis=-1)
    
    valid_pixels = ~np.isnan(feature_array).any(axis=-1)
    feature_array = feature_array[valid_pixels]
    
    logger.info(f"有效像素数量: {np.sum(valid_pixels)}")
    
    result = np.full((rows, cols), np.nan)
    
    if np.sum(valid_pixels) > 0:
        predictions = model.predict(feature_array)
        result[valid_pixels] = predictions
    else:
        logger.warning("此区块没有有效像素进行预测")
    
    return result

def predict_soil_type(model, feature_dir, output_path, feature_names, label_encoder, shapefile_path=None, chunk_size=1000):
    logger.info("开始预测土壤类型")
    feature_files = [Path(feature_dir) / f"{feature}.tif" for feature in feature_names]
    with rasterio.open(feature_files[0]) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress='lzw')
        height, width = src.shape

    mask = None
    if shapefile_path:
        with rasterio.open(feature_files[0]) as src:
            gdf = gpd.read_file(shapefile_path)
            geometries = [mapping(geom) for geom in gdf.geometry]
            mask, _ = rasterio_mask(src, geometries, crop=True, all_touched=True)
            mask = mask[0] != src.nodata
        logger.info(f"使用shapefile创建了掩码")
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for row in tqdm(range(0, height, chunk_size), desc="处理行"):
            for col in range(0, width, chunk_size):
                window = Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row))
                chunk_mask = None if mask is None else mask[row:row+window.height, col:col+window.width]
                chunk_result = process_raster_chunk(model, feature_files, window, feature_names, label_encoder)
                
                if chunk_mask is not None:
                    chunk_result[~chunk_mask] = np.nan
                
                dst.write(chunk_result.astype(rasterio.float32), 1, window=window)
    
    logger.info(f"预测结果已保存到: {output_path}")

def raster_to_vector(raster_path, output_shp_path, label_encoder):
    logger.info(f"正在将栅格转换为矢量: {raster_path}")
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        mask = ~np.isnan(image)
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(image, mask=mask, transform=src.transform))
        )

    gdf = gpd.GeoDataFrame.from_features(list(results))
    gdf['geometry'] = gdf['geometry'].apply(shape)
    gdf.crs = src.crs
    
    logger.info(f"标签编码器类别: {label_encoder.classes_}")
    logger.info(f"栅格值范围: {gdf['raster_val'].min()} to {gdf['raster_val'].max()}")
    
    def safe_inverse_transform(x):
        try:
            if np.isnan(x):
                return 'Unknown'
            rounded_x = int(round(float(x)))
            return label_encoder.inverse_transform([rounded_x])[0]
        except Exception as e:
            logger.error(f"转换值 {x} 时出错: {str(e)}")
            return 'Unknown'
    
    gdf['soil_type'] = gdf['raster_val'].apply(safe_inverse_transform)
    
    logger.info(f"唯一的土壤类型: {gdf['soil_type'].unique()}")
    
    gdf = gdf.drop(columns=['raster_val'])
    gdf.to_file(output_shp_path, driver='ESRI Shapefile', encoding='utf-8')
    logger.info(f"矢量文件已保存到: {output_shp_path}")

def predict_soil_type_main(model_path, feature_dir, output_dir, shapefile_path=None):
    model_path, feature_dir, output_dir = map(Path, [model_path, feature_dir, output_dir])
    output_dir.mkdir(parents=True, exist_ok=True)

    model, feature_names, label_encoder, _ = load_model_and_labels(model_path)

    logger.info(f"模型期望的特征: {feature_names}")
    logger.info(f"特征数量: {len(feature_names)}")
    logger.info(f"标签编码器类别: {label_encoder.classes_}")

    output_raster_path = output_dir / "soil_type_prediction.tif"
    predict_soil_type(model, feature_dir, output_raster_path, feature_names, label_encoder, shapefile_path)

    # 在转换为矢量之前，检查栅格文件
    with rasterio.open(output_raster_path) as src:
        unique_values = np.unique(src.read(1))
        logger.info(f"栅格中的唯一值: {unique_values}")

    output_shp_path = output_dir / "soil_type_prediction.shp"
    raster_to_vector(output_raster_path, output_shp_path, label_encoder)

    logger.info("土壤类型预测完成")

if __name__ == "__main__":
    model_path = Path(r"C:\Users\Runker\Desktop\GL\rf_type\models\tz_model.pkl")
    feature_dir = Path(r"C:\Users\Runker\Desktop\GL\gl_tif_aligin")
    output_dir = Path(r"C:\Users\Runker\Desktop\GL\gl_tif_type_predict")
    shapefile_path = Path(r"C:\Users\Runker\Desktop\GL\gl\gl_extent_500m.shp")  # 可选，如果不需要限制区域，设为 None

    predict_soil_type_main(model_path, feature_dir, output_dir, shapefile_path)