import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import box, mapping
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pykrige.rk import RegressionKriging
from sklearn.ensemble import RandomForestRegressor

# 设置日志
log_file = Path('logs/predict_soil_properties.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    logger.info(f"正在加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data:
        model = model_data['model']
        feature_names = [str(f) for f in model_data['feature_names']]  # 转换为普通字符串
        n_features = model.n_features_in_  # 使用模型的实际特征数量
        logger.info(f"加载的模型期望 {n_features} 个特征")
        logger.info(f"特征名称: {feature_names}")
        return model, feature_names[:n_features]
    else:
        raise ValueError("模型文件格式不正确，缺少模型或特征名称信息")

def load_data(file_path, label_col):
    logger.info(f"正在从 {file_path} 加载数据")
    data = pd.read_csv(file_path)
    numeric_cols = data.select_dtypes(include=[np.number])
    data[numeric_cols.columns] = data[numeric_cols.columns].fillna(numeric_cols.median())
    data = data[data[f"{label_col}_Sta"] == 'Normal']
    logger.info(f"数据加载完成。形状：{data.shape}")
    return data

def get_raster_info(raster_path):
    with rasterio.open(raster_path) as src:
        return src.profile, src.shape, src.bounds

def compare_models_and_train_kriging(X, y, rf_model, feature_names, coord_cols, use_rk=False, test_size=0.2, random_state=42):
    X_model = X[feature_names]
    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size, random_state=random_state)

    logger.info(f"X_test 列: {X_test.columns.tolist()}")
    logger.info(f"RF 模型特征数量: {rf_model.n_features_in_}")

    rf_predictions = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_predictions)

    X_full = X[feature_names + coord_cols]
    X_train_full, X_test_full, _, _ = train_test_split(X_full, y, test_size=test_size, random_state=random_state)

    rk = RegressionKriging(
        regression_model=rf_model,
        n_closest_points=8,
        variogram_model='linear',  # 可以尝试'linear'，'spherical'，'exponential'，其中'spherical'效果最好，速度最快的是'linear'
        exact_values=False
    )
    rk.fit(
        X_train_full[feature_names].values,
        X_train_full[coord_cols].values,
        y_train
    )
    rfrk_predictions = rk.predict(X_test_full[feature_names].values, X_test_full[coord_cols].values)
    rfrk_r2 = r2_score(y_test, rfrk_predictions)

    logger.info(f"RF R2 score: {rf_r2}")
    logger.info(f"RFRK R2 score: {rfrk_r2}")
    if use_rk:
        return (rk, True) if rfrk_r2 > rf_r2 else (rf_model, False)
    else:
        return (rf_model, False)

def predict_chunk(model, feature_data, coord_data, is_rfrk, feature_names):
    if feature_data.shape[0] == 0:
        logger.warning("没有有效数据进行预测")
        return np.array([])
    
    if is_rfrk:
        return model.predict(feature_data, coord_data)
    else:
        return model.predict(feature_data)

def get_memmap_array(raster_path, window=None):
    with rasterio.open(raster_path) as src:
        if window is None:
            window = Window(0, 0, src.width, src.height)
        return src.read(1, window=window, out_shape=(window.height, window.width), masked=True)

def process_raster_chunk(model, feature_files, window, coord_cols, is_rfrk, feature_names, mask=None):
    chunk_data = {Path(file).stem: get_memmap_array(file, window) for file in feature_files}
    
    rows, cols = next(iter(chunk_data.values())).shape
    feature_array = np.stack([chunk_data[f] for f in feature_names if f not in coord_cols], axis=-1)
    
    if mask is not None:
        feature_array[~mask] = np.nan
    
    valid_pixels = ~np.isnan(feature_array).any(axis=-1)
    feature_array = feature_array[valid_pixels]
    
    logger.info(f"有效像素数量: {np.sum(valid_pixels)}")
    
    result = np.full((rows, cols), np.nan)
    
    if np.sum(valid_pixels) > 0:
        if is_rfrk:
            coord_array = np.stack([chunk_data[c] for c in coord_cols], axis=-1)[valid_pixels]
            predictions = predict_chunk(model, feature_array, coord_array, is_rfrk, feature_names)
        else:
            predictions = predict_chunk(model, feature_array, None, is_rfrk, feature_names)
        
        result[valid_pixels] = predictions
    else:
        logger.warning("此区块没有有效像素进行预测")
    
    if mask is not None:
        result[~mask] = np.nan
    
    return result

def create_mask_from_shapefile(shapefile_path, raster_path):
    gdf = gpd.read_file(shapefile_path)
    with rasterio.open(raster_path) as src:
        geometries = [mapping(geom) for geom in gdf.geometry]
        mask = geometry_mask(geometries, out_shape=src.shape, transform=src.transform, invert=True)
    return mask

def predict_soil_property(model, feature_dir, output_path, coord_cols, is_rfrk, feature_names, shapefile_path=None, chunk_size=1000):
    logger.info("开始预测土壤属性")
    feature_files = list(Path(feature_dir).glob('*.tif'))
    with rasterio.open(feature_files[0]) as src:
        profile = src.profile
        height, width = src.shape

    mask = None
    if shapefile_path:
        mask = create_mask_from_shapefile(shapefile_path, feature_files[0])
        logger.info(f"使用shapefile创建了掩码")
        logger.info(f"掩码中有效像素数量: {np.sum(mask)}")
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for row in tqdm(range(0, height, chunk_size), desc="处理行"):
            for col in range(0, width, chunk_size):
                window = Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row))
                chunk_mask = None if mask is None else mask[row:row+window.height, col:col+window.width]
                chunk_result = process_raster_chunk(model, feature_files, window, coord_cols, is_rfrk, feature_names, chunk_mask)
                
                if chunk_mask is not None:
                    chunk_result[~chunk_mask] = dst.nodata  # 将掩膜外的区域设置为nodata值
                
                dst.write(chunk_result, 1, window=window)
    
    logger.info(f"预测结果已保存到: {output_path}")

def predict_soil_properties(model_dir, feature_dir, output_dir, training_data_path, coord_cols, use_rk=False, shapefile_path=None):
    model_dir, feature_dir, output_dir = map(Path, [model_dir, feature_dir, output_dir])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_files = list(model_dir.glob('*_model.pkl'))

    for model_file in tqdm(model_files, desc="预测土壤属性"):
        property_name = model_file.stem.replace('_model', '')
        logger.info(f"正在预测 {property_name}")

        model, feature_names = load_model(model_file)
        
        training_data = load_data(training_data_path, property_name)
        training_data = training_data.loc[:, ~training_data.columns.duplicated()]
        
        available_features = [f for f in feature_names if f in training_data.columns]
        if len(available_features) != len(feature_names):
            logger.warning(f"部分特征在训练数据中不可用。期望 {len(feature_names)} 个特征，实际可用 {len(available_features)} 个特征。")
            logger.warning(f"缺失的特征: {set(feature_names) - set(available_features)}")
        
        X = training_data[available_features + coord_cols]
        y = training_data[property_name]

        logger.info(f"模型期望的特征: {feature_names}")
        logger.info(f"实际使用的特征: {available_features}")
        logger.info(f"特征数量: 期望 {len(feature_names)}, 实际 {len(available_features)}")

        updated_model, is_rfrk = compare_models_and_train_kriging(X, y, model, available_features, coord_cols, use_rk)

        output_path = output_dir / f"{property_name}_prediction.tif"
        predict_soil_property(updated_model, feature_dir, output_path, coord_cols, is_rfrk, available_features, shapefile_path)

    logger.info("所有土壤属性预测完成")

if __name__ == "__main__":
    model_dir = Path(r"C:\Users\Runker\Desktop\GL\rfrk\models")
    feature_dir = Path(r"C:\Users\Runker\Desktop\GL\gl_tif_aligin")
    output_dir = Path(r"C:\Users\Runker\Desktop\GL\gl_tif_properte_predict")
    training_data_path = Path(r"C:\Users\Runker\Desktop\GL\sample_csv\feature_gl.csv")
    coord_cols = ['a_lon', 'a_lat']
    use_rk = False
    shapefile_path = Path(r"C:\Users\Runker\Desktop\GL\gl\gl_extent_500m.shp")  # 可选，如果不需要限制区域，设为 None
    # 预测所有土壤属性
    # predict_soil_properties(model_dir, feature_dir, output_dir, training_data_path, coord_cols, use_rk)
    # 预测特定区域土壤属性
    predict_soil_properties(model_dir, feature_dir, output_dir, training_data_path, coord_cols, use_rk, shapefile_path)
