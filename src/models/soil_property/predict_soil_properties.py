import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pykrige.rk import RegressionKriging
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class SoilPropertyPredictor:
    def __init__(self, log_file, model_dir, feature_dir, output_dir, training_data_path, coord_cols, use_rk=False, shapefile_path=None):
        self.logger = self._setup_logger(log_file)
        self.model_dir = Path(model_dir)
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path(output_dir)
        self.training_data_path = training_data_path
        self.coord_cols = coord_cols
        self.use_rk = use_rk
        self.shapefile_path = shapefile_path

    def _setup_logger(self, log_file):
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
        return logging.getLogger(__name__)

    def load_model(self, model_path):
        self.logger.info(f"正在加载模型: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data:
            model = model_data['model']
            feature_names = [str(f) for f in model_data['feature_names']]
            n_features = model.n_features_in_
            self.logger.info(f"加载的模型期望 {n_features} 个特征")
            return model, feature_names[:n_features]
        else:
            raise ValueError("模型文件格式不正确，缺少模型或特征名称信息")

    def load_data(self, file_path, label_col):
        self.logger.info(f"正在从 {file_path} 加载数据")
        data = pd.read_csv(file_path)
        numeric_cols = data.select_dtypes(include=[np.number])
        data[numeric_cols.columns] = data[numeric_cols.columns].fillna(numeric_cols.median())
        data = data[data[f"{label_col}_Sta"] == 'Normal']
        self.logger.info(f"数据加载完成。形状：{data.shape}")
        return data

    def compare_models_and_train_kriging(self, X, y, rf_model, feature_names, test_size=0.2, random_state=42):
        X_model = X[feature_names]
        X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size, random_state=random_state)

        rf_predictions = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_predictions)
        if self.use_rk:
            X_full = X[feature_names + self.coord_cols].astype(float)
            X_train_full, X_test_full, _, _ = train_test_split(X_full, y, test_size=test_size, random_state=random_state)

            rk = RegressionKriging(
                regression_model=rf_model,
                n_closest_points=8,
                variogram_model='linear',
                exact_values=False
            )
            self.logger.info("开始训练RFRK模型")
            rk.fit(
                X_train_full[feature_names].values,
                X_train_full[self.coord_cols].values,
                y_train
            )
            self.logger.info("RFRK模型训练完成")
            rfrk_predictions = rk.predict(X_test_full[feature_names].values, X_test_full[self.coord_cols].values)
            rfrk_r2 = r2_score(y_test, rfrk_predictions)

            self.logger.info(f"RF R2 score: {rf_r2}")
            self.logger.info(f"RFRK R2 score: {rfrk_r2}")
            return (rk, True) if rfrk_r2 > rf_r2 else (rf_model, False)
        else:
            return (rf_model, False)

    @staticmethod
    def predict_chunk(model, feature_data, coord_data, is_rfrk):
        if feature_data.shape[0] == 0:
            return np.array([])
        
        feature_data = feature_data.astype(float)
        if is_rfrk:
            coord_data = coord_data.astype(float)
            return model.predict(feature_data, coord_data)
        else:
            return model.predict(feature_data)

    @staticmethod
    def get_raster_info(raster_path):
        with rasterio.open(raster_path) as src:
            return src.profile, src.shape, src.transform

    @staticmethod
    def create_mask_from_shapefile(shapefile_path, raster_path):
        gdf = gpd.read_file(shapefile_path)
        with rasterio.open(raster_path) as src:
            geometries = [mapping(geom) for geom in gdf.geometry]
            mask = geometry_mask(geometries, out_shape=src.shape, transform=src.transform, invert=True)
        return mask

    @staticmethod
    def process_raster_chunk(args):
        model, feature_files, window, coord_cols, is_rfrk, feature_names, mask = args
        chunk_data = {}
        for file in feature_files:
            with rasterio.open(file) as src:
                chunk_data[Path(file).stem] = src.read(1, window=window)
        
        rows, cols = next(iter(chunk_data.values())).shape
        feature_array = np.stack([chunk_data[f] for f in feature_names if f not in coord_cols], axis=-1)
        
        if mask is not None:
            chunk_mask = mask[window.row_off:window.row_off+window.height, 
                              window.col_off:window.col_off+window.width]
            feature_array[~chunk_mask] = np.nan
        
        valid_pixels = ~np.isnan(feature_array).any(axis=-1)
        feature_array = feature_array[valid_pixels]
        
        result = np.full((rows, cols), np.nan)
        
        if np.sum(valid_pixels) > 0:
            if is_rfrk:
                coord_array = np.stack([chunk_data[c] for c in coord_cols], axis=-1)[valid_pixels]
                predictions = SoilPropertyPredictor.predict_chunk(model, feature_array, coord_array, is_rfrk)
            else:
                predictions = SoilPropertyPredictor.predict_chunk(model, feature_array, None, is_rfrk)
            
            result[valid_pixels] = predictions
        
        return window, result

    def calculate_uncertainty_statistics(self, uncertainty_path):
        with rasterio.open(uncertainty_path) as src:
            uncertainty_data = src.read(1)
            valid_data = uncertainty_data[~np.isnan(uncertainty_data)]
            
            statistics = {
                "最小值": np.min(valid_data),
                "最大值": np.max(valid_data),
                "平均值": np.mean(valid_data),
                "中位数": np.median(valid_data),
                "标准差": np.std(valid_data),
                "四分位数": np.percentile(valid_data, [25, 50, 75]),
                "十分位数": np.percentile(valid_data, range(10, 101, 10))
            }
            
            return statistics, valid_data

    def visualize_uncertainty_distribution(self, property_name, valid_data, statistics):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # 直方图和核密度估计
        sns.histplot(valid_data, kde=True, ax=ax1, color='skyblue')
        ax1.set_title(f'{property_name} Uncertainty Distribution', fontsize=16)
        ax1.set_xlabel('Uncertainty Value', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        
        # 在图上添加统计信息
        textstr = '\n'.join((
            f'Mean: {statistics["平均值"]:.2f}',
            f'Median: {statistics["中位数"]:.2f}',
            f'Std Dev: {statistics["标准差"]:.2f}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        # 箱线图
        sns.boxplot(x=valid_data, ax=ax2, color='lightgreen')
        ax2.set_title(f'{property_name} Uncertainty Boxplot', fontsize=16)
        ax2.set_xlabel('Uncertainty Value', fontsize=12)
        
        # 在箱线图上标记四分位数和中位数
        quartiles = statistics["四分位数"]
        for i, quartile in enumerate(quartiles):
            ax2.axvline(x=quartile, color='r', linestyle='--', alpha=0.7)
            ax2.text(quartile, 0.5, f'Q{i+1}: {quartile:.2f}', rotation=90, 
                     verticalalignment='center', horizontalalignment='right')
        
        # 添加图例
        ax2.axvline(x=quartiles[1], color='r', linestyle='--', alpha=0.7, label='Quartiles')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{property_name}_uncertainty_distribution.png', dpi=300)
        plt.close()

    def create_quantile_raster(self, uncertainty_path, property_name):
        with rasterio.open(uncertainty_path) as src:
            uncertainty_data = src.read(1)
            profile = src.profile
        
        quantiles = [0.2, 0.4, 0.6, 0.8]
        quantile_values = np.nanquantile(uncertainty_data, quantiles)
        
        quantile_raster = np.digitize(uncertainty_data, quantile_values)
        
        output_path = self.output_dir / f'{property_name}_uncertainty_quantiles.tif'
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(quantile_raster, 1)
        
        self.logger.info(f"不确定性分位数栅格已保存到: {output_path}")
        return quantile_values

    def predict_soil_property(self, model, feature_names, property_name, is_rfrk, chunk_size=1000):
        self.logger.info(f"开始预测土壤属性: {property_name}")
        feature_files = list(self.feature_dir.glob('*.tif'))
        profile, (height, width), transform = self.get_raster_info(feature_files[0])

        mask = self.create_mask_from_shapefile(self.shapefile_path, feature_files[0]) if self.shapefile_path else None
        
        chunks = [
            (model, feature_files, Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row)),
             self.coord_cols, is_rfrk, feature_names, mask)
            for row in range(0, height, chunk_size) 
            for col in range(0, width, chunk_size)
        ]

        output_path = self.output_dir / f"{property_name}_prediction.tif"
        uncertainty_path = self.output_dir / f"{property_name}_uncertainty.tif"
        
        with rasterio.open(output_path, 'w', **profile) as dst, \
             rasterio.open(uncertainty_path, 'w', **profile) as uncertainty_dst:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.process_raster_chunk_with_uncertainty, chunk) for chunk in chunks]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chunks"):
                    window, chunk_result, chunk_uncertainty = future.result()
                    dst.write(chunk_result, 1, window=window)
                    uncertainty_dst.write(chunk_uncertainty, 1, window=window)
        
        self.logger.info(f"预测结果已保存到: {output_path}")
        self.logger.info(f"不确定性结果已保存到: {uncertainty_path}")
        
        # 计算不确定性统计量并可视化
        statistics, valid_data = self.calculate_uncertainty_statistics(uncertainty_path)
        self.logger.info(f"{property_name} 不确定性统计量: {statistics}")
        
        self.visualize_uncertainty_distribution(property_name, valid_data, statistics)
        
        # 创建分位数栅格图
        quantile_values = self.create_quantile_raster(uncertainty_path, property_name)
        self.logger.info(f"{property_name} 不确定性分位数值: {quantile_values}")

    @staticmethod
    def predict_chunk_with_uncertainty_rf(model, feature_data, max_chunk_size=100000):
        if not isinstance(model, RandomForestRegressor):
            raise ValueError("此方法仅适用于随机森林回归器")
        
        # 将数据类型转换为float32以减少内存使用
        feature_data = feature_data.astype(np.float32)
        
        # 初始化结果数组
        predictions = np.zeros(feature_data.shape[0], dtype=np.float32)
        uncertainties = np.zeros(feature_data.shape[0], dtype=np.float32)
        
        # 分块处理数据
        for i in range(0, feature_data.shape[0], max_chunk_size):
            chunk = feature_data[i:i+max_chunk_size]
            
            # 使用前10棵树进行预测
            tree_predictions = np.array([tree.predict(chunk) for tree in model.estimators_], dtype=np.float32)
            
            predictions[i:i+max_chunk_size] = np.mean(tree_predictions, axis=0)
            uncertainties[i:i+max_chunk_size] = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainties

    @staticmethod
    def process_raster_chunk_with_uncertainty(args):
        model, feature_files, window, coord_cols, is_rfrk, feature_names, mask = args
        chunk_data = {}
        for file in feature_files:
            with rasterio.open(file) as src:
                chunk_data[Path(file).stem] = src.read(1, window=window).astype(np.float32)
        
        rows, cols = next(iter(chunk_data.values())).shape
        feature_array = np.stack([chunk_data[f] for f in feature_names if f not in coord_cols], axis=-1)
        
        if mask is not None:
            chunk_mask = mask[window.row_off:window.row_off+window.height, 
                            window.col_off:window.col_off+window.width]
            feature_array[~chunk_mask] = np.nan
        
        valid_pixels = ~np.isnan(feature_array).any(axis=-1)
        feature_array = feature_array[valid_pixels]
        
        result = np.full((rows, cols), np.nan, dtype=np.float32)
        uncertainty = np.full((rows, cols), np.nan, dtype=np.float32)
        
        if np.sum(valid_pixels) > 0:
            if is_rfrk:
                coord_array = np.stack([chunk_data[c] for c in coord_cols], axis=-1)[valid_pixels]
                predictions, uncertainties = SoilPropertyPredictor.predict_chunk_with_uncertainty_rfrk(model, feature_array, coord_array)
            else:
                predictions, uncertainties = SoilPropertyPredictor.predict_chunk_with_uncertainty_rf(model, feature_array)
            
            result[valid_pixels] = predictions
            uncertainty[valid_pixels] = uncertainties
        
        return window, result, uncertainty

    @staticmethod
    def predict_chunk_with_uncertainty_rfrk(model, feature_data, coord_data):
        predictions = model.predict(feature_data, coord_data)
        
        # 对于RFRK，我们可以使用kriging方差作为不确定性度量
        # 注意：这需要RegressionKriging类有一个predict_variance方法
        if hasattr(model, 'predict_variance'):
            uncertainties = model.predict_variance(feature_data, coord_data)
        else:
            # 如果没有predict_variance方法，我们可以使用一个替代方法
            # 例如，使用基础随机森林模型的不确定性
            rf_model = model.regression_model
            _, uncertainties = SoilPropertyPredictor.predict_chunk_with_uncertainty_rf(rf_model, feature_data)
        
        return predictions, uncertainties

    def predict_soil_properties(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        model_files = list(self.model_dir.glob('*_model.pkl'))
        for model_file in tqdm(model_files, desc="预测土壤属性"):
            property_name = model_file.stem.replace('_model', '')
            self.logger.info(f"正在预测 {property_name}")

            model, feature_names = self.load_model(model_file)
            
            training_data = self.load_data(self.training_data_path, property_name)
            training_data = training_data.loc[:, ~training_data.columns.duplicated()]
            
            available_features = [f for f in feature_names if f in training_data.columns]
            if len(available_features) != len(feature_names):
                self.logger.warning(f"部分特征在训练数据中不可用。期望 {len(feature_names)} 个特征，实际可用 {len(available_features)} 个特征。")
                self.logger.warning(f"缺失的特征: {set(feature_names) - set(available_features)}")
            
            X = training_data[available_features + self.coord_cols]
            y = training_data[property_name]

            updated_model, is_rfrk = self.compare_models_and_train_kriging(X, y, model, available_features)

            self.predict_soil_property(updated_model, available_features, property_name, is_rfrk)

        self.logger.info("所有土壤属性预测完成")

    def run(self):
        self.logger.info("开始预测土壤属性")
        try:
            self.predict_soil_properties()
            self.logger.info("土壤属性预测完成")
        except Exception as e:
            self.logger.error(f"预测土壤属性过程中发生错误: {str(e)}")
            raise

if __name__ == "__main__":
    log_file = r"D:\soil-mapping\logs\predict_soil_properties.log"
    model_dir = r"D:\soil-mapping\models\soil_property\models"
    feature_dir = r"D:\soil-mapping\data\soil_property"
    output_dir = r"D:\soil-mapping\data\soil_property_predict"
    training_data_path = r"D:\soil-mapping\data\soil_property_table\soil_property_point.csv"
    coord_cols = ["a_lon", "a_lat"]
    use_rk = True
    shapefile_path = r"D:\soil-mapping\data\raw\study_area\studyarea_500_s.shp"

    predictor = SoilPropertyPredictor(log_file, model_dir, feature_dir, output_dir, training_data_path, coord_cols, use_rk, shapefile_path)
    predictor.run()