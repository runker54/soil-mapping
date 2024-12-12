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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import gc
import psutil

class SoilPropertyPredictor:
    def __init__(self, log_file, model_dir, feature_dir, output_dir, training_data_path, coord_cols, use_rk=False, shapefile_path=None, chunk_size=1000, max_workers=8, batch_size=100, enable_uncertainty_viz=True):
        self.logger = self._setup_logger(log_file)
        self.model_dir = Path(model_dir)
        self.feature_dir = Path(feature_dir)
        self.output_dir = Path(output_dir)
        self.training_data_path = training_data_path
        self.coord_cols = coord_cols
        self.use_rk = use_rk
        self.shapefile_path = shapefile_path
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_uncertainty_viz = enable_uncertainty_viz
        plt.switch_backend('Agg')

    def _setup_logger(self, log_file):
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
        return logging.getLogger(__name__)

    def load_model(self, model_path):
        self.logger.info(f"正在加载模型: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            if not isinstance(model_data, dict):
                raise ValueError("模型文件格式错误：不是字典格式")
                
            required_keys = ['model', 'feature_names']
            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                raise ValueError(f"模型文件缺少必要信息: {missing_keys}")
                
            model = model_data['model']
            feature_names = [str(f) for f in model_data['feature_names']]
            n_features = model.n_features_in_
            
            self.logger.info(f"模型类型: {type(model).__name__}")
            self.logger.info(f"特征数量: {n_features}")
            self.logger.info(f"特征名称: {', '.join(feature_names[:5])}...")
            
            return model, feature_names[:n_features]
            
        except Exception as e:
            self.logger.error(f"加载模型时发生错误: {str(e)}")
            raise

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
        """可视化不确定性分布"""
        try:
            # 创建新的图形
            plt.clf()
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
            
            # 保存图形
            output_path = self.output_dir / f'{property_name}_uncertainty_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')  # 确保关闭所有图形
            
            self.logger.info(f"不确定性分布图已保存至: {output_path}")
            
        except Exception as e:
            self.logger.error(f"生成不确定性分布图时发生错误: {str(e)}")
            plt.close('all')  # 确保出错时也关闭所有图形

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

    def predict_soil_property(self, model, feature_names, property_name, is_rfrk):
        try:
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"初始内存使用: {initial_memory:.2f} MB")
            
            self.logger.info(f"开始预测土壤属性: {property_name}")
            self.logger.info(f"使用特征数量: {len(feature_names)}")
            self.logger.info(f"预测模式: {'RFRK' if is_rfrk else 'RF'}")
            
            feature_files = list(self.feature_dir.glob('*.tif'))
            self.logger.info(f"找到特征文件数量: {len(feature_files)}")
            
            profile, (height, width), transform = self.get_raster_info(feature_files[0])
            self.logger.info(f"栅格大小: {width}x{height}")
            
            # 优化chunk_size以减少内存使用
            chunk_size = min(self.chunk_size, 2000)  # 限制最大chunk大小
            
            mask = self.create_mask_from_shapefile(self.shapefile_path, feature_files[0]) if self.shapefile_path else None
            self.logger.info(f"创建掩膜完成")
            chunks = [
                (model, feature_files, Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row)),
                 self.coord_cols, is_rfrk, feature_names, mask)
                for row in range(0, height, chunk_size) 
                for col in range(0, width, chunk_size)
            ]
            self.logger.info(f"创建数据块完成")
            output_path = self.output_dir / f"{property_name}_prediction.tif"
            uncertainty_path = self.output_dir / f"{property_name}_uncertainty.tif"
            self.logger.info(f"创建输出文件完成")
            # 计算最大进程数
            max_workers = min(os.cpu_count(), self.max_workers)  # 限制最大进程数
            self.logger.info(f"最大进程数: {max_workers}")
            
            # 确保输出目录存在
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 如果文件已存在，先删除
            if output_path.exists():
                output_path.unlink()
            if uncertainty_path.exists():
                uncertainty_path.unlink()
            
            # 确保profile包含正确的数据类型
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                nodata=np.nan
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst, \
                 rasterio.open(uncertainty_path, 'w', **profile) as uncertainty_dst:
                
                total_chunks = len(chunks)
                self.logger.info(f"总计需要处理 {total_chunks} 个数据块")
                
                # 直接处理所有chunks，不进行分批
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self.process_raster_chunk_with_uncertainty, chunk) 
                        for chunk in chunks
                    ]
                    
                    # 处理结果
                    completed = 0
                    failed = 0
                    for future in tqdm(as_completed(futures), 
                                     total=len(futures), 
                                     desc=f"处理 {property_name}",
                                     ncols=100):
                        try:
                            window, chunk_result, chunk_uncertainty = future.result(timeout=600)
                            dst.write(chunk_result, 1, window=window)
                            uncertainty_dst.write(chunk_uncertainty, 1, window=window)
                            completed += 1
                        except Exception as e:
                            failed += 1
                            self.logger.error(f"处理chunk失败: {str(e)}")
                            continue
            
            # 清理内存
            del chunks
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"当前内存使用: {current_memory:.2f} MB")
            self.logger.info(f"内存增长: {current_memory - initial_memory:.2f} MB")
            
            self.logger.info(f"属性 {property_name} 预测完成")
            self.logger.info(f"输出文件保存至: {output_path}")
            self.logger.info(f"不确定性文件保存至: {uncertainty_path}")
            
            return output_path, uncertainty_path
            
        except Exception as e:
            self.logger.error(f"预测属性 {property_name} 时发生错误: {str(e)}")
            raise

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
            
            # 使用所有树进行预测
            tree_predictions = np.array([tree.predict(chunk) for tree in model.estimators_], dtype=np.float32)
            
            predictions[i:i+max_chunk_size] = np.mean(tree_predictions, axis=0)
            uncertainties[i:i+max_chunk_size] = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainties

    @staticmethod
    def process_raster_chunk_with_uncertainty(args):
        """处理栅格块并计算不确定性"""
        try:
            model, feature_files, window, coord_cols, is_rfrk, feature_names, mask = args
            chunk_data = {}
            
            # 使用上下文管理器确保资源释放
            for file in feature_files:
                with rasterio.open(file) as src:
                    data = src.read(1, window=window)
                    # 立即转换为float32以节省内存
                    chunk_data[Path(file).stem] = data.astype(np.float32)
                    del data  # 显式释放内存
            
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
                    predictions, uncertainties = SoilPropertyPredictor.predict_chunk_with_uncertainty_rfrk(
                        model, feature_array, coord_array)
                else:
                    predictions, uncertainties = SoilPropertyPredictor.predict_chunk_with_uncertainty_rf(
                        model, feature_array)
                
                result[valid_pixels] = predictions
                uncertainty[valid_pixels] = uncertainties
            
            # 清理不需要的数据
            del feature_array
            gc.collect()
            
            return window, result, uncertainty
            
        except Exception as e:
            logging.error(f"处理数据块时发生错误: {str(e)}")
            raise

    @staticmethod
    def predict_chunk_with_uncertainty_rfrk(model, feature_data, coord_data):
        predictions = model.predict(feature_data, coord_data)
        
        # 对于RFRK，我们可以使用kriging方差作为不确定性度量
        # 注意：这需要RegressionKriging类有一个predict_variance方法
        if hasattr(model, 'predict_variance'):
            uncertainties = model.predict_variance(feature_data, coord_data)
        else:
            # 如果没有predict_variance方法，我们可以使用基础随机森林模型的不确定性
            rf_model = model.regression_model
            _, uncertainties = SoilPropertyPredictor.predict_chunk_with_uncertainty_rf(rf_model, feature_data)
        
        return predictions, uncertainties

    def predict_soil_properties(self):
        self.logger.info("开始批量预测土壤属性")
        self.logger.info(f"输出目录: {self.output_dir}")
        
        model_files = list(self.model_dir.glob('*_model.pkl'))
        self.logger.info(f"找到模型文件数量: {len(model_files)}")
        
        for model_file in tqdm(model_files, 
                              desc="预测进度", 
                              ncols=100,
                              position=0):
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

            output_path, uncertainty_path = self.predict_soil_property(updated_model, available_features, property_name, is_rfrk)
            
            # 根据参数决定是否执行不确定性分析和可视化
            if self.enable_uncertainty_viz:
                try:
                    # 计算不确定性统计信息
                    statistics, valid_data = self.calculate_uncertainty_statistics(uncertainty_path)
                    
                    # 生成可视化
                    self.visualize_uncertainty_distribution(property_name, valid_data, statistics)
                    
                    # 创建分位数栅格
                    quantile_values = self.create_quantile_raster(uncertainty_path, property_name)
                    
                    self.logger.info(f"{property_name} 的不确定性分析和可视化完成")
                    
                except Exception as e:
                    self.logger.error(f"生成 {property_name} 的不确定性分析和可视化时发生错误: {str(e)}")

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
    log_file = r"F:\soil_mapping\dy\soil-mapping\logs\predict_soil_properties.log"
    model_dir = r"F:\soil_mapping\dy\figures\model\soil_property\models"
    feature_dir = r"F:\tif_features\county_feature\dy"
    output_dir = r"F:\soil_mapping\dy\figures\soil_property_predict"
    training_data_path = r'F:\soil_mapping\dy\figures\properoty_table\soil_property_point.csv'
    coord_cols = ["lon", "lat"]
    use_rk = True
    chunk_size = 1000
    max_workers = 12
    batch_size = 100
    shapefile_path = r"F:\cache_data\shp_file\dy\dy_studyarea.shp"
    # 添加不确定性可视化参数
    enable_uncertainty_viz = True  # 设置为 True 启用可视化，False 禁用
    
    predictor = SoilPropertyPredictor(
        log_file, model_dir, feature_dir, output_dir, 
        training_data_path, coord_cols, use_rk, shapefile_path, 
        chunk_size=2000, max_workers=12, batch_size=100,
        enable_uncertainty_viz=enable_uncertainty_viz  # 传入参数
    )
    predictor.run()
