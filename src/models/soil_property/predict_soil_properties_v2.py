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
from sklearn.metrics import r2_score, mean_squared_error
from pykrige.rk import RegressionKriging
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import gc
import psutil
import time
from functools import wraps

# ================ 默认配置 ================
DEFAULT_CONFIG = {
    'paths': {
        'log_file': r"F:\soil_mapping\qz\soil-mapping\logs\predict_soil_properties_2.log",
        'model_dir': r"F:\soil_mapping\qz\figures\model\soil_property\models",
        'feature_dir': r"F:\tif_features\county_feature\qz",
        'output_dir':  r"F:\soil_mapping\qz\figures\soil_property_predict_2",
        'training_data_path': r'F:\soil_mapping\qz\figures\properoty_table\soil_property_point.csv',
        'shapefile_path': r"F:\cache_data\shp_file\qz\qz_extent_p_500.shp"
    },
    'model_params': {
        'coord_cols': ["lon", "lat"],
        'use_rk': True,
        'chunk_size': 2000,
        'max_workers': 12,
        'batch_size': 100,
        'enable_uncertainty_viz': True
    },
    'data_processing': {
        'max_chunk_size': 100000,
        'dtype': 'float32',
        'nodata_value': -9999
    },
    'visualization': {
        'dpi': 300,
        'figure_size': [12, 16],
        'font_size': {
            'title': 16,
            'label': 12,
            'text': 10
        }
    },
    'logging': {
        'level': 'INFO',
        'format': "%(asctime)s - %(levelname)s - %(message)s",
        'encoding': "utf-8"
    }
}

# ================ 自定义异常类 ================

class SoilMappingError(Exception):
    """土壤制图基础异常类"""
    pass

class ModelError(SoilMappingError):
    """模型相关错误"""
    pass

class DataError(SoilMappingError):
    """数据处理相关错误"""
    pass

class PredictionError(SoilMappingError):
    """预测过程相关错误"""
    pass

class ValidationError(SoilMappingError):
    """数据验证错误"""
    pass

class VisualizationError(SoilMappingError):
    """可视化相关错误"""
    pass

# ================ 工具函数和装饰器 ================

def memory_monitor(func):
    """监控函数内存使用的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            end_memory = process.memory_info().rss
            end_time = time.time()
            
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            time_used = end_time - start_time
            
            logging.info(
                f"函数 {func.__name__} 执行完成:\n"
                f"- 内存使用: {memory_used:.2f} MB\n"
                f"- 执行时间: {time_used:.2f} 秒"
            )
            
            return result
            
        except Exception as e:
            logging.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
            
    return wrapper

def validate_array(array, name="array"):
    """验证numpy数组的有效性"""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} 必须是numpy数组")
    if np.isnan(array).all():
        raise ValueError(f"{name} 包含全部为NaN的值")
    if array.size == 0:
        raise ValueError(f"{name} 是空数组")
    return True

def optimize_array(array):
    """优化数组存储和类型"""
    return np.ascontiguousarray(array, dtype=np.float32)

class MemoryManager:
    """内存管理工具类"""
    @staticmethod
    def clear_memory():
        """清理内存"""
        gc.collect()
        
    @staticmethod
    def get_memory_usage():
        """获取当前内存使用情况"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
        
    @staticmethod
    def monitor_memory(threshold_mb=1000):
        """监控内存使用"""
        if MemoryManager.get_memory_usage() > threshold_mb:
            logging.warning(f"内存使用超过阈值: {threshold_mb}MB")
            MemoryManager.clear_memory()

class PathManager:
    """路径管理工具类"""
    @staticmethod
    def ensure_dir(path):
        """确保目录存在"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def validate_path(path, check_exists=True):
        """验证路径"""
        path = Path(path)
        if check_exists and not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        return path
# ================ 可视化类 ================

class SoilPropertyVisualizer:
    """土壤属性可视化工具类"""
    
    def __init__(self, config):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        plt.switch_backend('Agg')
        
    def plot_uncertainty_distribution(self, property_name, valid_data, statistics, output_path):
        """
        绘制不确定性分布图
        
        Args:
            property_name: 属性名称
            valid_data: 有效数据
            statistics: 统计信息字典
            output_path: 输出路径
        """
        try:
            plt.clf()
            fig_size = self.config['visualization']['figure_size']
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)
            
            # 直方图和核密度估计
            sns.histplot(valid_data, kde=True, ax=ax1, color='skyblue')
            ax1.set_title(f'{property_name} Uncertainty Distribution', 
                         fontsize=self.config['visualization']['font_size']['title'])
            ax1.set_xlabel('Uncertainty Value', 
                          fontsize=self.config['visualization']['font_size']['label'])
            ax1.set_ylabel('Density', 
                          fontsize=self.config['visualization']['font_size']['label'])
            
            # 统计信息
            textstr = '\n'.join((
                f'Mean: {statistics["平均值"]:.2f}',
                f'Median: {statistics["中位数"]:.2f}',
                f'Std Dev: {statistics["标准差"]:.2f}'
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.05, 0.95, textstr, 
                    transform=ax1.transAxes, 
                    fontsize=self.config['visualization']['font_size']['text'],
                    verticalalignment='top', 
                    bbox=props)
            
            # 箱线图
            sns.boxplot(x=valid_data, ax=ax2, color='lightgreen')
            ax2.set_title(f'{property_name} Uncertainty Boxplot', 
                         fontsize=self.config['visualization']['font_size']['title'])
            ax2.set_xlabel('Uncertainty Value', 
                          fontsize=self.config['visualization']['font_size']['label'])
            
            # 四分位数标记
            quartiles = statistics["四分位数"]
            for i, quartile in enumerate(quartiles):
                ax2.axvline(x=quartile, color='r', linestyle='--', alpha=0.7)
                ax2.text(quartile, 0.5, f'Q{i+1}: {quartile:.2f}', 
                        rotation=90, 
                        verticalalignment='center', 
                        horizontalalignment='right')
            
            plt.tight_layout()
            plt.savefig(output_path, 
                       dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            plt.close('all')
            
        except Exception as e:
            raise VisualizationError(f"生成不确定性分布图失败: {str(e)}")
            
    def plot_prediction_validation(self, actual, predicted, property_name, output_path):
        """
        绘制预测验证图
        
        Args:
            actual: 实际值
            predicted: 预测值
            property_name: 属性名称
            output_path: 输出路径
        """
        try:
            plt.figure(figsize=self.config['visualization']['figure_size'])
            
            # 散点图
            plt.scatter(actual, predicted, alpha=0.5)
            
            # 1:1线
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # 计算R²
            r2 = np.corrcoef(actual, predicted)[0, 1]**2
            
            plt.xlabel('Actual Values', 
                      fontsize=self.config['visualization']['font_size']['label'])
            plt.ylabel('Predicted Values', 
                      fontsize=self.config['visualization']['font_size']['label'])
            plt.title(f'{property_name} Prediction Validation (R² = {r2:.3f})', 
                     fontsize=self.config['visualization']['font_size']['title'])
            
            plt.savefig(output_path, 
                       dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            raise VisualizationError(f"生成预测验证图失败: {str(e)}")
# ================ 主预测类 ================
class SoilPropertyPredictor:
    """土壤属性预测器"""
    
    def __init__(self, config_path=None):
        """
        初始化土壤属性预测器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self._init_paths()
        self._init_parameters()
        self.visualizer = SoilPropertyVisualizer(self.config)
        
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            if config_path is None:
                return DEFAULT_CONFIG
                
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                
            # 合并用户配置和默认配置
            config = DEFAULT_CONFIG.copy()
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
                    
            return config
            
        except Exception as e:
            raise SoilMappingError(f"加载配置文件失败: {str(e)}")
            
    def _setup_logger(self):
        """设置日志系统"""
        log_config = self.config['logging']
        log_path = Path(self.config['paths']['log_file'])
        
        PathManager.ensure_dir(log_path.parent)
        
        logging.basicConfig(
            filename=str(log_path),
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            encoding=log_config['encoding']
        )
        
        return logging.getLogger(__name__)
        
    def _init_paths(self):
        """初始化路径"""
        paths = self.config['paths']
        self.model_dir = PathManager.validate_path(paths['model_dir'])
        self.feature_dir = PathManager.validate_path(paths['feature_dir'])
        self.output_dir = Path(paths['output_dir'])
        self.training_data_path = PathManager.validate_path(paths['training_data_path'])
        self.shapefile_path = PathManager.validate_path(paths['shapefile_path'])
        
        PathManager.ensure_dir(self.output_dir)
        
    def _init_parameters(self):
        """初始化模型参数"""
        params = self.config['model_params']
        self.coord_cols = params['coord_cols']
        self.use_rk = params['use_rk']
        self.chunk_size = params['chunk_size']
        self.max_workers = params['max_workers']
        self.batch_size = params['batch_size']
        self.enable_uncertainty_viz = params['enable_uncertainty_viz']
        
    @memory_monitor
    def load_model(self, model_path):
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            tuple: (模型对象, 特征名称列表)
        """
        self.logger.info(f"正在加载模型: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            if not isinstance(model_data, dict):
                raise ModelError("模型文件格式错误：不是字典格式")
                
            required_keys = ['model', 'feature_names']
            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                raise ModelError(f"模型文件缺少必要信息: {missing_keys}")
                
            model = model_data['model']
            feature_names = [str(f) for f in model_data['feature_names']]
            n_features = model.n_features_in_
            
            self.logger.info(f"模型类型: {type(model).__name__}")
            self.logger.info(f"特征数量: {n_features}")
            self.logger.info(f"特征名称: {', '.join(feature_names[:5])}...")
            
            return model, feature_names[:n_features]
            
        except Exception as e:
            raise ModelError(f"加载模型时发生错误: {str(e)}")
    @memory_monitor
    def load_data(self, file_path, label_col):
        """
        加载训练数据
        
        Args:
            file_path: 数据文件路径
            label_col: 标签列名
            
        Returns:
            pd.DataFrame: 处理���的数据框
        """
        self.logger.info(f"正在从 {file_path} 加载数据")
        
        try:
            data = pd.read_csv(file_path)
            
            # 数据验证
            if data.empty:
                raise DataError("加载的数据为空")
                
            if label_col not in data.columns:
                raise DataError(f"数据中缺少标签列: {label_col}")
                
            # 处理数值列的缺失值
            numeric_cols = data.select_dtypes(include=[np.number])
            data[numeric_cols.columns] = data[numeric_cols.columns].fillna(numeric_cols.median())
            
            # 筛选正常数据
            data = data[data[f"{label_col}_Sta"] == 'Normal']
            
            if data.empty:
                raise DataError("筛选后的数据为空")
                
            self.logger.info(f"数据加载完成。形状：{data.shape}")
            return data
            
        except Exception as e:
            raise DataError(f"加载数据时发生错误: {str(e)}")

    @memory_monitor
    def compare_models_and_train_kriging(self, X, y, rf_model, feature_names, test_size=0.2, random_state=42):
        """
        比较RF和RFRK模型性能并训练Kriging
        
        Args:
            X: 特征数据
            y: 标签数据
            rf_model: 随机森林模型
            feature_names: 特征名称列表
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            tuple: (最佳模型, 是否使用RFRK)
        """
        try:
            X_model = X[feature_names]
            X_train, X_test, y_train, y_test = train_test_split(
                X_model, y, test_size=test_size, random_state=random_state
            )

            # 评估RF模型
            rf_predictions = rf_model.predict(X_test)
            rf_r2 = r2_score(y_test, rf_predictions)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
            
            self.logger.info(f"RF模型性能 - R2: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")

            if self.use_rk:
                try:
                    X_full = X[feature_names + self.coord_cols].astype(float)
                    X_train_full, X_test_full, _, _ = train_test_split(
                        X_full, y, test_size=test_size, random_state=random_state
                    )

                    # 训练RFRK模型
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

                    # 评估RFRK模型
                    rfrk_predictions = rk.predict(
                        X_test_full[feature_names].values,
                        X_test_full[self.coord_cols].values
                    )
                    rfrk_r2 = r2_score(y_test, rfrk_predictions)
                    rfrk_rmse = np.sqrt(mean_squared_error(y_test, rfrk_predictions))
                    
                    self.logger.info(f"RFRK模型性能 - R2: {rfrk_r2:.4f}, RMSE: {rfrk_rmse:.4f}")

                    # 如果启用了可视化，绘制验证图
                    if self.enable_uncertainty_viz:
                        # 绘制RFRK模型验证图
                        self.visualizer.plot_prediction_validation(
                            y_test, rfrk_predictions,
                            "RFRK Model",
                            self.output_dir / "rfrk_validation.png"
                        )
                        # 绘制RF模型验证图
                        self.visualizer.plot_prediction_validation(
                            y_test, rf_predictions,
                            "RF Model",
                            self.output_dir / "rf_validation.png"
                        )

                    return (rk, True) if rfrk_r2 > rf_r2 else (rf_model, False)
                    
                except Exception as e:
                    self.logger.warning(f"RFRK模型训练失败，将使用RF模型: {str(e)}")
                    return (rf_model, False)
            else:
                return (rf_model, False)
                
        except Exception as e:
            raise ModelError(f"模型比较和训练过程发生错误: {str(e)}")

    @staticmethod
    def predict_chunk_with_uncertainty(model, feature_data, coord_data=None, is_rfrk=False):
        """
        对数据块进行预测并计算不确定性
        
        Args:
            model: 预测模型
            feature_data: 特征数据
            coord_data: 坐标数据（RFRK模型需要）
            is_rfrk: 是否使用RFRK模型
            
        Returns:
            tuple: (预测值, 不确定性)
        """
        try:
            feature_data = optimize_array(feature_data)
            
            if is_rfrk:
                coord_data = optimize_array(coord_data)
                predictions = model.predict(feature_data, coord_data)
                uncertainties = (model.predict_variance(feature_data, coord_data) 
                               if hasattr(model, 'predict_variance') 
                               else np.zeros_like(predictions))
            else:
                if not isinstance(model, RandomForestRegressor):
                    raise ValueError("非RFRK模式下必须使用RandomForestRegressor")
                    
                tree_predictions = np.array([
                    tree.predict(feature_data)
                    for tree in model.estimators_
                ])
                
                predictions = np.mean(tree_predictions, axis=0)
                uncertainties = np.std(tree_predictions, axis=0)
                
            return predictions, uncertainties
            
        except Exception as e:
            raise PredictionError(f"数据块预测失败: {str(e)}")
    def _optimize_chunk_size(self, total_pixels):
        """动态优化数据块大小"""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        estimated_memory_per_pixel = 0.0001  # 估计每个像素需要的内存（MB）
        optimal_pixels = int(available_memory / estimated_memory_per_pixel / self.max_workers)
        return min(self.chunk_size, optimal_pixels)

    @staticmethod
    def process_raster_chunk(args):
        """优化的栅格数据块处理"""
        try:
            model, feature_files, window, coord_cols, is_rfrk, feature_names, mask = args
            chunk_data = {}
            
            # 分批读取特征数据以减少内存使用
            for file in feature_files:
                with rasterio.open(file) as src:
                    data = src.read(1, window=window)
                    if Path(file).stem in feature_names or Path(file).stem in coord_cols:
                        chunk_data[Path(file).stem] = data
                    del data
                    gc.collect()
            
            rows, cols = next(iter(chunk_data.values())).shape
            total_pixels = rows * cols

            # 使用float32而不是float64
            feature_array = np.zeros((total_pixels, len([f for f in feature_names if f not in coord_cols])), 
                                   dtype=np.float32)
            
            # 逐个填充特征数组
            for i, f in enumerate([f for f in feature_names if f not in coord_cols]):
                if f in chunk_data:
                    feature_array[:, i] = chunk_data[f].ravel()
                    del chunk_data[f]
            
            # 应用掩膜 - 修复这里的维度不匹配问题
            chunk_mask = None
            if mask is not None:
                chunk_mask = mask[
                    window.row_off:window.row_off+window.height,
                    window.col_off:window.col_off+window.width
                ].ravel()
                # 确保掩膜维度与特征数组匹配
                if len(chunk_mask) != total_pixels:
                    raise ValueError(f"掩膜维度 ({len(chunk_mask)}) 与特征数组维度 ({total_pixels}) 不匹配")
                feature_array = feature_array[chunk_mask]
            
            valid_pixels = ~np.isnan(feature_array).any(axis=1)
            feature_array = feature_array[valid_pixels]
            
            result = np.full((rows, cols), np.nan, dtype=np.float32)
            uncertainty = np.full((rows, cols), np.nan, dtype=np.float32)
            
            if len(feature_array) > 0:
                try:
                    # 分批预测以减少内存使用
                    batch_size = min(10000, len(feature_array))
                    predictions = []
                    uncertainties = []
                    
                    for i in range(0, len(feature_array), batch_size):
                        batch = feature_array[i:i+batch_size]
                        if is_rfrk:
                            # 确保坐标数据的维度匹配
                            if chunk_mask is not None:
                                coord_batch = np.stack([
                                    chunk_data[c].ravel()[chunk_mask][valid_pixels][i:i+batch_size] 
                                    for c in coord_cols
                                ], axis=-1)
                            else:
                                coord_batch = np.stack([
                                    chunk_data[c].ravel()[valid_pixels][i:i+batch_size] 
                                    for c in coord_cols
                                ], axis=-1)
                            pred, unc = SoilPropertyPredictor.predict_chunk_with_uncertainty(
                                model, batch, coord_batch, is_rfrk
                            )
                        else:
                            pred, unc = SoilPropertyPredictor.predict_chunk_with_uncertainty(
                                model, batch
                            )
                        predictions.extend(pred)
                        uncertainties.extend(unc)
                        
                        del batch
                        gc.collect()
                    
                    # 将结果转换回原始形状
                    if chunk_mask is not None:
                        # 创建临时数组来存储结果
                        temp_result = np.full(len(chunk_mask), np.nan)
                        temp_uncertainty = np.full(len(chunk_mask), np.nan)
                        
                        # 只在有效像素位置填充预测结果
                        valid_indices = np.where(chunk_mask)[0][valid_pixels]
                        temp_result[valid_indices] = predictions
                        temp_uncertainty[valid_indices] = uncertainties
                        
                        # 重塑为原始维度
                        result = temp_result.reshape((rows, cols))
                        uncertainty = temp_uncertainty.reshape((rows, cols))
                    else:
                        # 如果没有掩膜，直接重塑结果
                        valid_indices = np.where(valid_pixels)[0]
                        temp_result = np.full(total_pixels, np.nan)
                        temp_uncertainty = np.full(total_pixels, np.nan)
                        
                        temp_result[valid_indices] = predictions
                        temp_uncertainty[valid_indices] = uncertainties
                        
                        result = temp_result.reshape((rows, cols))
                        uncertainty = temp_uncertainty.reshape((rows, cols))
                
                except Exception as e:
                    logging.error(f"预测过程中发生错误: {str(e)}")
                    return window, np.full((rows, cols), np.nan), np.full((rows, cols), np.nan)
            
            # 清理内存
            del chunk_data, feature_array
            gc.collect()
            
            return window, result, uncertainty
            
        except Exception as e:
            raise PredictionError(f"栅格块处理失败: {str(e)}")

    @memory_monitor
    def predict_soil_property(self, model, feature_names, property_name, is_rfrk):
        """优化的土壤属性预测方法"""
        try:
            self.logger.info(f"开始预测土壤属性: {property_name}")
            
            feature_files = list(self.feature_dir.glob('*.tif'))
            self.logger.info(f"找到特征文件数量: {len(feature_files)}")
            
            with rasterio.open(feature_files[0]) as src:
                profile = src.profile.copy()
                height, width = src.shape
                transform = src.transform
            
            # 动态调整块大小
            optimal_chunk_size = self._optimize_chunk_size(height * width)
            self.logger.info(f"优化后的数据块大小: {optimal_chunk_size}")
            
            # 创建掩膜
            mask = None
            if self.shapefile_path:
                gdf = gpd.read_file(self.shapefile_path)
                with rasterio.open(feature_files[0]) as src:
                    geometries = [mapping(geom) for geom in gdf.geometry]
                    mask = geometry_mask(geometries, out_shape=(height, width), 
                                      transform=transform, invert=True)
            
            # 准备数据块
            chunks = []
            for row in range(0, height, optimal_chunk_size):
                for col in range(0, width, optimal_chunk_size):
                    window = Window(
                        col, row,
                        min(optimal_chunk_size, width - col),
                        min(optimal_chunk_size, height - row)
                    )
                    chunks.append((
                        model, feature_files, window, self.coord_cols,
                        is_rfrk, feature_names, mask
                    ))
            
            # 设置输出路径
            output_path = self.output_dir / f"{property_name}_prediction.tif"
            uncertainty_path = self.output_dir / f"{property_name}_uncertainty.tif"
            
            # 更新输出配置
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                nodata=np.nan
            )
            
            # 执行预测
            with rasterio.open(output_path, 'w', **profile) as dst, \
                 rasterio.open(uncertainty_path, 'w', **profile) as uncertainty_dst:
                
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(self.process_raster_chunk, chunk) 
                             for chunk in chunks]
                    
                    for future in tqdm(as_completed(futures), 
                                     total=len(futures),
                                     desc=f"预测 {property_name}",
                                     ncols=100):
                        try:
                            window, result, uncertainty = future.result()
                            dst.write(result, 1, window=window)
                            uncertainty_dst.write(uncertainty, 1, window=window)
                        except Exception as e:
                            self.logger.error(f"处理数据块失败: {str(e)}")
            
            # 处理不确定性结果
            if self.enable_uncertainty_viz:
                self._process_uncertainty(uncertainty_path, property_name)
            
            return output_path, uncertainty_path
            
        except Exception as e:
            raise PredictionError(f"预测属性 {property_name} 时发生错误: {str(e)}")
    def _process_uncertainty(self, uncertainty_path, property_name):
        """
        处理不确定性结果
        
        Args:
            uncertainty_path: 不确定性结果文件路径
            property_name: 属性名称
        """
        try:
            # 计算统计信息
            statistics, valid_data = self._calculate_uncertainty_statistics(uncertainty_path)
            
            # 可视化分布
            self.visualizer.plot_uncertainty_distribution(
                property_name, valid_data, statistics,
                self.output_dir / f'{property_name}_uncertainty_distribution.png'
            )
            
            # 创建分位数栅格
            self._create_quantile_raster(uncertainty_path, property_name)
            
        except Exception as e:
            self.logger.error(f"处理不确定性结果时发生错误: {str(e)}")

    def _calculate_uncertainty_statistics(self, uncertainty_path):
        """
        计算不确定性统计信息
        
        Args:
            uncertainty_path: 不确定性文件路径
            
        Returns:
            tuple: (统计信息字典, 有效数据数组)
        """
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

    def _create_quantile_raster(self, uncertainty_path, property_name):
        """
        创建分位数栅格，并应用掩膜
        
        分位数值的含义：
        1: 极低不确定性 (0-20% 分位数)
        2: 低不确定性 (20-40% 分位数)
        3: 中等不确定性 (40-60% 分位数)
        4: 高不确定性 (60-80% 分位数)
        5: 极高不确定性 (80-100% 分位数)
        """
        try:
            # 读取不确定性数据和掩膜
            with rasterio.open(uncertainty_path) as src:
                uncertainty_data = src.read(1)
                profile = src.profile.copy()
                
            # 创建掩膜
            mask = None
            if self.shapefile_path:
                gdf = gpd.read_file(self.shapefile_path)
                with rasterio.open(uncertainty_path) as src:
                    geometries = [mapping(geom) for geom in gdf.geometry]
                    mask = geometry_mask(geometries, 
                                       out_shape=uncertainty_data.shape, 
                                       transform=src.transform, 
                                       invert=True)
            
            # 应用掩膜
            if mask is not None:
                uncertainty_data[~mask] = np.nan
                
            # 计算分位数
            valid_data = uncertainty_data[~np.isnan(uncertainty_data)]
            if len(valid_data) > 0:
                quantiles = [0.2, 0.4, 0.6, 0.8]
                quantile_values = np.nanquantile(valid_data, quantiles)
                
                # 创建分位数栅格
                quantile_raster = np.full_like(uncertainty_data, np.nan)
                valid_mask = ~np.isnan(uncertainty_data)
                quantile_raster[valid_mask] = np.digitize(uncertainty_data[valid_mask], quantile_values)
                
                # 记录分位数信息
                quantile_info = {
                    "1": f"极低不确定性 (0-{quantile_values[0]:.2f})",
                    "2": f"低不确定性 ({quantile_values[0]:.2f}-{quantile_values[1]:.2f})",
                    "3": f"中等不确定性 ({quantile_values[1]:.2f}-{quantile_values[2]:.2f})",
                    "4": f"高不确定性 ({quantile_values[2]:.2f}-{quantile_values[3]:.2f})",
                    "5": f"极高不确定性 (>{quantile_values[3]:.2f})"
                }
                
                # 保存分位数信息到文本文件
                info_path = self.output_dir / f'{property_name}_uncertainty_quantiles_info.txt'
                with open(info_path, 'w', encoding='utf-8') as f:
                    f.write(f"属性: {property_name}\n")
                    f.write("不确定性分位数说明：\n")
                    for level, desc in quantile_info.items():
                        f.write(f"等级 {level}: {desc}\n")
                    f.write(f"\n计算时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 保存栅格
                output_path = self.output_dir / f'{property_name}_uncertainty_quantiles.tif'
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(quantile_raster, 1)
                    
                self.logger.info(f"分位数栅格已保存至: {output_path}")
                self.logger.info(f"分位数说明已保存至: {info_path}")
                
            else:
                self.logger.warning("没有有效数据用于创建分位数栅格")
                
        except Exception as e:
            self.logger.error(f"创建分位数栅格失败: {str(e)}")

    @memory_monitor
    def run(self):
        """执行完整的预测流程"""
        self.logger.info("开始土壤属性预测流程")
        try:
            model_files = list(self.model_dir.glob('*_model.pkl'))
            self.logger.info(f"找到 {len(model_files)} 个模型文件")
            
            for model_file in tqdm(model_files, desc="总体进度", ncols=100):
                property_name = model_file.stem.replace('_model', '')
                self.logger.info(f"\n开始处理: {property_name}")
                
                try:
                    # 加载模型和数据
                    model, feature_names = self.load_model(model_file)
                    training_data = self.load_data(self.training_data_path, property_name)
                    
                    # 准备特征
                    available_features = [f for f in feature_names if f in training_data.columns]
                    if len(available_features) != len(feature_names):
                        self.logger.warning(
                            f"部分特征不可用: {set(feature_names) - set(available_features)}"
                        )
                    
                    # 准备数据
                    X = training_data[available_features + self.coord_cols]
                    y = training_data[property_name]
                    
                    # 比较和训练模型
                    updated_model, is_rfrk = self.compare_models_and_train_kriging(
                        X, y, model, available_features
                    )
                    
                    # 执行预测
                    output_path, uncertainty_path = self.predict_soil_property(
                        updated_model, available_features, property_name, is_rfrk
                    )
                    
                    self.logger.info(f"{property_name} 处理完成")
                    self.logger.info(f"预测结果: {output_path}")
                    self.logger.info(f"不确定性结果: {uncertainty_path}")
                    
                except Exception as e:
                    self.logger.error(f"{property_name} 处理失败: {str(e)}")
                    continue
                
            self.logger.info("所有属性处理完成")
            
        except Exception as e:
            self.logger.error(f"预测流程发生错误: {str(e)}")
            raise

# ================ 主程序入口 ================

if __name__ == "__main__":
    try:
        # 使用默认配置
        predictor = SoilPropertyPredictor()
        predictor.run()
    except Exception as e:
        logging.error(f"程序执行失败: {str(e)}")
