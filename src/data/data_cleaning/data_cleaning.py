import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging
from pathlib import Path
from typing import List, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

# 设置日志
log_file = Path('logs/data_cleaning.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, df: pd.DataFrame, lat_col: str, lon_col: str):
        self.df = df
        self.lat_col = lat_col
        self.lon_col = lon_col
        
    def identify_outliers(self, value_col: str, global_std_threshold: float = 5, local_std_threshold: float = 3, neighbors: int = 8) -> pd.DataFrame:
        """
        识别全局和局部异常值
        """
        logger.info(f"开始识别 {value_col} 列的异常值")
        
        if not pd.api.types.is_numeric_dtype(self.df[value_col]):
            logger.warning(f"{value_col} 不是数值类型,跳过异常值检测")
            return self.df
        
        values = self.df[value_col].values
        points = self.df[[self.lat_col, self.lon_col]].values
        
        # 全局异常值检测
        global_mean, global_std = np.nanmean(values), np.nanstd(values)
        global_threshold_upper = global_mean + global_std_threshold * global_std
        global_threshold_lower = global_mean - global_std_threshold * global_std
        
        # 构建KDTree
        tree = KDTree(points)
        
        status_col = f"{value_col}_Sta"
        self.df[status_col] = 'Normal'
        
        # 全局和局部异常值检测
        for i in range(len(points)):
            if pd.isna(values[i]):
                self.df.loc[self.df.index[i], status_col] = 'Missing'
                continue
            
            is_global_outlier = values[i] < global_threshold_lower or values[i] > global_threshold_upper
            
            # 局部异常检测
            distances, indices = tree.query(points[i], k=neighbors+1)
            local_values = values[indices[1:]]  # 排除自身
            local_mean, local_std = np.nanmean(local_values), np.nanstd(local_values)
            is_local_outlier = (values[i] < local_mean - local_std_threshold * local_std or 
                                values[i] > local_mean + local_std_threshold * local_std)
            
            if is_global_outlier and is_local_outlier:
                self.df.loc[self.df.index[i], status_col] = 'Global and Spatial Outlier'
            elif is_global_outlier:
                self.df.loc[self.df.index[i], status_col] = 'Global Outlier'
            elif is_local_outlier:
                self.df.loc[self.df.index[i], status_col] = 'Spatial Outlier'
        
        logger.info(f"{value_col} 列的异常值识别完成")
        return self.df
    
    def plot_filtered_data(self, value_col: str, output_path: Optional[str] = None):
        """
        可视化清洗前后的数据分布
        """
        logger.info(f"开始绘制 {value_col} 列的数据分布图")
        
        status_col = f"{value_col}_Sta"
        if status_col not in self.df.columns:
            logger.error(f"列 {status_col} 不存在，跳过绘图")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(28, 8))
        
        all_points = self.df[[self.lon_col, self.lat_col]].values
        
        # 所有点 - 使用单一颜色
        axes[0].scatter(all_points[:, 0], all_points[:, 1], color='blue', alpha=0.5, s=10)
        axes[0].set_title(f'Original Data ({value_col})')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        
        # 全局异常点
        normal = self.df[self.df[status_col] == 'Normal']
        global_outliers = self.df[self.df[status_col].isin(['Global Outlier', 'Global and Spatial Outlier'])]
        axes[1].scatter(normal[self.lon_col], normal[self.lat_col], color='blue', label='Normal', alpha=0.5, s=10)
        axes[1].scatter(global_outliers[self.lon_col], global_outliers[self.lat_col], color='red', label='Global Outliers', s=20)
        axes[1].set_title(f'Normal and Global Outliers ({value_col})')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].legend()
        
        # 全局和空间异常点
        spatial_outliers = self.df[self.df[status_col] == 'Spatial Outlier']
        global_and_spatial_outliers = self.df[self.df[status_col] == 'Global and Spatial Outlier']
        axes[2].scatter(normal[self.lon_col], normal[self.lat_col], color='blue', label='Normal', alpha=0.5, s=10)
        axes[2].scatter(global_outliers[self.lon_col], global_outliers[self.lat_col], color='orange', label='Global Outliers', s=20)
        axes[2].scatter(spatial_outliers[self.lon_col], spatial_outliers[self.lat_col], color='green', label='Spatial Outliers', s=20)
        axes[2].scatter(global_and_spatial_outliers[self.lon_col], global_and_spatial_outliers[self.lat_col], color='red', label='Global and Spatial Outliers', s=20)
        axes[2].set_title(f'All Outliers ({value_col})')
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        axes[2].legend()
        
        # 调整所有子图的刻度
        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"图像已保存至 {output_path}")
        
        plt.close(fig)
        
    def plot_summary(self, columns: List[str], output_path: Optional[str] = None):
        """
        绘制所有处理列的汇总图
        """
        logger.info("开始绘制汇总图")
        
        # 计算子图的行数和列数
        n_cols = len(columns)
        n_rows = int(np.ceil(np.sqrt(n_cols)))
        n_cols = int(np.ceil(n_cols / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten()  # 将多维数组展平，方便索引
        
        for i, col in enumerate(columns):
            status_col = f"{col}_Sta"
            if status_col not in self.df.columns:
                logger.warning(f"列 {status_col} 不存在，跳过绘图")
                continue
            
            normal = self.df[self.df[status_col] == 'Normal']
            global_outliers = self.df[self.df[status_col] == 'Global Outlier']
            spatial_outliers = self.df[self.df[status_col] == 'Spatial Outlier']
            global_and_spatial_outliers = self.df[self.df[status_col] == 'Global and Spatial Outlier']
            
            axes[i].scatter(normal[self.lon_col], normal[self.lat_col], color='blue', label='Normal', alpha=0.5, s=10)
            axes[i].scatter(global_outliers[self.lon_col], global_outliers[self.lat_col], color='orange', label='Global Outliers', s=20)
            axes[i].scatter(spatial_outliers[self.lon_col], spatial_outliers[self.lat_col], color='green', label='Spatial Outliers', s=20)
            axes[i].scatter(global_and_spatial_outliers[self.lon_col], global_and_spatial_outliers[self.lat_col], color='red', label='Global and Spatial Outliers', s=20)
            
            axes[i].set_title(f'Outliers for {col}', fontsize=10)
            axes[i].set_xlabel('Longitude', fontsize=8)
            axes[i].set_ylabel('Latitude', fontsize=8)
            
            # 设置x轴和y轴的刻度
            x_min, x_max = axes[i].get_xlim()
            y_min, y_max = axes[i].get_ylim()
            x_ticks = np.linspace(x_min, x_max, 5)
            y_ticks = np.linspace(y_min, y_max, 5)
            
            axes[i].xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
            axes[i].yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
            axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            
            # 设置x轴标签的旋转和对齐
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            
            axes[i].tick_params(axis='both', which='major', labelsize=6)
            
            axes[i].legend(fontsize='xx-small', loc='lower left')
        
        # 移除多余的子图
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.5, wspace=0.4)  # 增加子图之间的间距
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"汇总图已保存至 {output_path}")
        
        plt.close(fig)
        
    def clean_data(self, columns: List[str], output_folder: str, global_std_threshold: float = 5, local_std_threshold: float = 3, neighbors: int = 8):
        """
        清洗指定列的数据并导出结果
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for col in columns:
            try:
                self.identify_outliers(col, global_std_threshold, local_std_threshold, neighbors)
            except Exception as e:
                logger.error(f"处理 {col} 列时发生错误: {str(e)}")
        
        for col in columns:
            try:
                self.plot_filtered_data(col, str(output_path / f"{col}_distribution.png"))
            except Exception as e:
                logger.error(f"绘制 {col} 列的图像时发生错误: {str(e)}")
        
        try:
            self.plot_summary(columns, str(output_path / "summary_plot.png"))
        except Exception as e:
            logger.error(f"绘制汇总图时发生错误: {str(e)}")
        
        # 导出清洗后的数据
        cleaned_data_path = output_path / "cleaned_data.csv"
        self.df.to_csv(cleaned_data_path, index=False)
        logger.info(f"清洗后的数据已保存至 {cleaned_data_path}")

def main(df_path: str, lon_col: str, lat_col: str, columns: List[str], output_folder: str, global_std_threshold: float = 5, local_std_threshold: float = 3, neighbors: int = 8):
    """
    主函数
    """
    logger.info("开始数据清洗过程")
    
    try:
        # 读取数据
        df = pd.read_csv(df_path)
        logger.info(f"已加载数据，共 {len(df)} 行")
    except Exception as e:
        logger.error(f"读取数据时发生错误: {str(e)}")
        return
    
    # 创建DataCleaner实例
    cleaner = DataCleaner(df, lat_col, lon_col)
    
    # 清洗数据
    cleaner.clean_data(columns, output_folder, global_std_threshold, local_std_threshold, neighbors)
    
    logger.info("数据清洗过程完成")

if __name__ == "__main__":
    # 示例用法
    df_path = r"C:\Users\Runker\Desktop\GL\data\result.csv"
    lon_col, lat_col = 'dwjd', 'dwwd'
    columns_to_clean = ['ph', 'ylzjhl', 'yjz', 'qdan', 'qlin', 'qjia', 'qxi', 'yxlin', 'sxjia','hxjia']
    output_folder = r"C:\Users\Runker\Desktop\GL\data_clean_result"
    global_std_threshold = 5
    local_std_threshold = 3
    neighbors = 8
    main(df_path, lon_col, lat_col, columns_to_clean, output_folder, global_std_threshold, local_std_threshold, neighbors)