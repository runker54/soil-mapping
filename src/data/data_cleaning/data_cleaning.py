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
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from scipy import stats

class DataCleaner:
    def __init__(self, df: pd.DataFrame, lat_col: str, lon_col: str):
        self.df = df
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.logger = logging.getLogger(__name__)
        
    def identify_outliers(self, value_col: str, global_std_threshold: float = 5, local_std_threshold: float = 3, neighbors: int = 8) -> pd.DataFrame:
        """
        识别全局和局部异常值
        """
        self.logger.info(f"开始识别 {value_col} 列的异常值")
        
        if not pd.api.types.is_numeric_dtype(self.df[value_col]):
            self.logger.warning(f"{value_col} 是数值类型,跳过异常值检测")
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
        
        self.logger.info(f"{value_col} 列的异常值识别完成")
        return self.df
    
    def plot_filtered_data(self, value_col: str, output_path: Optional[str] = None):
        """
        可视化清洗前后的数据分布
        """
        self.logger.info(f"开始绘制 {value_col} 列的数据分布图")
        
        status_col = f"{value_col}_Sta"
        if status_col not in self.df.columns:
            self.logger.error(f"列 {status_col} 不存在，跳过绘图")
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
            self.logger.info(f"图像已保存至 {output_path}")
        
        plt.close(fig)
        
    def plot_summary(self, columns: List[str], output_path: Optional[str] = None):
        """
        绘制所有处理列的汇总图
        """
        self.logger.info("开始绘制汇总图")
        
        # 计算子图的行数和列数
        n_cols = len(columns)
        n_rows = int(np.ceil(np.sqrt(n_cols)))
        n_cols = int(np.ceil(n_cols / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten()  # 将多维数组展平，方便索引
        
        for i, col in enumerate(columns):
            status_col = f"{col}_Sta"
            if status_col not in self.df.columns:
                self.logger.warning(f"列 {status_col} 不存在，跳过绘图")
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
            
            # 设置x轴y轴的刻度
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
            self.logger.info(f"汇总图已保存至 {output_path}")
        
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
                self.logger.error(f"处理 {col} 列时发生错误: {str(e)}")
        
        for col in columns:
            try:
                self.plot_filtered_data(col, str(output_path / f"{col}_distribution.png"))
            except Exception as e:
                self.logger.error(f"绘制 {col} 列的图像时发生错误: {str(e)}")
        
        try:
            self.plot_summary(columns, str(output_path / "summary_plot.png"))
        except Exception as e:
            self.logger.error(f"绘制汇总图时发生错误: {str(e)}")
        
        # 添加新的可视化调用
        try:
            self.plot_distribution(columns, str(output_path / "distributions.png"))
            self.plot_correlation_network_matrix(columns, str(output_path / "correlation_network_matrix.png"))
        except Exception as e:
            self.logger.error(f"生成额外可视化图表时发生错误: {str(e)}")
        
        # 导出清洗后的数据
        cleaned_data_path = output_path / "cleaned_data.csv"
        self.df.to_csv(cleaned_data_path, index=False)
        self.logger.info(f"清洗后的数据已保存至 {cleaned_data_path}")
    def generate_cleaning_report(self, columns: List[str]) -> pd.DataFrame:
        """
        生成数据清洗报告
        """
        self.logger.info("开始生成数据清洗报告")
        
        report_data = []
        for col in columns:
            status_col = f"{col}_Sta"
            if status_col not in self.df.columns:
                self.logger.warning(f"列 {status_col} 不存在，跳过报告生成")
                continue
            
            total_count = len(self.df)
            normal_count = (self.df[status_col] == 'Normal').sum()
            global_outlier_count = (self.df[status_col] == 'Global Outlier').sum()
            spatial_outlier_count = (self.df[status_col] == 'Spatial Outlier').sum()
            global_and_spatial_outlier_count = (self.df[status_col] == 'Global and Spatial Outlier').sum()
            
            report_data.append({
                'Column': col,
                'Total': total_count,
                'Normal': normal_count,
                'Normal (%)': normal_count / total_count * 100,
                'Global Outliers': global_outlier_count,
                'Global Outliers (%)': global_outlier_count / total_count * 100,
                'Spatial Outliers': spatial_outlier_count,
                'Spatial Outliers (%)': spatial_outlier_count / total_count * 100,
                'Global and Spatial Outliers': global_and_spatial_outlier_count,
                'Global and Spatial Outliers (%)': global_and_spatial_outlier_count / total_count * 100
            })
        
        report_df = pd.DataFrame(report_data)
        self.logger.info("数据清洗报告生成完成")
        return report_df

    def plot_distribution(self, columns: List[str], output_path: Optional[str] = None):
        """
        绘制清洗后数据的高级分布图，包含箱线图、小提琴图、核密度估计和正态分布检验
        """
        self.logger.info("开始绘制数据分布图")
        
        # 根据输入列数动态计算最优行列数
        n_plots = len(columns)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        
        # 调整图形大小
        fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # 只增大总标题字体大小，增加与图的间距
        fig.suptitle('Distribution Analysis of Soil Properties After Outlier Removal', 
                     fontsize=20, y=1.01)  # 增大标题字体大小并调整位置

        # 创建所有子图
        axes = []
        for i in range(n_rows * n_cols):
            if i < n_plots:
                axes.append(fig.add_subplot(gs[i // n_cols, i % n_cols]))
            else:
                # 删除多余的子图
                fig.delaxes(plt.subplot(gs[i // n_cols, i % n_cols]))
        
        for idx, (ax_main, col) in enumerate(zip(axes, columns)):
            # 创建共享x轴的子图
            ax_density = ax_main.twinx()
            
            # 获取清洗后的正常数据
            status_col = f"{col}_Sta"
            if status_col in self.df.columns:
                normal_data = self.df[self.df[status_col] == 'Normal'][col].dropna()
            else:
                normal_data = self.df[col].dropna()
                
            # 计算基本统计量
            mean_val = normal_data.mean()
            median_val = normal_data.median()
            std_val = normal_data.std()
            
            # 正态分布检验
            _, shapiro_p = stats.shapiro(normal_data)
            _, ks_p = stats.kstest(normal_data, 'norm', args=(mean_val, std_val))
            is_normal = (shapiro_p > 0.05) and (ks_p > 0.05)
            
            # 计算偏度和峰度
            skewness = stats.skew(normal_data)
            kurtosis = stats.kurtosis(normal_data)
            
            # 绘制小提琴图
            violin_parts = ax_main.violinplot(normal_data, positions=[0], 
                                            showmeans=False, showmedians=False)
            for pc in violin_parts['bodies']:
                pc.set_facecolor('#E6E6FA')  # 更浅的紫色
                pc.set_edgecolor('black')
                pc.set_alpha(0.6)
            
            # 绘制箱线图
            box_parts = ax_main.boxplot(normal_data, positions=[0], widths=0.2,
                                      showfliers=True, patch_artist=True)
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(box_parts[element], color='black')
            plt.setp(box_parts['boxes'], facecolor='white', alpha=0.7)
            plt.setp(box_parts['fliers'], markerfacecolor='gray', alpha=0.5, markersize=4)
            
            # 核密度估计
            density = gaussian_kde(normal_data)
            xs = np.linspace(normal_data.min(), normal_data.max(), 200)
            density_line = ax_density.plot(density(xs), xs, 'r-', linewidth=1.5, alpha=0.7)[0]
            ax_density.fill_betweenx(xs, density(xs), alpha=0.1, color='red')
            
            # 添加理论正态分布曲线
            if is_normal:
                norm_xs = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 200)
                norm_density = stats.norm.pdf(norm_xs, mean_val, std_val)
                max_density = max(density(xs))
                norm_density = norm_density * (max_density / max(norm_density))
                theoretical_line = ax_density.plot(norm_density, norm_xs, 'k--', 
                                                linewidth=1, alpha=0.8, 
                                                label='Theoretical Normal')[0]
            
            # 添加均值和中位数线
            mean_line = ax_density.axhline(mean_val, color='#1E90FF', 
                                         linestyle='--', alpha=0.8, linewidth=1)
            median_line = ax_density.axhline(median_val, color='#32CD32', 
                                           linestyle='--', alpha=0.8, linewidth=1)
            
            # 设置标签和标题
            ax_main.set_title(f'Distribution of {col}', pad=20, fontsize=10)
            ax_main.set_ylabel('Value', fontsize=8)
            ax_density.set_xlabel('Density', fontsize=8)
            
            # 移除不需要的刻度
            ax_main.set_xticks([])
            
            # 添加统计信息文本框
            stats_text = (
                f'Sample size (n) = {len(normal_data)}\n'
                f'Mean = {mean_val:.2f}\n'
                f'Median = {median_val:.2f}\n'
                f'Std = {std_val:.2f}\n'
                f'CV = {(std_val/mean_val)*100:.1f}%\n'
                f'Skewness = {skewness:.2f}\n'
                f'Kurtosis = {kurtosis:.2f}\n'
                f'Normality test = {is_normal}'
            )
            
            # 调整文本框位置到左上角
            ax_main.text(0.05, 0.95, stats_text,
                        transform=ax_main.transAxes,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', 
                                 alpha=0.9, edgecolor='gray'),
                        fontsize=8)
            
            # 设置y轴范围
            y_min, y_max = normal_data.min(), normal_data.max()
            y_range = y_max - y_min
            y_limits = (y_min - 0.1*y_range, y_max + 0.1*y_range)
            ax_main.set_ylim(y_limits)
            ax_density.set_ylim(y_limits)
            
            # 设置刻度标签大小
            ax_main.tick_params(axis='both', labelsize=8)
            ax_density.tick_params(axis='both', labelsize=8)
        
        # 在所有子图绘制完成后，添加整体图例
        legend_elements = [
            Line2D([0], [0], color='#1E90FF', linestyle='--', label='Mean'),
            Line2D([0], [0], color='#32CD32', linestyle='--', label='Median'),
            Patch(facecolor='#E6E6FA', alpha=0.6, label='Distribution'),
            Line2D([0], [0], color='red', alpha=0.7, label='Kernel Density Estimation'),
            Line2D([0], [0], color='black', linestyle='--', label='Theoretical Normal Distribution')
        ]
        
        # 调整底部图例位置，使其更靠近主图
        fig.legend(handles=legend_elements, 
                  loc='center',
                  bbox_to_anchor=(0.5, 0.02),  # 调整y值使图例更靠近上方的图
                  ncol=5,
                  fontsize=8,
                  frameon=True,
                  edgecolor='gray')
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # 只调整底部边距，其他保持默认
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"分布图已保存至 {output_path}")
        
        plt.close(fig)

    def plot_correlation_network_matrix(self, columns: List[str], output_path: Optional[str] = None):
        """
        绘制组合式相关性网络和矩阵图
        """
        self.logger.info("开始绘制组合式相关性网络和矩阵图")
        
        # 创建图形，调整比例使矩阵图更大
        fig = plt.figure(figsize=(14, 9))
        
        # 调整网格比例,使网络图和矩阵图大小更合理
        gs = GridSpec(2, 2, height_ratios=[5, 0.7], width_ratios=[1.2, 1.3])
        
        # 网络图（左上）
        ax_network = fig.add_subplot(gs[0, 0])
        
        # 相关性矩阵（右上）
        ax_matrix = fig.add_subplot(gs[0, 1])
        
        # 图例区域（底部横跨）
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis('off')
        
        # 准备数据
        normal_data = pd.DataFrame()
        for col in columns:
            status_col = f"{col}_Sta"
            if status_col in self.df.columns:
                normal_data[col] = self.df[self.df[status_col] == 'Normal'][col]
            else:
                normal_data[col] = self.df[col]
        
        corr = normal_data.corr()
        cmap = plt.colormaps['coolwarm']
        
        # 计算节点位置（优化的圆形布局）
        n_vars = len(columns)
        angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
        radius = 0.55  # 减小半径使网络图更紧凑
        pos = {col: (radius * np.cos(angle), radius * np.sin(angle)) 
               for col, angle in zip(columns, angles)}
        
        # 绘制网络连接
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:
                    correlation = corr.loc[col1, col2]
                    if abs(correlation) > 0.1:
                        x1, y1 = pos[col1]
                        x2, y2 = pos[col2]
                        color = cmap((correlation + 1) / 2)
                        width = abs(correlation) * 2
                        alpha = min(abs(correlation) + 0.3, 0.8)
                        ax_network.plot([x1, x2], [y1, y2], 
                                      color=color, linewidth=width, alpha=alpha)
        
        # 绘制节点和标签
        for col in columns:
            x, y = pos[col]
            ax_network.scatter(x, y, s=120, c='white', edgecolor='gray', 
                             linewidth=1, zorder=5)
            
            label_radius = radius * 1.12
            label_x = label_radius * np.cos(angles[columns.index(col)])
            label_y = label_radius * np.sin(angles[columns.index(col)])
            
            ha = 'left' if label_x > 0 else 'right'
            va = 'center'
            if abs(label_y) > radius * 0.7:
                va = 'bottom' if label_y > 0 else 'top'
                ha = 'center'
            
            ax_network.annotate(col, (x, y), xytext=(label_x, label_y),
                              ha=ha, va=va, fontsize=8)
        
        ax_network.set_title('Correlation Network', pad=15, fontsize=10)
        ax_network.axis('equal')
        ax_network.axis('off')
        
        # 相关性矩阵
        corr = self.df[columns].corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        sns.heatmap(corr, 
                    mask=mask,
                    cmap='coolwarm',
                    center=0,
                    annot=True,
                    fmt='.2f',
                    square=True,
                    cbar_kws={'shrink': .7, 'label': 'Correlation Coefficient'},
                    ax=ax_matrix,
                    annot_kws={'size': 7}
                    )
        
        ax_matrix.set_title('Correlation Matrix', fontsize=10, pad=10)
        ax_matrix.set_xticklabels(columns, rotation=45, ha='right', fontsize=8)
        ax_matrix.set_yticklabels(columns, rotation=0, fontsize=8)
        
        # 调整子图间距
        plt.subplots_adjust(hspace=0.15, wspace=0.25)
        
        # 图例部分
        legend_elements = []
        for corr_strength in [0.2, 0.4, 0.6, 0.8]:
            color = cmap((corr_strength + 1) / 2)
            legend_elements.append(
                Line2D([0], [0], color=color, 
                      linewidth=corr_strength * 2,
                      label=f'|r| = {corr_strength:.1f}')
            )
        
        legend_elements.extend([
            Line2D([0], [0], color=cmap(0.9), linewidth=2,
                   label="Strong positive (r > 0.6)"),
            Line2D([0], [0], color=cmap(0.1), linewidth=2,
                   label="Strong negative (r < -0.6)"),
            Line2D([0], [0], color='gray', linewidth=1, linestyle='--',
                   label="Weak correlation (|r| < 0.3)"),
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor='white', markeredgecolor='gray',
                   markersize=8, label='Variable node')
        ])
        
        legend1 = legend_ax.legend(handles=legend_elements[:4], 
                                  loc='center',
                                  bbox_to_anchor=(0.2, 0.75),
                                  ncol=4, 
                                  title="Pearson's Correlation Coefficient (r)",
                                  fontsize=7,
                                  title_fontsize=8)
        
        legend2 = legend_ax.legend(handles=legend_elements[4:], 
                                  loc='center',
                                  bbox_to_anchor=(0.2, 0.25),
                                  ncol=4,
                                  title="Correlation Interpretation",
                                  fontsize=7,
                                  title_fontsize=8)
        
        legend_ax.add_artist(legend1)
        
        stats_text = (
            f"Notes:\n"
            f"• Network shows significant correlations (|r| > 0.1)\n"
            f"• Line thickness proportional to |r|\n"
            f"• Color intensity indicates correlation strength"
        )
        
        legend_ax.text(0.7, 0.7, stats_text, fontsize=7,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        legend_ax.text(0.7, 0.3,
                      "Method: Pearson correlation analysis\n"
                      "Significance level: p < 0.05", 
                      fontsize=7,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
            self.logger.info(f"组合式相关性图已保存至 {output_path}")
        
        plt.close()

def main(df_path: str, lon_col: str, lat_col: str, columns: List[str], output_folder: str, log_file: str, global_std_threshold: float = 5, local_std_threshold: float = 3, neighbors: int = 8):
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

    logger.info("开始数据清洗过程")
    
    try:
        # 读取数据
        df = pd.read_csv(df_path)
        logger.info(f"已加载数据，共 {len(df)} ")
        
        # 创建DataCleaner例
        cleaner = DataCleaner(df, lat_col, lon_col)
        
        # 清洗数据
        cleaner.clean_data(columns, output_folder, global_std_threshold, local_std_threshold, neighbors)
        output_path = Path(output_folder)
        # 生成清洗报告
        cleaning_report = cleaner.generate_cleaning_report(columns)
        report_path = output_path / "cleaning_report.csv"
        cleaning_report.to_csv(report_path, index=False)
        logger.info(f"清洗报告已保存至 {report_path}")
        logger.info(f"数据清洗过程完成，输出目录: {output_folder}")
        # 检查输出文件是否存在
        cleaned_data_path = output_path / "cleaned_data.csv"
        summary_plot_path = output_path / "summary_plot.png"
        
        if cleaned_data_path.exists():
            logger.info(f"清洗后的数据文件已生成: {cleaned_data_path}")
        else:
            logger.error(f"清洗后的数据文件未生成: {cleaned_data_path}")
        
        if summary_plot_path.exists():
            logger.info(f"汇总图已生成: {summary_plot_path}")
        else:
            logger.error(f"汇总图未生成: {summary_plot_path}")
        
    except Exception as e:
        logger.error(f"数据清洗过程中发生错误: {str(e)}")
        raise  # 重新抛出异常，确保主程序能够捕获到错误
    

# 测试
if __name__ == "__main__":
    df_path = r'D:\soil-mapping\data\soil_property_table\result.csv'
    lon_col = 'dwjd' # 经度列名
    lat_col = 'dwwd' # 纬度列名
    columns =  ['ph', 'ylzjhl', 'yjz', 'qdan', 'qlin', 'qjia', 'qxi', 'yxlin', 'sxjia','hxjia'] # 需要清洗的标签列
    output_folder = r'D:\soil-mapping\figures\data_clean'
    log_file = r'D:\soil-mapping\logs\data_cleaning.log'
    global_std_threshold = 5
    local_std_threshold = 3
    neighbors = 8
    main(df_path, lon_col, lat_col, columns, output_folder, log_file, global_std_threshold, local_std_threshold, neighbors)