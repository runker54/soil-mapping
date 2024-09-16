import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import pickle
from pathlib import Path
from tqdm import tqdm
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from scipy import stats
import math
from concurrent.futures import ProcessPoolExecutor
from rasterio.plot import show_hist

# 设置日志
log_file = Path('logs/generate_soil_property_report.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_evaluation_results(eval_dir):
    """加载评估结果"""
    logger.info(f"正在加载评估结果: {eval_dir}")
    file_path = Path(eval_dir) / "reports" / "model_summary.xlsx"
    logger.info(f"尝试加载文件: {file_path}")
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"找不到文件: {file_path}")
    eval_summary = pd.read_excel(file_path, sheet_name="Model Performance", index_col=0)
    logger.info(f"加载的评估结果形状: {eval_summary.shape}")
    logger.info(f"评估结果列名: {eval_summary.columns}")
    return eval_summary

def load_prediction_rasters(prediction_dir):
    """加载预测结果栅格"""
    logger.info(f"正在加载预测结果栅格: {prediction_dir}")
    prediction_files = list(Path(prediction_dir).glob('*.tif'))
    predictions = {}
    
    for file in tqdm(prediction_files, desc="加载预测栅格"):
        with rasterio.open(file) as src:
            property_name = file.stem.replace('_prediction', '')
            predictions[property_name] = src.read(1, masked=True)
            predictions['transform'] = src.transform
            predictions['crs'] = src.crs
    
    return predictions

def process_property(args):
    property_name, raster_path, plot_type = args
    try:
        with rasterio.open(raster_path) as src:
            if plot_type == 'maps':
                return property_name, src.read(1, masked=True)
            else:  # histograms
                data = src.read(1, masked=True)
                valid_data = data.compressed()  # 移除掩码值
                if len(valid_data) == 0:
                    logger.warning(f"{property_name} 没有有效数据")
                    return property_name, None
                
                hist, bin_edges = np.histogram(valid_data, bins=50, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_widths = np.diff(bin_edges)
                
                logger.info(f"{property_name} 直方图数据: 最小值={np.min(valid_data):.2f}, "
                            f"最大值={np.max(valid_data):.2f}, 平均值={np.mean(valid_data):.2f}")
                
                return property_name, (bin_centers, hist, bin_widths)
    except Exception as e:
        logger.error(f"处理属性 {property_name} 时出错: {str(e)}")
        return property_name, None

def create_combined_plots(prediction_dir, output_dir, plot_type='maps'):
    """创建合并的预测图或直方图"""
    logger.info(f"创建合并的{'预测图' if plot_type == 'maps' else '直方图'}")
    prediction_files = list(Path(prediction_dir).glob('*.tif'))
    n_properties = len(prediction_files)
    
    # 动态计算行数和列数
    aspect_ratio = 16/9
    n_cols = math.ceil(math.sqrt(n_properties * aspect_ratio))
    n_rows = math.ceil(n_properties / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    
    # 使用多进程处理数据
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_property, (file.stem, str(file), plot_type)) 
                   for file in prediction_files]
        results = []
        for future in tqdm(futures, total=len(prediction_files), desc=f"处理{plot_type}"):
            results.append(future.result())
    
    for i, (property_name, data) in enumerate(results):
        if data is None:
            logger.warning(f"跳过 {property_name} 因为没有有效数据")
            continue
        row = i // n_cols
        col = i % n_cols
        
        try:
            if plot_type == 'maps':
                im = axs[row, col].imshow(data, cmap='viridis')
                plt.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
            else:  # histograms
                bin_centers, hist, bin_widths = data
                axs[row, col].bar(bin_centers, hist, width=bin_widths, alpha=0.7, align='center')
                axs[row, col].set_xlabel('Predicted Value')
                axs[row, col].set_ylabel('Density')
            
            axs[row, col].set_title(property_name)
            logger.info(f"成功绘制 {property_name} 的{'预测图' if plot_type == 'maps' else '直方图'}")
        except Exception as e:
            logger.error(f"绘制 {property_name} 的图表时出错: {str(e)}")
    
    # 隐藏多余的子图
    for i in range(n_properties, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    filename = 'combined_prediction_maps.png' if plot_type == 'maps' else 'combined_histograms.png'
    try:
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        logger.info(f"合并的{'预测图' if plot_type == 'maps' else '直方图'}已保存")
    except Exception as e:
        logger.error(f"保存图像时出错: {str(e)}")
    finally:
        plt.close()

def create_correlation_heatmap(predictions, output_dir):
    """创建土壤属性相关性热图"""
    logger.info("创建土壤属性相关性热图")
    pred_df = pd.DataFrame({k: v.flatten() for k, v in predictions.items() if k not in ['transform', 'crs']})
    logger.info(f"相关性数据形状: {pred_df.shape}")
    logger.info(f"相关性数据列: {pred_df.columns}")
    
    # 移除极端值
    for col in pred_df.columns:
        pred_df = pred_df[pred_df[col] < pred_df[col].quantile(0.99)]
    
    corr = pred_df.corr()
    logger.info(f"相关性矩阵形状: {corr.shape}")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Soil Property Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'soil_property_correlation_heatmap.png')
    plt.close()
    logger.info("相关性热图已保存")

def create_model_performance_comparison(eval_results, output_dir):
    """创建优化后的模型性能指标比较图"""
    logger.info("创建模型性能指标比较图")
    logger.info(f"评估结果列名: {eval_results.columns}")
    
    models = ['RF', 'RFRK']
    metrics = ['R2', 'MAE', 'MSE', 'RMSE']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Model Performance Comparison', fontsize=20)
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        x = np.arange(len(eval_results.index))
        width = 0.35
        
        for j, model in enumerate(models):
            column = f'{model}_{metric}'
            values = eval_results[column]
            bars = ax.bar(x + j*width, values, width, label=model, alpha=0.7)
            
            # 添加数值标注
            for bar in bars:
                height = bar.get_height()
                label = f'{height:.2f}' if 0.01 <= height <= 100 else f'{height:.2e}'
                ax.annotate(label,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=8)
        
        ax.set_xlabel('Soil Properties', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(eval_results.index, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        
        # 设置刻度和格式化标签
        if metric == 'R2':
            ax.set_ylim(0, min(1.1, eval_results[[f'{model}_{metric}' for model in models]].max().max() * 1.1))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        else:
            ax.set_yscale('symlog', linthresh=1e-1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2e}' if y < 0.01 or y > 100 else f'{y:.2f}'))
            
            # 动态调整 y 轴上限
            max_value = eval_results[[f'{model}_{metric}' for model in models]].max().max()
            ax.set_ylim(top=max_value * 1.2)
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plot(model_dir, output_dir):
    """创建特征重要性图"""
    logger.info("创建特征重要性图")
    model_files = list(Path(model_dir).glob('*_model.pkl'))
    importance_data = {}
    
    for model_file in model_files:
        property_name = model_file.stem.replace('_model', '')
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data:
            model = model_data['model']
            feature_names = model_data['feature_names']
            importance = model.feature_importances_
            importance_data[property_name] = dict(zip(feature_names, importance))
    
    if not importance_data:
        logger.warning("没有找到有效的特征重要性数据")
        return
    
    importance_df = pd.DataFrame(importance_data).fillna(0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(importance_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Feature Importance Comparison')
    plt.ylabel('Features')
    plt.xlabel('Soil Properties')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png')
    plt.close()
    logger.info("特征重要性图已保存")

def generate_pdf_report(eval_results, predictions, output_dir, model_dir):
    """生成PDF报告"""
    logger.info("生成PDF报告")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "soil_property_report.pdf"
    
    doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    story = []
    
    # 添加标题
    story.append(Paragraph("Soil Property Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))
    
    # 添加执行摘要
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_text = """
    This report presents the results of a soil property prediction study using machine learning techniques. 
    The study aimed to predict various soil properties across the study area using environmental covariates. 
    Random Forest models were employed for each soil property, and their performance was evaluated using metrics 
    such as R-squared (R2), Root Mean Square Error (RMSE), and Mean Absolute Error (MAE). The report includes 
    visualizations of predicted soil properties, model performance comparisons, feature importance analysis, 
    and distribution of predicted values.
    """
    story.append(Paragraph(summary_text, styles['Justify']))
    story.append(Spacer(1, 12))
    
    # 添加评估结果表格
    story.append(Paragraph("Model Evaluation Results", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # 将评估结果四舍五入到两位小数
    eval_results_rounded = eval_results.round(2)
    
    # 转置数据框
    eval_results_transposed = eval_results_rounded.transpose()
    
    # 重置索引，将原来的列名变成新的列
    eval_results_transposed = eval_results_transposed.reset_index()
    
    # 重命名列
    eval_results_transposed.columns = ['Metric'] + list(eval_results.index)
    
    # 创建表格数据
    data = [eval_results_transposed.columns.tolist()] + eval_results_transposed.values.tolist()
    
    # 创建表格
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    
    story.append(PageBreak())
    
    # 修改图像添加部分
    def add_image_to_story(image_path, width=7*inch):
        if image_path.exists():
            img = Image(str(image_path))
            img.drawHeight = img.drawHeight * (width / img.drawWidth)
            img.drawWidth = width
            story.append(img)
    
    # 添加合并的预测图
    story.append(Paragraph("Combined Prediction Maps", styles['Heading2']))
    story.append(Spacer(1, 12))
    add_image_to_story(output_dir / 'combined_prediction_maps.png', width=6.5*inch)
    
    story.append(PageBreak())
    
    # 添加合并的直方图
    story.append(Paragraph("Combined Prediction Distributions", styles['Heading2']))
    story.append(Spacer(1, 12))
    add_image_to_story(output_dir / 'combined_histograms.png', width=6.5*inch)
    
    story.append(PageBreak())
    
    # 添加相关性热图
    story.append(Paragraph("Soil Property Correlation", styles['Heading2']))
    story.append(Spacer(1, 12))
    add_image_to_story(output_dir / 'soil_property_correlation_heatmap.png', width=6.5*inch)
    
    story.append(PageBreak())
    
    # 添加模型性能比较图
    story.append(Paragraph("Model Performance Comparison", styles['Heading2']))
    story.append(Spacer(1, 12))
    add_image_to_story(output_dir / 'model_performance_comparison.png', width=6.5*inch)
    
    story.append(PageBreak())
    
    # 添加特征重要性图
    story.append(Paragraph("Feature Importance", styles['Heading2']))
    story.append(Spacer(1, 12))
    add_image_to_story(output_dir / 'feature_importance_comparison.png', width=6.5*inch)
    
    # 生成PDF
    doc.build(story)
    logger.info(f"PDF报告已生成: {pdf_path}")

def generate_soil_property_report(eval_dir, prediction_dir, output_dir, model_dir):
    """生成土壤属性报告"""
    # 加载评估结果
    eval_results = load_evaluation_results(eval_dir)
    
    # 加载预测结果
    predictions = load_prediction_rasters(prediction_dir)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建合并的预测图
    create_combined_plots(prediction_dir, output_dir, plot_type='maps')
    
    # 创建合并的直方图
    create_combined_plots(prediction_dir, output_dir, plot_type='histograms')
    
    # 创建相关性热图
    create_correlation_heatmap(predictions, output_dir)
    
    # 创建模型性能比较图
    create_model_performance_comparison(eval_results, output_dir)
    
    # 创建特征重要性图
    create_feature_importance_plot(model_dir, output_dir)
    
    # 生成PDF报告
    generate_pdf_report(eval_results, predictions, output_dir, model_dir)
    
    logger.info("报告生成完成")

if __name__ == "__main__":
    eval_dir = Path(r"C:\Users\Runker\Desktop\GL\rfrk")
    prediction_dir = Path(r"C:\Users\Runker\Desktop\GL\gl_tif_properte_predict")
    output_dir = Path(r"C:\Users\Runker\Desktop\GL\report_output")
    model_dir = Path(r"C:\Users\Runker\Desktop\GL\rfrk\models")
    
    generate_soil_property_report(eval_dir, prediction_dir, output_dir, model_dir)