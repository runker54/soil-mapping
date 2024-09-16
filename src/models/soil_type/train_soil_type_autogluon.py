import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor, TabularDataset
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
log_file = Path('logs/train_soil_type_autogluon.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path, label_col, feature_cols, categorical_features):
    """加载数据,选择特征,处理缺失值,并转换类型"""
    logger.info(f"正在从 {file_path} 加载数据")
    data = pd.read_csv(file_path)
    
    # 选择指定的特征列和标签列
    data = data[feature_cols + categorical_features + [label_col]]
    
    # 处理标签列的缺失值
    if data[label_col].isnull().any():
        logger.warning(f"标签列 '{label_col}' 存在缺失值,将删除这些行")
        data = data.dropna(subset=[label_col])
    
    # 重复只有一个样本的类别
    value_counts = data[label_col].value_counts()
    single_sample_classes = value_counts[value_counts == 1].index
    
    if len(single_sample_classes) > 0:
        single_samples = data[data[label_col].isin(single_sample_classes)]
        data = pd.concat([data, single_samples], ignore_index=True)
        logger.info(f"重复了 {len(single_sample_classes)} 个只有一个样本的类别")
    
    # 处理类别特征
    for col in categorical_features:
        logger.info(f"将类别特征 {col} 转换为 'category' 类型")
        data[col] = data[col].astype('category')
    
    # 处理数值特征的缺失值
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    # 处理类别特征的缺失值
    for col in categorical_features:
        data[col] = data[col].fillna(data[col].mode()[0])
    
    # 将DataFrame转换为TabularDataset
    data = TabularDataset(data)
    
    logger.info(f"数据加载和预处理完成。形状：{data.shape}")
    return data

def train_model(df, label_col, feature_cols, categorical_features, save_dir, time_limit, eval_metric, problem_type, hyperparameters):
    """训练模型"""
    logger.info(f"正在训练 {label_col} 的模型")
    
    # 计算类别数量
    n_classes = df[label_col].nunique()
    
    # 计算测试集的最小大小
    min_test_size = max(0.2, n_classes / len(df))
    
    # 使用分层抽样进行训练集和测试集的划分
    train_data, test_data = train_test_split(df, test_size=min_test_size, random_state=42, stratify=df[label_col])
    
    # 确保测试集包含所有类别
    while test_data[label_col].nunique() < n_classes:
        train_data, test_data = train_test_split(df, test_size=min_test_size, random_state=42, stratify=df[label_col])
        min_test_size += 0.05
    
    logger.info(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
    
    # 设置 AutoGluon 训练参数
    predictor = TabularPredictor(
        label=label_col,
        path=save_dir,
        eval_metric=eval_metric,
        problem_type=problem_type
    )
    
    # 训练模型
    predictor.fit(
        train_data=train_data,
        time_limit=time_limit,
        hyperparameters=hyperparameters
    )
    
    # 评估模型
    performance = predictor.evaluate(test_data)
    
    logger.info(f"模型性能: {performance}")
    
    return predictor, performance, train_data, test_data

def create_excel_report(predictor, train_data, save_dir):
    """生成AutoGluon模型Excel报告"""
    report_dir = Path(save_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    excel_path = report_dir / "autogluon_model_summary.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 模型性能 sheet
        leaderboard = predictor.leaderboard(silent=True)
        leaderboard.to_excel(writer, sheet_name='Model Performance')
        
        # 特征重要性 sheet
        try:
            feature_importance = predictor.feature_importance(train_data)
            feature_importance.to_excel(writer, sheet_name='Feature Importance')
        except Exception as e:
            logger.warning(f"无法生成特征重要性: {str(e)}")
        
        # 模型堆叠信息 sheet
        try:
            model_info = predictor.get_model_full_dict()
            model_names = list(model_info.keys())
            model_types = [str(type(model)) for model in model_info.values()]
            pd.DataFrame({'Model Name': model_names, 'Model Type': model_types}).to_excel(writer, sheet_name='Model Stack Info',index=False)
        except Exception as e:
            logger.warning(f"无法获取模型堆叠信息: {str(e)}")

    logger.info(f"Excel报告已保存至 {excel_path}")

def visualize_performance(predictor, train_data, test_data, save_dir):
    """生成AutoGluon模型性能对比图"""
    report_dir = Path(save_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取训练集和测试集的性能指标
    train_metrics = predictor.leaderboard(train_data, silent=True)
    test_metrics = predictor.leaderboard(test_data, silent=True)
    
    # 合并训练集和测试集的结果
    performance_data = pd.merge(train_metrics[['model', 'score_val']], 
                                test_metrics[['model', 'score_test']], 
                                on='model')
    
    # 重命名列
    performance_data.rename(columns={
        'score_val': 'Train F1',
        'score_test': 'Test F1'
    }, inplace=True)
    
    # 创建图表，适当增大图表高度
    plt.figure(figsize=(14, 10))
    
    # 设置柱状图的宽度和位置
    bar_width = 0.35
    r1 = range(len(performance_data))
    r2 = [x + bar_width for x in r1]
    
    # 绘制柱状图
    plt.bar(r1, performance_data['Train F1'], color='skyblue', width=bar_width, label='Train F1')
    plt.bar(r2, performance_data['Test F1'], color='lightgreen', width=bar_width, label='Test F1')
    
    # 设置图表标题和标签
    plt.title('Model Performance Comparison (F1 Score)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.xticks([r + bar_width/2 for r in range(len(performance_data))], performance_data['model'], rotation=45, ha='right')
    
    # 添加图例，调整位置到图表右上方
    
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    # 添加图例，调整位置到图表内部右上角
    plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), ncol=2)
    
    # 添加数值标签
    for i, (train, test) in enumerate(zip(performance_data['Train F1'], performance_data['Test F1'])):
        plt.text(i, train, f'{train:.3f}', ha='center', va='bottom')
        plt.text(i + bar_width, test, f'{test:.3f}', ha='center', va='bottom')
    
    # 调整布局，增加顶部边距
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 增加顶部边距
    
    # 保存图表
    plt.savefig(report_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"模型性能对比图已保存至 {report_dir / 'model_performance_comparison.png'}")

def visualize_feature_importance(predictor, data, save_dir):
    """生成特征重要性热图"""
    report_dir = Path(save_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        importance = predictor.feature_importance(data)
        # 确保 importance 是一维的
        importance = importance.squeeze()
        if importance.ndim != 1:
            raise ValueError("特征重要性不是一维的")
    except Exception as e:
        logger.warning(f"无法使用AutoGluon的feature_importance方法：{str(e)}。尝试使用permutation_importance。")
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(predictor, data.drop(columns=[predictor.label]), data[predictor.label])
            importance = pd.Series(result.importances_mean, index=data.drop(columns=[predictor.label]).columns)
        except Exception as e:
            logger.warning(f"无法生成特征重要性：{str(e)}")
            return

    plt.figure(figsize=(12, 10))
    sns.barplot(x=importance.values, y=importance.index)
    plt.title('特征重要性')
    plt.xlabel('重要性得分')
    plt.ylabel('特征')
    plt.tight_layout()
    plt.savefig(report_dir / 'feature_importance.png')
    plt.close()
    
    logger.info(f"特征重要性图已保存至 {report_dir / 'feature_importance.png'}")

def generate_training_summary(predictor, save_dir):
    """生成训练摘要"""
    report_dir = Path(save_dir) / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = report_dir / "training_summary.txt"
    
    with open(summary_path, 'w') as f:
        summary = predictor.fit_summary()
        for key, value in summary.items():
            f.write(f"{key}:\n{value}\n\n")
    
    logger.info(f"训练摘要已保存至 {summary_path}")

def main(file_path, label_col, feature_cols, categorical_features, save_dir, time_limit, eval_metric, problem_type, hyperparameters):
    """主函数"""
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        df = load_data(file_path, label_col, feature_cols, categorical_features)
        
        logger.info(f"标签列 '{label_col}' 的唯一值: {df[label_col].unique()}")
        logger.info(f"标签列 '{label_col}' 的值计数:\n{df[label_col].value_counts()}")
        logger.info(f"类别数量: {df[label_col].nunique()}")
        logger.info(f"总样本数: {len(df)}")
        
        predictor, performance, train_data, test_data = train_model(df, label_col, feature_cols, categorical_features, save_dir, time_limit, eval_metric, problem_type, hyperparameters)
        
        # 生成Excel报告
        create_excel_report(predictor, df, save_dir)
        
        # 生成性能对比图
        visualize_performance(predictor, train_data, test_data, save_dir)
        
        # 生成特征重要性热图
        visualize_feature_importance(predictor, test_data, save_dir)
        
        # 生成训练摘要
        generate_training_summary(predictor, save_dir)
        
        logger.info("模型训练和评估完毕。结果和可视化已保存。")
        return performance
    except Exception as e:
        logger.error(f"主函数中发生错误：{str(e)}")
        logger.error(f"错误详情：", exc_info=True)
        raise

if __name__ == "__main__":
    # 示例用法
    file_path = Path(r"D:\soil-mapping\data\table\type\feature_gl_type.csv")
    label_col = "tz"
    feature_cols = ['a_DEM','a_evi','a_lat','a_lon','a_lswi','a_Mean','a_mndwi','a_ndmi','a_ndvi','a_ndwi','a_NIGHT2022','a_pca_1','a_pca_2']
    categorical_features = ['tl','yl']
    save_dir = Path(r"D:\soil-mapping\models\soil_type\autogluon_type")
    time_limit = 3600  # 1小时
    eval_metric = 'f1_weighted'
    problem_type = 'multiclass'
    hyperparameters = {
        'NN_TORCH': {},
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}
        ]
    }

    try:
        performance = main(file_path, label_col, feature_cols, categorical_features, save_dir, time_limit, eval_metric, problem_type, hyperparameters)
        print(f"模型性能: {performance}")
    except Exception as e:
        logger.error(f"程序执行过程中发生错误：{str(e)}")
        logger.error(f"错误详情：", exc_info=True)