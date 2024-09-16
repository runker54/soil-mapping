import os
import logging
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from pykrige.rk import RegressionKriging
from tqdm import tqdm
from pathlib import Path

# 设置日志
log_file = Path('logs/train_soil_property.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path,label_col):
    """加载数据并处理缺失值"""
    logger.info(f"正在从 {file_path} 加载数据")
    data = pd.read_csv(file_path)
    numeric_cols = data.select_dtypes(include=[np.number])
    means = numeric_cols.mean()
    data[numeric_cols.columns] = data[numeric_cols.columns].fillna(means)
    # 仅选择标记正常的点位
    data = data[data[f"{label_col}_Sta"]=='Normal']
    logger.info(f"数据加载完成。形状：{data.shape}")
    return data

def preprocess_data(df, feature_cols, label_col):
    """预处理数据"""
    # 检查并转换特征列的数据类型
    for col in feature_cols:
        if df[col].dtype == 'object':
            logger.warning(f"列 {col} 包含非数值数据。尝试进行转换。")
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                logger.error(f"无法将列 {col} 转换为数值。考虑删除此列或对其进行编码。")
                return None
    
    # 处理缺失值
    df = df.dropna(subset=[label_col] + feature_cols)
    
    # 检查是否有无限值，并将其替换为 NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 再次删除包含 NaN 的行
    df = df.dropna(subset=[label_col] + feature_cols)
    
    logger.info(f"数据预处理完成。新形状：{df.shape}")
    return df

def feature_optimization(X, y, estimator, feature_cols):
    """特征优化"""
    logger.info("开始特征优化")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X[feature_cols], y)
    selected_features = list(np.array(feature_cols)[selector.support_])
    logger.info(f"选择的特征：{selected_features}")
    return selected_features

def hyperparameter_tuning(X, y, estimator, param_grid):
    """超参数调优"""
    logger.info("开始超参数调优")
    n_iter_search = 50
    random_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, n_iter=n_iter_search, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X, y)
    best_params = random_search.best_params_
    logger.info(f"随机搜索得到的最佳参数：{best_params}")
    
    param_grid_fine = {
        'n_estimators': [max(10, best_params['n_estimators'] - 50), best_params['n_estimators'], min(1000, best_params['n_estimators'] + 50)],
        'min_samples_split': [max(2, best_params['min_samples_split'] - 2), best_params['min_samples_split'], best_params['min_samples_split'] + 2],
        'min_samples_leaf': [max(1, best_params['min_samples_leaf'] - 1), best_params['min_samples_leaf'], best_params['min_samples_leaf'] + 1]
    }
    
    # 处理 max_depth 可能为 None 的情况
    if best_params['max_depth'] is None:
        param_grid_fine['max_depth'] = [None, 10, 20]
    else:
        param_grid_fine['max_depth'] = [max(1, best_params['max_depth'] - 5), best_params['max_depth'], best_params['max_depth'] + 5]
    
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid_fine, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    logger.info(f"网格搜索得到的最佳参数：{best_params}")
    return grid_search.best_estimator_

def train_model(df, label_col, feature_cols, coord_cols, param_grid, save_dir, problem_type, use_feature_optimization):
    """训练模型"""
    logger.info(f"正在训练 {label_col} 的模型")
    
    # 预处理数据
    df_processed = preprocess_data(df, feature_cols + coord_cols, label_col)
    if df_processed is None:
        logger.error(f"{label_col} 的数据预处理失败")
        return None
    
    X = df_processed[feature_cols + coord_cols]
    y = df_processed[label_col]
    
    # 检查 y 的类型，如果是分类问题且 y 不是数值类型，进行编码
    if problem_type == 'classification' and y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    if use_feature_optimization:
        if problem_type == 'regression':
            estimator = RandomForestRegressor(random_state=42)
        else:
            estimator = RandomForestClassifier(random_state=42)
        selected_features = feature_optimization(X, y, estimator, feature_cols)
        X_selected = X[selected_features + coord_cols]
    else:
        selected_features = feature_cols
        X_selected = X[feature_cols + coord_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # 确保所有特征都是数值类型
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    if problem_type == 'regression':
        estimator = RandomForestRegressor(random_state=42)
    else:
        estimator = RandomForestClassifier(random_state=42)
    
    # 超参数调优
    best_model = hyperparameter_tuning(X_train, y_train, estimator, param_grid)
    
    # 评估RF模型
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    if problem_type == 'regression':
        rf_metrics = evaluate_regression(y_test, y_test_pred)
        
        # 应用克里金
        rk = RegressionKriging(regression_model=best_model, n_closest_points=8,variogram_model='spherical')
        rk.fit(X_train[selected_features].values, X_train[coord_cols].values, y_train)
        y_pred_rk = rk.predict(X_test[selected_features].values, X_test[coord_cols].values)
        rk_metrics = evaluate_regression(y_test, y_pred_rk)
    else:
        rf_metrics = evaluate_classification(y_test, y_test_pred)
        rk_metrics = None
    
    # 保存模型
    final_features = selected_features + coord_cols
    save_model(best_model, final_features, save_dir, label_col)
    
    return {
        "selected_features": selected_features,
        "rf_metrics": rf_metrics,
        "rk_metrics": rk_metrics,
        "best_params": best_model.get_params(),
        "best_model": best_model,
        "final_features": final_features  # 添加这一行
    }

def save_model(model, feature_names, save_dir, label_col):
    """保存模型"""
    model_dir = save_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'n_features': len(feature_names)
    }
    
    with open(model_dir / f"{label_col}_model.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"模型已保存，特征数量：{len(feature_names)}，特征列表：{feature_names}")

def evaluate_regression(y_true, y_pred):
    """评估回归模型"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

def evaluate_classification(y_true, y_pred):
    """评估分类模型"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}

def save_model(model, feature_names, save_dir, label_col):
    """保存模型"""
    model_dir = save_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / f"{label_col}_model.pkl", 'wb') as f:
        pickle.dump({'model': model, 'feature_names': feature_names}, f)
def create_excel_report(results, save_dir):
    report_dir = save_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    excel_path = report_dir / "model_summary.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 特征选择 sheet
        all_features = set()
        for data in results.values():
            if data:
                all_features.update(data['selected_features'])
        all_features = sorted(list(all_features))
        
        feature_df = pd.DataFrame(index=all_features)
        for label, data in results.items():
            if data:
                feature_df[label] = [1 if feature in data['selected_features'] else 0 for feature in all_features]
        feature_df.to_excel(writer, sheet_name='Selected Features')
        
        # 模型性能 sheet
        performance_data = []
        for label, data in results.items():
            if data:
                rf_metrics = data['rf_metrics']
                rk_metrics = data['rk_metrics']
                performance_data.append({
                    'Label': label,
                    'RF_R2': rf_metrics['R2'],
                    'RF_MAE': rf_metrics['MAE'],
                    'RF_MSE': rf_metrics['MSE'],
                    'RF_RMSE': rf_metrics['RMSE'],
                    'RFRK_R2': rk_metrics['R2'] if rk_metrics else None,
                    'RFRK_MAE': rk_metrics['MAE'] if rk_metrics else None,
                    'RFRK_MSE': rk_metrics['MSE'] if rk_metrics else None,
                    'RFRK_RMSE': rk_metrics['RMSE'] if rk_metrics else None
                })
        pd.DataFrame(performance_data).set_index('Label').to_excel(writer, sheet_name='Model Performance')
        
        # 超参数 sheet
        hyperparams_data = []
        for label, data in results.items():
            if data:
                params = data['best_params']
                hyperparams_data.append({
                    'Label': label,
                    'n_estimators': params['n_estimators'],
                    'max_depth': params['max_depth'],
                    'min_samples_split': params['min_samples_split'],
                    'min_samples_leaf': params['min_samples_leaf']
                })
        pd.DataFrame(hyperparams_data).set_index('Label').to_excel(writer, sheet_name='Hyperparameters')

    logger.info(f"Excel报告已保存至 {excel_path}")

def visualize_performance(results, save_dir):
    """生成优化后的性能对比图"""
    report_dir = save_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    summary_data = []
    
    for label, data in results.items():
        if data is None:
            logger.warning(f"由于 {label} 的模型训练失败，跳过可视化。")
            continue
        
        rf_metrics = data['rf_metrics']
        rk_metrics = data['rk_metrics']
        
        for metric, value in rf_metrics.items():
            summary_data.append({'Label': label, 'Metric': metric, 'RF': value, 'RFRK': rk_metrics[metric] if rk_metrics else None})
    
    if not summary_data:
        logger.error("没有有效结果可供可视化。")
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    # 绘制性能对比图
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    metrics = summary_df['Metric'].unique()
    
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        metric_data = summary_df[summary_df['Metric'] == metric]
        
        x = np.arange(len(metric_data['Label']))
        width = 0.35
        
        rf_bars = ax.bar(x - width/2, metric_data['RF'], width, label='RF', alpha=0.7)
        rfrk_bars = ax.bar(x + width/2, metric_data['RFRK'], width, label='RFRK', alpha=0.7)
        
        ax.set_xlabel('Labels')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_data['Label'], rotation=45, ha='right')
        ax.legend()
        
        # 设置刻度和格式化标签
        if metric == 'R2':
            ax.set_ylim(0, min(1.1, max(metric_data['RF'].max(), metric_data['RFRK'].max()) * 1.1))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))
        else:
            ax.set_yscale('symlog', linthresh=1e-1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2e}' if y < 0.01 or y > 100 else f'{y:.2f}'))
            
            # 动态调整 y 轴上限
            max_value = max(metric_data['RF'].max(), metric_data['RFRK'].max())
            ax.set_ylim(top=max_value * 2)
        
        # 添加数值标签
        for bars in [rf_bars, rfrk_bars]:
            for bar in bars:
                height = bar.get_height()
                label = f'{height:.2f}' if 0.01 <= height <= 100 else f'{height:.2e}'
                ax.annotate(label,
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=45, fontsize=8)
        
        ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(report_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_importance(results, save_dir):
    report_dir = save_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    importance_data = {}
    for label, data in results.items():
        if data and hasattr(data['best_model'], 'feature_importances_'):
            importance_data[label] = dict(zip(data['selected_features'], data['best_model'].feature_importances_))
    
    if not importance_data:
        logger.warning("没有特征重要性数据可供可视化。")
        return
    
    importance_df = pd.DataFrame(importance_data).fillna(0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(importance_df, annot=True, cmap='YlOrRd', fmt='.3f')
    plt.title('Feature Importance Comparison')
    plt.ylabel('Features')
    plt.xlabel('Labels')
    plt.tight_layout()
    plt.savefig(report_dir / 'feature_importance_comparison.png')
    plt.close()

def main(file_path, label_dict, feature_cols, coord_cols, param_grid, save_dir, use_feature_optimization):
    """主函数"""
    try:
        results = {}
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保所有特征名称都是字符串
        feature_cols = [str(col) for col in feature_cols]
        coord_cols = [str(col) for col in coord_cols]
        
        for label_col, problem_type in tqdm(label_dict.items(), desc="处理标签"):
            df = load_data(file_path, label_col)
            result = train_model(df, label_col, feature_cols, coord_cols, param_grid, save_dir, problem_type, use_feature_optimization)
            results[label_col] = result
            if result is None:
                logger.warning(f"模型训练失败：{label_col}")
        
        # 生成Excel报告
        create_excel_report(results, save_dir)
        
        # 生成性能对比图
        visualize_performance(results, save_dir)
        
        # 生成特征重要性热图
        visualize_feature_importance(results, save_dir)
        
        logger.info("所有模型已训练和评估完毕。结果和可视化已保存。")
    except Exception as e:
        logger.error(f"主函数中发生错误：{str(e)}")

if __name__ == "__main__":
    # 示例用法
    file_path = Path(r"C:\Users\Runker\Desktop\GL\sample_csv\feature_gl.csv")
    label_dict = {
        "ph": "regression",
        "yjz": "regression",
        "ylzjhl": "regression",
        "qdan": "regression",
        "qlin": "regression",
        "yxlin": "regression",
        "sxjia": "regression",
        "hxjia": "regression",

    }
    feature_cols = ['a_DEM', 'a_evi',  'a_lswi', 'a_Mean','a_mndwi', 'a_ndmi', 'a_ndvi', 'a_ndwi', 'a_NIGHT2022', 'a_pca_1','a_pca_2']  # 您的特征列
    coord_cols = ["a_lon", "a_lat"]
    param_grid = {
        'n_estimators': np.arange(10, 200, 10),
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    save_dir = Path(r"C:\Users\Runker\Desktop\GL\rfrk")
    use_feature_optimization = True

    try:
        main(file_path, label_dict, feature_cols, coord_cols, param_grid, save_dir, use_feature_optimization)
    except Exception as e:
        logger.error(f"发生错误：{str(e)}")