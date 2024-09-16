import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns

# 设置日志
log_file = Path('logs/train_soil_type_rf.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path, label_col):
    """加载数据并处理缺失值"""
    logger.info(f"正在从 {file_path} 加载数据")
    data = pd.read_csv(file_path)
    numeric_cols = data.select_dtypes(include=[np.number])
    data[numeric_cols.columns] = data[numeric_cols.columns].fillna(numeric_cols.median())
    logger.info(f"数据加载完成。形状：{data.shape}")
    return data

def preprocess_data(df, feature_cols, categorical_features, label_col, save_dir):
    """预处理数据"""
    # 处理类别特征
    for col in categorical_features:
        if col in feature_cols:
            logger.info(f"对类别特征 {col} 进行编码")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # 处理标签列
    le_label = LabelEncoder()
    df[label_col] = le_label.fit_transform(df[label_col].astype(str))
    logger.info(f"标签编码器类别: {le_label.classes_}")
    logger.info(f"标签编码器类别数量: {len(le_label.classes_)}")
    logger.info(f"标签编码器类别类型: {type(le_label.classes_[0])}")
    model_dir = Path(save_dir) / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    # 保存标签编码器
    with open(model_dir / f"{label_col}_encoder.pkl", 'wb') as f:
        pickle.dump(le_label, f)
    
    logger.info(f"标签编码器已保存为 {label_col}_encoder.pkl")
    
    # 处理缺失值
    df = df.dropna(subset=[label_col] + feature_cols)
    
    logger.info(f"数据预处理完成。新形状：{df.shape}")
    return df, le_label, categorical_features

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
    
    if best_params['max_depth'] is None:
        param_grid_fine['max_depth'] = [None, 10, 20]
    else:
        param_grid_fine['max_depth'] = [max(1, best_params['max_depth'] - 5), best_params['max_depth'], best_params['max_depth'] + 5]
    
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid_fine, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    logger.info(f"网格搜索得到的最佳参数：{best_params}")
    return grid_search.best_estimator_

def train_model(df, label_col, feature_cols, categorical_features, param_grid, save_dir, use_feature_optimization):
    """训练模型"""
    logger.info(f"正在训练 {label_col} 的模型")
    
    # 预处理数据
    df_processed, le_label, categorical_features = preprocess_data(df, feature_cols, categorical_features, label_col,save_dir)
    
    X = df_processed[feature_cols]
    y = df_processed[label_col]
    
    if use_feature_optimization:
        estimator = RandomForestClassifier(random_state=42)
        selected_features = feature_optimization(X, y, estimator, feature_cols)
        X_selected = X[selected_features]
    else:
        selected_features = feature_cols
        X_selected = X
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # 对于随机森林，我们不需要特别处理类别特征，它可以自动处理
    estimator = RandomForestClassifier(random_state=42)
    
    # 超参数调优
    best_model = hyperparameter_tuning(X_train, y_train, estimator, param_grid)
    
    # 评估模型
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    metrics = evaluate_classification(y_test, y_test_pred)
    
    # 保存模型
    save_model(best_model, selected_features, save_dir, label_col, le_label, categorical_features)
    
    return {
        "selected_features": selected_features,
        "metrics": metrics,
        "best_params": best_model.get_params(),
        "best_model": best_model
    }

def evaluate_classification(y_true, y_pred):
    """评估分类模型"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "Confusion Matrix": cm}

def save_model(model, feature_names, save_dir, label_col, le_label, categorical_features):
    """保存模型"""
    model_dir = Path(save_dir) / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'label_encoder': le_label,
        'categorical_features': categorical_features
    }
    
    logger.info(f"保存的标签编码器类别: {le_label.classes_}")
    logger.info(f"保存的标签编码器类别数量: {len(le_label.classes_)}")
    
    with open(model_dir / f"{label_col}_model.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"模型已保存，特征数量：{len(feature_names)}，特征列表：{feature_names}")
    logger.info(f"类别特征：{categorical_features}")

def create_excel_report(results, save_dir):
    """生成Excel报告"""
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
                metrics = data['metrics']
                performance_data.append({
                    'Label': label,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1': metrics['F1']
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
    """生成性能对比图"""
    report_dir = save_dir / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        
        data = [results[label]['metrics'][metric] for label in results if results[label]]
        labels = [label for label in results if results[label]]
        
        bars = ax.bar(labels, data)
        
        ax.set_xlabel('Labels')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric} Comparison')
        ax.set_ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(report_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_importance(results, save_dir):
    """生成特征重要性热图"""
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

def main(file_path, label_col, feature_cols, categorical_features, param_grid, save_dir, use_feature_optimization):
    """主函数"""
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        df = load_data(file_path, label_col)
        result = train_model(df, label_col, feature_cols, categorical_features, param_grid, save_dir, use_feature_optimization)
        
        if result:
            results = {label_col: result}
            
            # 生成Excel报告
            create_excel_report(results, save_dir)
            
            # 生成性能对比图
            visualize_performance(results, save_dir)
            
            # 生成特征重要性热图
            visualize_feature_importance(results, save_dir)
        
        logger.info("模型训练和评估完毕。结果和可视化已保存。")
    except Exception as e:
        logger.error(f"主函数中发生错误：{str(e)}")

if __name__ == "__main__":
    # 示例用法
    file_path = Path(r"C:\Users\Runker\Desktop\GL\sample_csv_type\feature_gl_type.csv")
    label_col = "tz"
    feature_cols = ['a_DEM','a_evi','a_lat','a_lon','a_lswi','a_Mean','a_mndwi','a_ndmi','a_ndvi','a_ndwi','a_NIGHT2022','a_pca_1','a_pca_2']
    categorical_features = ['tl','yl']
    param_grid = {
        'n_estimators': np.arange(10, 100, 10),
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    save_dir = Path(r"C:\Users\Runker\Desktop\GL\rf_type")
    use_feature_optimization = True

    try:
        main(file_path, label_col, feature_cols, categorical_features, param_grid, save_dir, use_feature_optimization)
    except Exception as e:
        logger.error(f"发生错误：{str(e)}")