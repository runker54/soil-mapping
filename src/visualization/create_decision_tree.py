import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from dtreeviz.trees import dtreeviz
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# 设置日志
log_file = Path('logs/create_decision_tree.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def create_decision_tree(X, y, feature_names, target_name, task='classification', max_depth=3, min_samples_leaf=5):
    """
    创建并可视化决策树
    
    参数:
    X : 特征数据
    y : 目标变量
    feature_names : 特征名称列表
    target_name : 目标变量名称
    task : 'classification' 或 'regression'
    max_depth : 树的最大深度
    min_samples_leaf : 叶节点的最小样本数
    返回:
    None (保存图像到文件)
    """
    if task == 'classification':
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    else:
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    
    tree.fit(X, y)
    
    viz = dtreeviz(tree,
                   X,
                   y,
                   target_name=target_name,
                   feature_names=feature_names,
                   class_names=list(set(y)) if task == 'classification' else None)
    
    viz.save(f"decision_tree_{target_name}.svg")
    print(f"决策树已保存为 decision_tree_{target_name}.svg")

# 使用示例
if __name__ == "__main__":
    # 加载数据
    data = pd.read_csv("path_to_your_data.csv")
    
    # 土壤属性决策树
    X_soil_properties = data[['DEM', 'Slope', 'NDVI', 'Precipitation']]
    y_soil_ph = data['Soil_pH']
    create_decision_tree(X_soil_properties, y_soil_ph, 
                         X_soil_properties.columns.tolist(), 'Soil_pH', 
                         task='regression', max_depth=4, min_samples_leaf=10)
    
    # 土壤类型决策树
    X_soil_types = data[['Clay', 'Sand', 'Organic_Matter', 'CEC']]
    y_soil_type = data['Soil_Type']
    create_decision_tree(X_soil_types, y_soil_type, 
                         X_soil_types.columns.tolist(), 'Soil_Type', 
                         task='classification', max_depth=5, min_samples_leaf=15)