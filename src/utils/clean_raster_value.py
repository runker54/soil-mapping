import rasterio
import numpy as np
from scipy.signal import convolve2d
from pathlib import Path
from enum import Enum
from typing import List, Tuple
import logging
# 设置日志
log_file = Path('logs/clearn_raster_value.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Condition(Enum):
    GREATER_THAN = '>'
    LESS_THAN = '<'
    EQUAL_TO = '=='
    GREATER_EQUAL = '>='
    LESS_EQUAL = '<='
    NOT_EQUAL = '!='

class FillMethod(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MIN = 'min'
    MAX = 'max'
    CUSTOM = 'custom'

def create_condition_mask(data, conditions):
    condition_map = {
        Condition.GREATER_THAN: np.greater,
        Condition.LESS_THAN: np.less,
        Condition.EQUAL_TO: np.equal,
        Condition.GREATER_EQUAL: np.greater_equal,
        Condition.LESS_EQUAL: np.less_equal,
        Condition.NOT_EQUAL: np.not_equal
    }
    
    mask = np.zeros_like(data, dtype=bool)
    for condition, threshold in conditions:
        mask |= condition_map[condition](data, threshold)
    
    return mask

def calculate_fill_value(data, window_size, fill_method, custom_value=None):
    if fill_method == FillMethod.CUSTOM:
        return np.full_like(data, custom_value)
    
    kernel = np.ones((window_size, window_size))
    
    if fill_method == FillMethod.MEAN:
        local_sum = convolve2d(data, kernel, mode='same', boundary='symm')
        local_count = convolve2d(np.ones_like(data), kernel, mode='same', boundary='symm')
        return local_sum / local_count
    elif fill_method == FillMethod.MEDIAN:
        h, w = data.shape
        extended_data = np.pad(data, window_size//2, mode='symmetric')
        local_values = np.array([[extended_data[i:i+window_size, j:j+window_size].ravel() 
                                  for j in range(w)] for i in range(h)])
        return np.median(local_values, axis=2)
    elif fill_method == FillMethod.MIN:
        return convolve2d(-data, kernel, mode='same', boundary='symm') / (window_size**2)
    elif fill_method == FillMethod.MAX:
        return -convolve2d(-data, kernel, mode='same', boundary='symm') / (window_size**2)

def process_raster(input_raster: str, output_raster: str, 
                   conditions: List[Tuple[Condition, float]], 
                   fill_method: FillMethod, window_size: int = 3, 
                   custom_value: float = None):
    """
    Process a raster file by replacing values based on multiple specified conditions and fill methods.
    
    :param input_raster: Path to the input raster file
    :param output_raster: Path to save the output raster file
    :param conditions: List of tuples, each containing a Condition enum and a threshold value
    :param fill_method: Method to calculate replacement values (use FillMethod enum)
    :param window_size: Size of the window for local calculations (default: 3)
    :param custom_value: Custom value to use if fill_method is CUSTOM
    """
    with rasterio.open(input_raster) as src:
        data = src.read(1)  # Assuming single band raster
        
        mask = create_condition_mask(data, conditions)
        
        fill_values = calculate_fill_value(data, mask, window_size, fill_method, custom_value)
        
        # Replace values where the mask is True
        data[mask] = fill_values[mask]
        
        # Prepare the output raster
        output_profile = src.profile.copy()
        output_profile.update(dtype=rasterio.float32)
        
        # Write the result
        with rasterio.open(output_raster, 'w', **output_profile) as dst:
            dst.write(data.astype(rasterio.float32), 1)

    print(f"处理完成。输出文件: {output_raster}")

# 使用示例
if __name__ == "__main__":
    input_raster = r"F:\cache_data\tif_file\saga\SB\Relative Slope Position.tif"
    output_raster = r"F:\cache_data\tif_file\saga\SB\Relative Slope Position2.tif"
    # 示例：将大于1000或小于0的值替换为3x3窗口的均值
    conditions = [
        (Condition.GREATER_THAN, 100),
        (Condition.LESS_THAN, -10)
    ]
    
    # process_raster(input_raster, output_raster, 
    #                conditions=conditions, 
    #                fill_method=FillMethod.MEAN, 
    #                window_size=3)
    
    # # 示例：将等于-9999或大于10000的值替换为自定义值0
    # conditions = [
    #     (Condition.EQUAL_TO, -9999),
    #     (Condition.GREATER_THAN, 10000)
    # ]
    
    process_raster(input_raster, output_raster, 
                   conditions=conditions, 
                   fill_method=FillMethod.CUSTOM, 
                   custom_value=0)

