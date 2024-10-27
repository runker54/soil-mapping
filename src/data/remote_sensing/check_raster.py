import rasterio
import numpy as np
from rasterio.windows import Window
import os
import logging
import sys
from tqdm import tqdm
from pathlib import Path


def replace_nodata(raster_path, window_size=3, chunk_size=1024, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    total_replaced = 0
    with rasterio.open(raster_path, 'r+') as src:
        nodata = src.nodata
        width = src.width
        height = src.height
        
        nodata_mask = src.read_masks(1) == 0
        total_nodata = np.sum(nodata_mask)
        
        where_nodata = np.where(nodata_mask)
        if len(where_nodata[0]) == 0:
            logger.info(f"在 {raster_path} 中未找到无数据值")
            return 0
        
        y_min, y_max = np.min(where_nodata[0]), np.max(where_nodata[0])
        x_min, x_max = np.min(where_nodata[1]), np.max(where_nodata[1])
        
        logger.info(f"无数据区域: ({x_min}, {y_min}) 到 ({x_max}, {y_max})")

        data = src.read(1)
        valid_data = data[data != nodata]
        global_mean = np.mean(valid_data) if valid_data.size > 0 else 0
        logger.info(f"全局平均值: {global_mean}")

        for y in tqdm(range(y_min, y_max + 1, chunk_size), desc="处理数据块"):
            for x in range(x_min, x_max + 1, chunk_size):
                window = Window(x, y, min(chunk_size, width - x), min(chunk_size, height - y))
                data = src.read(1, window=window)
                mask = nodata_mask[y:y+window.height, x:x+window.width]
                
                if not np.any(mask):
                    continue
                
                local_mean = np.zeros_like(data, dtype=np.float32)
                count = np.zeros_like(data, dtype=np.float32)
                
                for i in range(-window_size//2, window_size//2 + 1):
                    for j in range(-window_size//2, window_size//2 + 1):
                        if i == 0 and j == 0:
                            continue
                        y_shift = max(0, -i)
                        x_shift = max(0, -j)
                        y_slice = slice(max(0, i), min(data.shape[0], data.shape[0] + i))
                        x_slice = slice(max(0, j), min(data.shape[1], data.shape[1] + j))
                        
                        shifted_data = data[y_slice, x_slice]
                        shifted_mask = shifted_data != nodata
                        
                        local_mean[y_shift:y_shift+shifted_data.shape[0], x_shift:x_shift+shifted_data.shape[1]] += \
                            np.where(shifted_mask, shifted_data, 0)
                        count[y_shift:y_shift+shifted_data.shape[0], x_shift:x_shift+shifted_data.shape[1]] += shifted_mask

                count[count == 0] = 1
                local_mean /= count

                replaced_mask = mask & (data == nodata)
                data[replaced_mask] = np.where(count[replaced_mask] > 0, local_mean[replaced_mask], global_mean)
                
                total_replaced += np.sum(replaced_mask)
                
                src.write(data, 1, window=window)

    logger.info(f"在 {raster_path} 中替换的无数据值: {total_replaced} / {total_nodata} ({total_replaced/total_nodata*100:.2f}%)")
    return total_replaced

def check_raster_consistency(raster_path, logger):
    raster_files = [f for f in os.listdir(raster_path) if f.endswith(('.tif', '.tiff'))]
    
    if not raster_files:
        logger.warning(f"在 {raster_path} 中未找到栅格文件")
        return
    
    logger.info(f"在 {raster_path} 中找到 {len(raster_files)} 个栅格文件")
    
    reference_raster = None
    inconsistencies = []
    
    for raster_file in tqdm(raster_files, desc="处理栅格"):
        full_path = os.path.join(raster_path, raster_file)
        try:
            with rasterio.open(full_path) as src:
                crs = src.crs
                pixel_size = src.res
                dimensions = (src.width, src.height)
                
                if reference_raster is None:
                    reference_raster = {
                        'crs': crs,
                        'pixel_size': pixel_size,
                        'dimensions': dimensions
                    }
                    logger.info(f"参考栅格设置为: {raster_file}")
                    logger.info(f"坐标参考系统: {crs}")
                    logger.info(f"像素大小: {pixel_size}")
                    logger.info(f"尺寸: {dimensions}")
                else:
                    if crs != reference_raster['crs']:
                        inconsistencies.append(f"{raster_file}: 坐标参考系统不匹配。预期 {reference_raster['crs']}, 实际 {crs}")
                    if pixel_size != reference_raster['pixel_size']:
                        inconsistencies.append(f"{raster_file}: 像素大小不匹配。预期 {reference_raster['pixel_size']}, 实际 {pixel_size}")
                    if dimensions != reference_raster['dimensions']:
                        inconsistencies.append(f"{raster_file}: 尺寸不匹配。预期 {reference_raster['dimensions']}, 实际 {dimensions}")
            
            replaced_count = replace_nodata(full_path, logger=logger)
            logger.info(f"在 {raster_file} 中替换了 {replaced_count} 个无数据值")
        
        except rasterio.errors.RasterioIOError:
            logger.error(f"无法打开 {raster_file}。可能已损坏或不是有效的栅格文件。")
        except Exception as e:
            logger.error(f"处理 {raster_file} 时出错: {str(e)}")
    
    if inconsistencies:
        logger.warning("发现以下不一致:")
        for inconsistency in inconsistencies:
            logger.warning(inconsistency)
    else:
        logger.info("所有栅格在坐标参考系统、像素大小和尺寸上都是一致的。")

def main(raster_path,log_file):
    # 设置日志
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',encoding='utf-8')
    logger = logging.getLogger(__name__)
    logger.info("开始检查栅格数据")
    try:
        check_raster_consistency(raster_path, logger)
    except Exception as e:
        logger.error(f"检查栅格数据时发生错误: {e}")
        raise
    finally:
        logger.info("栅格数据检查完成")
        

# 测试
if __name__ == "__main__":
    raster_path = r'D:\soil-mapping\data\soil_property'
    log_file = r'D:\soil-mapping\logs\check_raster.log'
    main(raster_path, log_file)
