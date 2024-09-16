
# from src.utils.logger import setup_logger
import rasterio
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
# 设置日志
log_file = Path('logs/multiband_index_calculator.log')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Safely divide two arrays, handling division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        if isinstance(fill_value, (int, float)):
            result[~np.isfinite(result)] = fill_value
        else:
            result = np.where(np.isfinite(result), result, fill_value)
    return result

class RemoteSensingIndices:
    def __init__(self, image_path: str):
        self.image_path = image_path
        with rasterio.open(self.image_path) as src:
            self.meta = src.meta.copy()
            self.transform = src.transform
            self.crs = src.crs

    def _read_bands(self, band_numbers: List[int]) -> List[np.ndarray]:
        with rasterio.open(self.image_path) as src:
            bands = [src.read(i).astype(np.float32) for i in band_numbers]
        return bands

    def _handle_extreme_values(self, band: np.ndarray) -> np.ndarray:
        band[band > 100] = np.median(band)
        return band

    def calculate_ndvi(self, nir_band_number: int, red_band_number: int) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index"""
        nir, red = self._read_bands([nir_band_number, red_band_number])
        nir, red = self._handle_extreme_values(nir), self._handle_extreme_values(red)
        return safe_divide(nir - red, nir + red)

    def calculate_savi(self, nir_band_number: int, red_band_number: int, L: float = 0.5) -> np.ndarray:
        """Calculate Soil-Adjusted Vegetation Index"""
        nir, red = self._read_bands([nir_band_number, red_band_number])
        nir, red = self._handle_extreme_values(nir), self._handle_extreme_values(red)
        return safe_divide(nir - red, nir + red + L, fill_value=0) * (1.0 + L)

    def calculate_ndwi(self, nir_band_number: int, green_band_number: int) -> np.ndarray:
        """Calculate Normalized Difference Water Index"""
        nir, green = self._read_bands([nir_band_number, green_band_number])
        nir, green = self._handle_extreme_values(nir), self._handle_extreme_values(green)
        return safe_divide(green - nir, green + nir)

    def calculate_evi(self, nir_band_number: int, red_band_number: int, blue_band_number: int, 
                      G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
        """Calculate Enhanced Vegetation Index"""
        nir, red, blue = self._read_bands([nir_band_number, red_band_number, blue_band_number])
        nir, red, blue = self._handle_extreme_values(nir), self._handle_extreme_values(red), self._handle_extreme_values(blue)
        return G * safe_divide(nir - red, nir + C1 * red - C2 * blue + L)

    def calculate_lswi(self, nir_band_number: int, swir_band_number: int) -> np.ndarray:
        """Calculate Land Surface Water Index"""
        nir, swir = self._read_bands([nir_band_number, swir_band_number])
        nir, swir = self._handle_extreme_values(nir), self._handle_extreme_values(swir)
        return safe_divide(nir - swir, nir + swir)

    def calculate_mndwi(self, green_band_number: int, swir_band_number: int) -> np.ndarray:
        """Calculate Modified Normalized Difference Water Index"""
        green, swir = self._read_bands([green_band_number, swir_band_number])
        green, swir = self._handle_extreme_values(green), self._handle_extreme_values(swir)
        return safe_divide(green - swir, green + swir)

    def calculate_ndmi(self, nir_band_number: int, swir_band_number: int) -> np.ndarray:
        """Calculate Normalized Difference Moisture Index"""
        nir, swir = self._read_bands([nir_band_number, swir_band_number])
        nir, swir = self._handle_extreme_values(nir), self._handle_extreme_values(swir)
        return safe_divide(nir - swir, nir + swir)

    def calculate_vari(self, red_band_number: int, green_band_number: int, blue_band_number: int) -> np.ndarray:
        """Calculate Visible Atmospherically Resistant Index"""
        red, green, blue = self._read_bands([red_band_number, green_band_number, blue_band_number])
        red, green, blue = self._handle_extreme_values(red), self._handle_extreme_values(green), self._handle_extreme_values(blue)
        return safe_divide(green - red, green + red - blue)

    def perform_pca(self, select_bands: List[int], n_components: int) -> np.ndarray:
        """Perform Principal Component Analysis"""
        # Read and preprocess bands
        bands = [self._handle_extreme_values(band) for band in self._read_bands(select_bands)]
        # Reshape data
        n_bands, height, width = len(bands), bands[0].shape[0], bands[0].shape[1]
        reshaped_data = np.array(bands).reshape(n_bands, -1).T
        # Perform PCA
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(reshaped_data)
        # Reshape results
        transformed_images = transformed_data.T.reshape((n_components, height, width))
        # Log the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        logger.info(f"Explained variance ratio: {explained_variance_ratio}")
        return transformed_images

    def save_raster(self, data: np.ndarray, output_path: str, count: int = 1) -> None:
        """Save raster data to file"""
        out_meta = self.meta.copy()
        out_meta.update({"count": 1, "dtype": 'float32'})  # Always set count to 1 for single-band output

        if count == 1:
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(data.astype(rasterio.float32), 1)
            logger.info(f"Single-band data saved to {output_path}")
        else:
            # Split multi-band data into separate files
            base_name, ext = os.path.splitext(output_path)
            for i in range(count):
                band_output_path = f"{base_name}_{i+1}{ext}"
                with rasterio.open(band_output_path, "w", **out_meta) as dest:
                    dest.write(data[i].astype(rasterio.float32), 1)
                logger.info(f"Band {i+1} saved to {band_output_path}")

    def plot_pca(self, pca_data: np.ndarray, n_components: int = 2) -> None:
        """Plot PCA components"""
        fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 6))
        if n_components == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            im = ax.imshow(pca_data[i], cmap='RdYlGn')
            ax.set_title(f'PCA Component {i+1}')
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

def process_indices(input_path: str, output_dir: str, indices_config: Dict[str, Dict[str, Any]], 
                    pca_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Process remote sensing indices and optionally perform PCA.
    :param input_path: Path to the input raster file
    :param output_dir: Directory to save output files
    :param indices_config: Configuration for indices to calculate
    :param pca_config: Configuration for PCA (optional)
    """
    rsi = RemoteSensingIndices(input_path)
    
    # Calculate and save indices
    for index, params in indices_config.items():
        result = getattr(rsi, f"calculate_{index}")(*params['bands'])
        rsi.save_raster(result, os.path.join(output_dir, f"{index}.tif"))
    
    # Perform PCA if specified
    if pca_config:
        pca_data = rsi.perform_pca(pca_config['bands'], pca_config['n_components'])
        rsi.save_raster(pca_data, os.path.join(output_dir, 'pca.tif'), count=pca_config['n_components'])
        if pca_config.get('plot', False):
            rsi.plot_pca(pca_data, pca_config['n_components'])

# Example usage (can be removed or commented out in the actual module)
if __name__ == "__main__":
    # This is just for demonstration and can be removed
    input_path = r'F:\tif_features\temp\mosaic.tif'
    output_dir = r"F:\tif_features\temp\calc"
    indices_config = {
        "ndvi": {"bands": [8, 4]},
        "savi": {"bands": [8, 4]},
        "ndwi": {"bands": [8, 3]},
        "evi": {"bands": [8, 4, 2]},
        "lswi": {"bands": [8, 11]},
        "mndwi": {"bands": [3, 11]},
        "ndmi": {"bands": [8, 11]},
        "vari": {"bands": [4, 3, 2]}
    }
    pca_config = {
        "bands": [1,2,3,4,5,6,7,8,9,10,11,12],
        "n_components": 2,
        "plot": False
    }
    process_indices(input_path, output_dir, indices_config, pca_config)


