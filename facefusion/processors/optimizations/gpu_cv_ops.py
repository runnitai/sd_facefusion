"""
GPU-accelerated OpenCV operations with CPU fallback.
This module provides GPU-accelerated versions of common CV operations.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
from facefusion import logger

# Global state to track GPU availability
_gpu_available = None

def check_gpu_availability() -> bool:
    """Check if CUDA-enabled OpenCV operations are available."""
    global _gpu_available
    if _gpu_available is None:
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            _gpu_available = device_count > 0
            if _gpu_available:
                logger.debug(f"Found {device_count} CUDA-enabled devices for OpenCV operations", __name__)
            else:
                logger.debug("No CUDA-enabled devices found, falling back to CPU", __name__)
        except Exception as e:
            logger.debug(f"CUDA OpenCV not available: {e}", __name__)
            _gpu_available = False
    return _gpu_available

def resize_gpu_or_cpu(img: np.ndarray, size: Tuple[int, int], 
                     interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize image using GPU if available, otherwise CPU."""
    if check_gpu_availability():
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_resized = cv2.cuda.resize(gpu_img, size, interpolation=interpolation)
            result = gpu_resized.download()
            return result
        except Exception as e:
            logger.debug(f"GPU resize failed, falling back to CPU: {e}", __name__)
    
    return cv2.resize(img, size, interpolation=interpolation)

def warp_affine_gpu_or_cpu(img: np.ndarray, matrix: np.ndarray, size: Tuple[int, int],
                          flags: int = cv2.INTER_LINEAR, 
                          border_mode: int = cv2.BORDER_CONSTANT,
                          border_value: Union[int, Tuple[int, ...]] = 0) -> np.ndarray:
    """Warp affine using GPU if available, otherwise CPU."""
    if check_gpu_availability():
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_warped = cv2.cuda.warpAffine(gpu_img, matrix, size, flags=flags, 
                                           borderMode=border_mode, borderValue=border_value)
            result = gpu_warped.download()
            return result
        except Exception as e:
            logger.debug(f"GPU warpAffine failed, falling back to CPU: {e}", __name__)
    
    return cv2.warpAffine(img, matrix, size, flags=flags, borderMode=border_mode, borderValue=border_value)

def gaussian_blur_gpu_or_cpu(img: np.ndarray, ksize: Tuple[int, int], 
                            sigmaX: float, sigmaY: Optional[float] = None,
                            border_type: int = cv2.BORDER_DEFAULT) -> np.ndarray:
    """Apply Gaussian blur using GPU if available, otherwise CPU."""
    if check_gpu_availability():
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_blurred = cv2.cuda.GaussianBlur(gpu_img, ksize, sigmaX, sigmaY, border_type)
            result = gpu_blurred.download()
            return result
        except Exception as e:
            logger.debug(f"GPU GaussianBlur failed, falling back to CPU: {e}", __name__)
    
    return cv2.GaussianBlur(img, ksize, sigmaX, sigmaY, border_type)

def bilateral_filter_gpu_or_cpu(img: np.ndarray, d: int, sigmaColor: float, 
                               sigmaSpace: float, border_type: int = cv2.BORDER_DEFAULT) -> np.ndarray:
    """Apply bilateral filter using GPU if available, otherwise CPU."""
    if check_gpu_availability():
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_filtered = cv2.cuda.bilateralFilter(gpu_img, d, sigmaColor, sigmaSpace, border_type)
            result = gpu_filtered.download()
            return result
        except Exception as e:
            logger.debug(f"GPU bilateralFilter failed, falling back to CPU: {e}", __name__)
    
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace, border_type)

def cvt_color_gpu_or_cpu(img: np.ndarray, code: int) -> np.ndarray:
    """Convert color space using GPU if available, otherwise CPU."""
    if check_gpu_availability():
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_converted = cv2.cuda.cvtColor(gpu_img, code)
            result = gpu_converted.download()
            return result
        except Exception as e:
            logger.debug(f"GPU cvtColor failed, falling back to CPU: {e}", __name__)
    
    return cv2.cvtColor(img, code)

def morph_gpu_or_cpu(img: np.ndarray, op: int, kernel: np.ndarray,
                    iterations: int = 1, border_type: int = cv2.BORDER_CONSTANT,
                    border_value: Union[int, Tuple[int, ...]] = 0) -> np.ndarray:
    """Apply morphological operations using GPU if available, otherwise CPU."""
    if check_gpu_availability():
        try:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            gpu_kernel = cv2.cuda_GpuMat()
            gpu_kernel.upload(kernel)
            gpu_result = cv2.cuda.morphologyEx(gpu_img, op, gpu_kernel, iterations=iterations,
                                             borderType=border_type, borderValue=border_value)
            result = gpu_result.download()
            return result
        except Exception as e:
            logger.debug(f"GPU morphologyEx failed, falling back to CPU: {e}", __name__)
    
    return cv2.morphologyEx(img, op, kernel, iterations=iterations, 
                           borderType=border_type, borderValue=border_value)

class GPUMatManager:
    """Context manager for GPU matrix operations to reduce upload/download overhead."""
    
    def __init__(self, img: np.ndarray):
        self.img = img
        self.gpu_img = None
        self.gpu_available = check_gpu_availability()
    
    def __enter__(self):
        if self.gpu_available:
            try:
                self.gpu_img = cv2.cuda_GpuMat()
                self.gpu_img.upload(self.img)
                return self.gpu_img
            except Exception as e:
                logger.debug(f"Failed to upload to GPU: {e}", __name__)
                self.gpu_available = False
        return self.img
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # GPU memory is automatically cleaned up
        pass
    
    def download(self) -> np.ndarray:
        """Download result from GPU if available."""
        if self.gpu_available and self.gpu_img is not None:
            try:
                return self.gpu_img.download()
            except Exception as e:
                logger.debug(f"Failed to download from GPU: {e}", __name__)
        return self.img 