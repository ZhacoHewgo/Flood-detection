"""
图像处理器
Image Processor for loading, preprocessing and saving images
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from typing import Tuple, Optional, Union
from pathlib import Path
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
import time

from .exceptions import ImageProcessingError, FileOperationError, ValidationError
from .config import config_manager


class ImageProcessor:
    """图像处理器类 - 性能优化版本"""
    
    # 类级别的线程池，用于并行处理
    _thread_pool = ThreadPoolExecutor(max_workers=2)
    _processing_lock = threading.Lock()
    
    @staticmethod
    @lru_cache(maxsize=16)
    def _get_image_info_cached(file_path: str, file_size: int) -> dict:
        """缓存图像信息以避免重复读取"""
        return {
            'path': file_path,
            'size': file_size,
            'cached_at': time.time()
        }
    
    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """
        加载图像文件
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            np.ndarray: BGR格式的图像数组
            
        Raises:
            ImageProcessingError: 图像加载失败
            ValidationError: 文件格式不支持
        """
        if not os.path.exists(file_path):
            raise FileOperationError("读取", file_path, "文件不存在")
        
        # 验证文件格式
        if not config_manager.validate_image_format(file_path):
            raise ValidationError("图像格式", file_path, "不支持的图像格式")
        
        try:
            # 优化的图像加载策略
            # 首先尝试使用OpenCV（通常更快）
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            
            if image is None:
                # 回退到PIL（支持更多格式）
                try:
                    with Image.open(file_path) as pil_image:
                        # 转换为RGB然后转为BGR（OpenCV格式）
                        if pil_image.mode != 'RGB':
                            pil_image = pil_image.convert('RGB')
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ImageProcessingError("图像加载", file_path, f"PIL加载失败: {str(e)}")
            
            # 优化的尺寸检查和缩放
            max_size = config_manager.config.max_image_size
            h, w = image.shape[:2]
            
            if h > max_size[1] or w > max_size[0]:
                print(f"警告: 图像尺寸 {(h, w)} 超过最大限制 {max_size}，将进行缩放")
                # 使用更快的缩放方法
                image = ImageProcessor.resize_with_aspect_ratio_fast(image, max_size)
            
            return image
            
        except Exception as e:
            if isinstance(e, (ImageProcessingError, ValidationError, FileOperationError)):
                raise
            raise ImageProcessingError("图像加载", file_path, str(e))
    
    @staticmethod
    def save_image(image: np.ndarray, file_path: str, quality: int = 95) -> bool:
        """
        保存图像到文件
        
        Args:
            image: BGR格式的图像数组
            file_path: 保存路径
            quality: JPEG质量 (1-100)
            
        Returns:
            bool: 保存是否成功
            
        Raises:
            ImageProcessingError: 图像保存失败
        """
        try:
            # 创建目录（如果不存在）
            dir_path = os.path.dirname(file_path)
            if dir_path:  # 只有当目录路径不为空时才创建
                os.makedirs(dir_path, exist_ok=True)
            
            # 根据文件扩展名设置保存参数
            _, ext = os.path.splitext(file_path.lower())
            
            if ext in ['.jpg', '.jpeg']:
                # JPEG格式
                success = cv2.imwrite(file_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif ext == '.png':
                # PNG格式
                success = cv2.imwrite(file_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                # 其他格式
                success = cv2.imwrite(file_path, image)
            
            if not success:
                raise ImageProcessingError("图像保存", file_path, "OpenCV保存失败")
            
            return True
            
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            raise ImageProcessingError("图像保存", file_path, str(e))
    
    @staticmethod
    def resize_with_aspect_ratio_fast(
        image: np.ndarray, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        快速保持宽高比的图像缩放
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            
        Returns:
            np.ndarray: 缩放后的图像
        """
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # 计算缩放比例
            scale = min(target_w / w, target_h / h)
            
            # 如果不需要缩放，直接返回
            if scale >= 1.0:
                return image
            
            # 计算新尺寸
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 使用更快的插值方法进行缩放
            if scale < 0.5:
                # 大幅缩放时使用INTER_AREA
                interpolation = cv2.INTER_AREA
            else:
                # 小幅缩放时使用INTER_LINEAR
                interpolation = cv2.INTER_LINEAR
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
            return resized
            
        except Exception as e:
            raise ImageProcessingError("图像缩放", None, str(e))
    
    @staticmethod
    def resize_with_aspect_ratio(
        image: np.ndarray, 
        target_size: Tuple[int, int], 
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        保持宽高比的图像缩放
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            interpolation: 插值方法
            
        Returns:
            np.ndarray: 缩放后的图像
        """
        # 兼容性方法，调用快速版本
        return ImageProcessor.resize_with_aspect_ratio_fast(image, target_size)
    
    @staticmethod
    def preprocess_for_model_fast(
        image: np.ndarray, 
        model_type: str, 
        input_size: Tuple[int, int] = (640, 640)
    ) -> torch.Tensor:
        """
        为模型推理快速预处理图像
        
        Args:
            image: BGR格式的输入图像
            model_type: 模型类型 ('yolo', 'deeplabv3', 'rtdetr')
            input_size: 模型输入尺寸 (width, height)
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        try:
            # 转换为RGB格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if model_type.lower() in ['yolo', 'yolov11', 'rtdetr']:
                # 🔥 YOLO/RT-DETR预处理 - 使用与Ultralytics训练时一致的方法
                # 1. 计算缩放比例（保持宽高比）
                h, w = rgb_image.shape[:2]
                target_w, target_h = input_size
                scale = min(target_w / w, target_h / h)
                
                # 2. 缩放图像
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # 3. 创建填充图像（使用114作为填充值，与Ultralytics一致）
                padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                # 4. 归一化
                normalized = padded.astype(np.float32) / 255.0
                
            elif model_type.lower() in ['deeplabv3', 'deeplab']:
                # DeepLabV3预处理 - 直接缩放（与训练时一致）
                resized = cv2.resize(rgb_image, input_size, interpolation=cv2.INTER_LINEAR)
                
                # 只做简单归一化（与训练时的Albumentations一致）
                normalized = resized.astype(np.float32) / 255.0
                
            else:
                # 通用预处理 - 直接缩放
                resized = cv2.resize(rgb_image, input_size, interpolation=cv2.INTER_LINEAR)
                normalized = resized.astype(np.float32) / 255.0
            
            # 转换为张量并调整维度 (H, W, C) -> (1, C, H, W)
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            raise ImageProcessingError("模型预处理", None, str(e))
    
    @staticmethod
    def preprocess_for_model(
        image: np.ndarray, 
        model_type: str, 
        input_size: Tuple[int, int] = (640, 640)
    ) -> torch.Tensor:
        """
        为模型推理预处理图像
        
        Args:
            image: BGR格式的输入图像
            model_type: 模型类型 ('yolo', 'deeplabv3', 'rtdetr')
            input_size: 模型输入尺寸 (width, height)
            
        Returns:
            torch.Tensor: 预处理后的张量
        """
        # 兼容性方法，调用快速版本
        return ImageProcessor.preprocess_for_model_fast(image, model_type, input_size)
    
    @staticmethod
    def postprocess_mask(
        mask: np.ndarray, 
        original_size: Tuple[int, int],
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        后处理分割掩码
        
        Args:
            mask: 模型输出的掩码
            original_size: 原始图像尺寸 (width, height)
            threshold: 二值化阈值
            
        Returns:
            np.ndarray: 处理后的二值掩码
        """
        try:
            # 确保掩码是2D的
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # 二值化
            binary_mask = (mask > threshold).astype(np.uint8)
            
            # 调整到原始尺寸
            if binary_mask.shape != original_size[::-1]:  # OpenCV使用 (height, width)
                binary_mask = cv2.resize(
                    binary_mask, 
                    original_size, 
                    interpolation=cv2.INTER_NEAREST
                )
            
            return binary_mask
            
        except Exception as e:
            raise ImageProcessingError("掩码后处理", None, str(e))
    
    @staticmethod
    def create_overlay(
        image: np.ndarray, 
        mask: np.ndarray, 
        color: Tuple[int, int, int] = (0, 255, 255),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        创建掩码叠加图像
        
        Args:
            image: 原始图像 (BGR)
            mask: 二值掩码
            color: 叠加颜色 (BGR)
            alpha: 透明度
            
        Returns:
            np.ndarray: 叠加后的图像
        """
        try:
            overlay = image.copy()
            
            # 确保掩码是二值的
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # 创建彩色掩码
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            
            # 叠加
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
            
            return overlay
            
        except Exception as e:
            raise ImageProcessingError("掩码叠加", None, str(e))
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        验证图像数据的有效性
        
        Args:
            image: 图像数组
            
        Returns:
            bool: 是否有效
        """
        try:
            if image is None:
                return False
            
            if not isinstance(image, np.ndarray):
                return False
            
            if len(image.shape) not in [2, 3]:
                return False
            
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False
            
            if image.size == 0:
                return False
            
            return True
            
        except:
            return False
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """
        获取图像信息
        
        Args:
            image: 图像数组
            
        Returns:
            dict: 图像信息
        """
        try:
            if not ImageProcessor.validate_image(image):
                return {"valid": False}
            
            info = {
                "valid": True,
                "shape": image.shape,
                "dtype": str(image.dtype),
                "size": image.size,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "width": image.shape[1],
                "height": image.shape[0]
            }
            
            return info
            
        except Exception as e:
            return {"valid": False, "error": str(e)}