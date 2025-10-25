"""
配置管理模块
Configuration Management Module
"""

import os
from typing import Dict, List
from .data_models import AppConfig, ModelConfig
from .exceptions import ConfigurationError


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config = None
        self._load_default_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        try:
            # 默认模型配置
            vehicle_models = [
                ModelConfig(
                    name="YOLOv11 Car Detection",
                    file_path="models/yolov11_car_detection.pt",
                    input_size=(640, 640),
                    confidence_threshold=0.4,  # 降低置信度阈值
                    nms_threshold=0.4
                ),
                ModelConfig(
                    name="RT-DETR Car Detection", 
                    file_path="models/rtdetr_car_detection.pt",
                    input_size=(640, 640),
                    confidence_threshold=0.4,  # 降低置信度阈值
                    nms_threshold=0.4
                )
            ]
            
            water_models = [
                ModelConfig(
                    name="DeepLabV3 Water Segmentation",
                    file_path="models/deeplabv3_water.pt",
                    input_size=(240, 240),  # 修复：与训练时的尺寸一致
                    confidence_threshold=0.5,
                    nms_threshold=0.0  # 分割模型不需要NMS
                ),
                ModelConfig(
                    name="YOLOv11 Water Segmentation",
                    file_path="models/yolov11_seg_water.pt", 
                    input_size=(640, 640),
                    confidence_threshold=0.5,
                    nms_threshold=0.0
                )
            ]
            
            # 车辆淹没部位阈值（基于车辆高度的比例）
            flood_thresholds = {
                "wheel_level": 0.25,    # 车轮顶部及以下 <25%
                "window_level": 0.5    # 车轮顶部至车窗下沿 25-65%, 车窗及以上 >65%
            }
            
            # 支持的图像格式
            supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
            
            # 最大图像尺寸
            max_image_size = (2048, 2048)
            
            # 模型文件目录
            models_directory = "models"
            
            # 性能优化设置
            performance_settings = {
                "enable_model_warmup": True,
                "enable_gpu_optimization": True,
                "enable_half_precision": True,
                "enable_model_compilation": True,
                "max_cache_size": 50,
                "thread_pool_workers": 4,
                "enable_concurrent_processing": True,
                "memory_optimization_interval": 300,  # 秒
                "enable_request_caching": True
            }
            
            self._config = AppConfig(
                vehicle_models=vehicle_models,
                water_models=water_models,
                flood_thresholds=flood_thresholds,
                supported_image_formats=supported_formats,
                max_image_size=max_image_size,
                models_directory=models_directory,
                performance_settings=performance_settings
            )
            
        except Exception as e:
            raise ConfigurationError("默认配置加载", str(e))
    
    @property
    def config(self) -> AppConfig:
        """获取当前配置"""
        return self._config
    
    def get_vehicle_model_names(self) -> List[str]:
        """获取车辆检测模型名称列表"""
        return [model.name for model in self._config.vehicle_models]
    
    def get_water_model_names(self) -> List[str]:
        """获取水面分割模型名称列表"""
        return [model.name for model in self._config.water_models]
    
    def get_vehicle_model_config(self, name: str) -> ModelConfig:
        """根据名称获取车辆检测模型配置"""
        for model in self._config.vehicle_models:
            if model.name == name:
                return model
        raise ConfigurationError(f"车辆检测模型配置", f"未找到模型: {name}")
    
    def get_water_model_config(self, name: str) -> ModelConfig:
        """根据名称获取水面分割模型配置"""
        for model in self._config.water_models:
            if model.name == name:
                return model
        raise ConfigurationError(f"水面分割模型配置", f"未找到模型: {name}")
    
    def validate_image_format(self, file_path: str) -> bool:
        """验证图像格式是否支持"""
        _, ext = os.path.splitext(file_path.lower())
        return ext in self._config.supported_image_formats
    
    def get_flood_threshold(self, level: str) -> float:
        """获取淹没等级阈值"""
        if level not in self._config.flood_thresholds:
            raise ConfigurationError(f"淹没等级阈值", f"未知等级: {level}")
        return self._config.flood_thresholds[level]
    
    def validate_model_files(self) -> Dict[str, bool]:
        """验证模型文件是否存在"""
        results = {}
        
        # 检查车辆检测模型
        for model in self._config.vehicle_models:
            results[model.name] = os.path.exists(model.file_path)
        
        # 检查水面分割模型
        for model in self._config.water_models:
            results[model.name] = os.path.exists(model.file_path)
            
        return results
    
    def get_performance_setting(self, key: str, default=None):
        """获取性能设置"""
        if hasattr(self._config, 'performance_settings'):
            return self._config.performance_settings.get(key, default)
        return default
    
    def is_performance_feature_enabled(self, feature: str) -> bool:
        """检查性能功能是否启用"""
        return self.get_performance_setting(feature, False)


# 全局配置管理器实例
config_manager = ConfigManager()