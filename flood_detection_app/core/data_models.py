"""
核心数据模型定义
Core Data Models Definition
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from enum import Enum


class FloodLevel(Enum):
    """车辆淹没部位等级枚举"""
    WHEEL_LEVEL = "wheel_level"      # 车轮顶部及以下
    WINDOW_LEVEL = "window_level"    # 车轮顶部至车窗下沿  
    ROOF_LEVEL = "roof_level"        # 车窗及以上


@dataclass
class BoundingBox:
    """边界框数据结构"""
    x1: float  # 左上角x坐标
    y1: float  # 左上角y坐标
    x2: float  # 右下角x坐标
    y2: float  # 右下角y坐标
    confidence: float  # 置信度
    
    def area(self) -> float:
        """计算边界框面积"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self) -> Tuple[float, float]:
        """获取边界框中心点"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class Detection:
    """检测结果数据结构"""
    bbox: BoundingBox  # 边界框
    class_id: int      # 类别ID
    class_name: str    # 类别名称


@dataclass
class VehicleResult:
    """车辆分析结果数据结构"""
    detection: Detection      # 检测结果
    flood_level: FloodLevel  # 淹没等级
    overlap_ratio: float     # 重叠比例
    vehicle_id: int         # 车辆编号


@dataclass
class Statistics:
    """统计信息数据结构"""
    total_vehicles: int           # 车辆总数
    wheel_level_count: int       # 车轮级淹没车辆数
    window_level_count: int      # 车窗级淹没车辆数
    roof_level_count: int        # 车顶级淹没车辆数
    water_coverage_percentage: float  # 积水覆盖率
    processing_time: float       # 处理时间（秒）
    
    # 保持向后兼容性的属性
    @property
    def light_flood_count(self) -> int:
        """轻度淹没车辆数（向后兼容）"""
        return self.wheel_level_count
    
    @property
    def moderate_flood_count(self) -> int:
        """中度淹没车辆数（向后兼容）"""
        return self.window_level_count
    
    @property
    def severe_flood_count(self) -> int:
        """重度淹没车辆数（向后兼容）"""
        return self.roof_level_count


@dataclass
class AnalysisResult:
    """完整分析结果数据结构"""
    vehicles: List[VehicleResult]  # 车辆分析结果列表
    water_mask: np.ndarray        # 水面掩码
    statistics: Statistics        # 统计信息
    original_image_shape: Tuple[int, int]  # 原始图像尺寸 (height, width)


@dataclass
class ModelConfig:
    """模型配置数据结构"""
    name: str                    # 模型名称
    file_path: str              # 模型文件路径
    input_size: Tuple[int, int] # 输入尺寸 (width, height)
    confidence_threshold: float  # 置信度阈值
    nms_threshold: float        # NMS阈值


@dataclass
class AppConfig:
    """应用配置数据结构"""
    vehicle_models: List[ModelConfig]     # 车辆检测模型列表
    water_models: List[ModelConfig]       # 水面分割模型列表
    flood_thresholds: Dict[str, float]    # 淹没等级阈值
    supported_image_formats: List[str]    # 支持的图像格式
    max_image_size: Tuple[int, int]      # 最大图像尺寸
    models_directory: str                # 模型文件目录
    performance_settings: Dict[str, Any] = None  # 性能优化设置