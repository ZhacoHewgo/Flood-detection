"""
淹没分析器
Flood Analyzer for calculating vehicle flood levels
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
from functools import lru_cache
import threading

# 尝试导入numba，如果不可用则使用普通函数
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 创建装饰器的占位符
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

from .data_models import (
    Detection, VehicleResult, Statistics, AnalysisResult, 
    BoundingBox, FloodLevel
)
from .exceptions import InferenceError, ValidationError
from .config import config_manager


class FloodAnalyzer:
    """淹没分析器类 - 性能优化版本"""
    
    def __init__(self):
        """初始化淹没分析器"""
        self.flood_thresholds = config_manager.config.flood_thresholds
        self.wheel_threshold = self.flood_thresholds["wheel_level"]    # 0.25 (25%)
        self.window_threshold = self.flood_thresholds["window_level"]  # 0.65 (65%)
        
        # 性能优化设置
        self._analysis_lock = threading.Lock()
        self._cache_enabled = True
    
    @staticmethod
    @jit(nopython=True, cache=True) if NUMBA_AVAILABLE else lambda func: func
    def _calculate_overlap_fast(x1: int, y1: int, x2: int, y2: int, water_mask: np.ndarray) -> float:
        """使用numba优化的重叠计算（如果可用）"""
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        overlap_count = 0
        total_pixels = (x2 - x1) * (y2 - y1)
        
        for y in prange(y1, y2):
            for x in prange(x1, x2):
                if water_mask[y, x] > 0:
                    overlap_count += 1
        
        return overlap_count / total_pixels if total_pixels > 0 else 0.0
    
    def calculate_overlap_ratio(self, bbox: BoundingBox, water_mask: np.ndarray) -> float:
        """
        计算车辆边界框与积水区域的重叠比例
        
        Args:
            bbox: 车辆边界框
            water_mask: 水面二值掩码 (0或1)
            
        Returns:
            float: 重叠比例 (0.0 - 1.0)
        """
        try:
            # 检查输入有效性
            if water_mask is None or water_mask.size == 0:
                return 0.0
            
            # 检查bbox坐标有效性
            if (np.isnan(bbox.x1) or np.isnan(bbox.y1) or 
                np.isnan(bbox.x2) or np.isnan(bbox.y2) or
                np.isinf(bbox.x1) or np.isinf(bbox.y1) or 
                np.isinf(bbox.x2) or np.isinf(bbox.y2)):
                print(f"警告: 边界框坐标包含NaN或Inf: {bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}")
                return 0.0
            
            # 确保坐标在图像范围内
            h, w = water_mask.shape
            x1 = max(0, min(int(bbox.x1), w - 1))
            y1 = max(0, min(int(bbox.y1), h - 1))
            x2 = max(0, min(int(bbox.x2), w - 1))
            y2 = max(0, min(int(bbox.y2), h - 1))
            
            # 确保边界框有效
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            # 计算边界框面积
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area <= 0:
                return 0.0
            
            # 使用优化的计算方法
            if NUMBA_AVAILABLE:
                try:
                    # 使用numba优化版本
                    overlap_ratio = self._calculate_overlap_fast(x1, y1, x2, y2, water_mask)
                except:
                    # 回退到numpy版本
                    bbox_mask = water_mask[y1:y2, x1:x2]
                    overlap_area = np.sum(bbox_mask > 0)
                    overlap_ratio = overlap_area / bbox_area
            else:
                # 直接使用numpy版本
                bbox_mask = water_mask[y1:y2, x1:x2]
                overlap_area = np.sum(bbox_mask > 0)
                overlap_ratio = overlap_area / bbox_area
            
            # 检查结果有效性
            if np.isnan(overlap_ratio) or np.isinf(overlap_ratio):
                print(f"警告: 计算出的重叠比例为NaN或Inf，返回0.0")
                return 0.0
            
            # 确保比例在有效范围内
            return max(0.0, min(1.0, float(overlap_ratio)))
            
        except Exception as e:
            print(f"警告: 重叠比例计算失败: {e}")
            return 0.0
    
    def determine_flood_level(self, overlap_ratio: float) -> FloodLevel:
        """
        根据重叠比例确定车辆淹没部位等级
        
        Args:
            overlap_ratio: 重叠比例 (0.0 - 1.0)
            
        Returns:
            FloodLevel: 淹没部位等级
        """
        if overlap_ratio < self.wheel_threshold:
            return FloodLevel.WHEEL_LEVEL      # 车轮顶部及以下
        elif overlap_ratio < self.window_threshold:
            return FloodLevel.WINDOW_LEVEL     # 车轮顶部至车窗下沿
        else:
            return FloodLevel.ROOF_LEVEL       # 车窗及以上
    
    def analyze_scene_batch(
        self, 
        vehicles: List[Detection], 
        water_mask: np.ndarray
    ) -> AnalysisResult:
        """
        批量分析场景的淹没情况 - 性能优化版本
        
        Args:
            vehicles: 车辆检测结果列表
            water_mask: 水面分割掩码
            
        Returns:
            AnalysisResult: 完整的分析结果
        """
        start_time = time.time()
        
        try:
            # 验证输入
            if not isinstance(vehicles, list):
                raise ValidationError("vehicles", str(type(vehicles)), "应该是列表类型")
            
            if not isinstance(water_mask, np.ndarray) or len(water_mask.shape) != 2:
                raise ValidationError("water_mask", str(water_mask.shape), "应该是2D数组")
            
            # 如果没有车辆，快速返回
            if not vehicles:
                statistics = Statistics(
                    total_vehicles=0,
                    wheel_level_count=0,
                    window_level_count=0,
                    roof_level_count=0,
                    water_coverage_percentage=self._calculate_water_coverage(water_mask),
                    processing_time=time.time() - start_time
                )
                
                return AnalysisResult(
                    vehicles=[],
                    water_mask=water_mask,
                    statistics=statistics,
                    original_image_shape=water_mask.shape
                )
            
            # 批量处理车辆分析
            vehicle_results = []
            
            # 预计算水面覆盖率（避免重复计算）
            water_coverage = self._calculate_water_coverage(water_mask)
            
            for i, vehicle in enumerate(vehicles):
                # 计算重叠比例
                overlap_ratio = self.calculate_overlap_ratio(vehicle.bbox, water_mask)
                
                # 🔥 修复：使用模型预测的淹没等级而不是基于重叠比例的判断
                # 首先尝试从检测结果中提取淹没等级
                flood_level = self._extract_flood_level_from_detection(vehicle)
                
                # 创建车辆结果
                vehicle_result = VehicleResult(
                    detection=vehicle,
                    flood_level=flood_level,
                    overlap_ratio=overlap_ratio,
                    vehicle_id=i + 1
                )
                
                vehicle_results.append(vehicle_result)
            
            # 计算统计信息
            processing_time = time.time() - start_time
            statistics = self.calculate_statistics(vehicle_results, water_mask.shape, water_mask, processing_time)
            
            # 创建分析结果
            analysis_result = AnalysisResult(
                vehicles=vehicle_results,
                water_mask=water_mask,
                statistics=statistics,
                original_image_shape=water_mask.shape
            )
            
            return analysis_result
            
        except Exception as e:
            if isinstance(e, (ValidationError, InferenceError)):
                raise
            raise InferenceError("场景分析", str(e))
    
    def analyze_scene(
        self, 
        vehicles: List[Detection], 
        water_mask: np.ndarray
    ) -> AnalysisResult:
        """
        分析整个场景的淹没情况
        
        Args:
            vehicles: 车辆检测结果列表
            water_mask: 水面分割掩码
            
        Returns:
            AnalysisResult: 完整的分析结果
        """
        # 兼容性方法，调用批量优化版本
        return self.analyze_scene_batch(vehicles, water_mask)
    
    def calculate_statistics(
        self, 
        vehicle_results: List[VehicleResult], 
        image_shape: Tuple[int, int],
        water_mask: np.ndarray,
        processing_time: float
    ) -> Statistics:
        """
        计算统计信息
        
        Args:
            vehicle_results: 车辆分析结果列表
            image_shape: 图像尺寸 (height, width)
            water_mask: 水面掩码
            processing_time: 处理时间
            
        Returns:
            Statistics: 统计信息
        """
        try:
            # 统计不同淹没部位等级的车辆数量
            wheel_count = sum(1 for v in vehicle_results if v.flood_level == FloodLevel.WHEEL_LEVEL)
            window_count = sum(1 for v in vehicle_results if v.flood_level == FloodLevel.WINDOW_LEVEL)
            roof_count = sum(1 for v in vehicle_results if v.flood_level == FloodLevel.ROOF_LEVEL)
            
            # 计算积水覆盖率
            total_pixels = image_shape[0] * image_shape[1]
            water_pixels = np.sum(water_mask > 0)
            water_coverage_percentage = (water_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
            
            statistics = Statistics(
                total_vehicles=len(vehicle_results),
                wheel_level_count=wheel_count,
                window_level_count=window_count,
                roof_level_count=roof_count,
                water_coverage_percentage=water_coverage_percentage,
                processing_time=processing_time
            )
            
            return statistics
            
        except Exception as e:
            print(f"警告: 统计计算失败: {e}")
            # 返回默认统计信息
            return Statistics(
                total_vehicles=0,
                wheel_level_count=0,
                window_level_count=0,
                roof_level_count=0,
                water_coverage_percentage=0.0,
                processing_time=processing_time
            )
    
    def get_flood_level_color(self, flood_level: FloodLevel) -> Tuple[int, int, int]:
        """
        获取淹没部位等级对应的颜色 (BGR格式)
        
        Args:
            flood_level: 淹没部位等级
            
        Returns:
            Tuple[int, int, int]: BGR颜色值
        """
        color_map = {
            FloodLevel.WHEEL_LEVEL: (0, 255, 0),    # 鲜绿色 - 轮胎级
            FloodLevel.WINDOW_LEVEL: (0, 165, 255), # 鲜橙色 - 车门级
            FloodLevel.ROOF_LEVEL: (0, 0, 255)      # 鲜红色 - 车窗级
        }
        return color_map.get(flood_level, (128, 128, 128))  # 默认灰色
    
    def get_flood_level_text(self, flood_level: FloodLevel) -> str:
        """
        获取淹没部位等级的简化描述
        
        Args:
            flood_level: 淹没部位等级
            
        Returns:
            str: 简化描述
        """
        text_map = {
            FloodLevel.WHEEL_LEVEL: "Tire Level",    # 轮胎级淹没 (车轮顶部及以下) - lt
            FloodLevel.WINDOW_LEVEL: "Door Level",   # 车门级淹没 (车轮顶部至车窗下沿) - cm  
            FloodLevel.ROOF_LEVEL: "Window Level"    # 车窗级淹没 (车窗及以上) - cc
        }
        return text_map.get(flood_level, "uk")
    
    def filter_vehicles_by_flood_level(
        self, 
        vehicle_results: List[VehicleResult], 
        flood_levels: List[FloodLevel]
    ) -> List[VehicleResult]:
        """
        根据淹没等级筛选车辆
        
        Args:
            vehicle_results: 车辆结果列表
            flood_levels: 要筛选的淹没等级列表
            
        Returns:
            List[VehicleResult]: 筛选后的车辆列表
        """
        return [v for v in vehicle_results if v.flood_level in flood_levels]
    
    def get_most_severe_vehicles(
        self, 
        vehicle_results: List[VehicleResult], 
        count: int = 5
    ) -> List[VehicleResult]:
        """
        获取淹没最严重的车辆
        
        Args:
            vehicle_results: 车辆结果列表
            count: 返回的车辆数量
            
        Returns:
            List[VehicleResult]: 按淹没程度排序的车辆列表
        """
        # 按重叠比例降序排序
        sorted_vehicles = sorted(
            vehicle_results, 
            key=lambda v: v.overlap_ratio, 
            reverse=True
        )
        
        return sorted_vehicles[:count]
    
    def calculate_area_statistics(
        self, 
        vehicle_results: List[VehicleResult]
    ) -> Dict[str, Any]:
        """
        计算面积相关统计信息
        
        Args:
            vehicle_results: 车辆结果列表
            
        Returns:
            Dict[str, Any]: 面积统计信息
        """
        if not vehicle_results:
            return {
                "total_vehicle_area": 0.0,
                "flooded_vehicle_area": 0.0,
                "average_vehicle_area": 0.0,
                "average_overlap_ratio": 0.0
            }
        
        total_area = 0.0
        flooded_area = 0.0
        total_overlap_ratio = 0.0
        
        for vehicle in vehicle_results:
            bbox_area = vehicle.detection.bbox.area()
            total_area += bbox_area
            flooded_area += bbox_area * vehicle.overlap_ratio
            total_overlap_ratio += vehicle.overlap_ratio
        
        return {
            "total_vehicle_area": total_area,
            "flooded_vehicle_area": flooded_area,
            "average_vehicle_area": total_area / len(vehicle_results),
            "average_overlap_ratio": total_overlap_ratio / len(vehicle_results)
        }
    
    def validate_analysis_result(self, result: AnalysisResult) -> bool:
        """
        验证分析结果的有效性
        
        Args:
            result: 分析结果
            
        Returns:
            bool: 是否有效
        """
        try:
            # 检查基本结构
            if not isinstance(result, AnalysisResult):
                return False
            
            # 检查车辆结果
            if not isinstance(result.vehicles, list):
                return False
            
            # 检查统计信息一致性
            stats = result.statistics
            expected_total = len(result.vehicles)
            actual_total = (stats.light_flood_count + 
                          stats.moderate_flood_count + 
                          stats.severe_flood_count)
            
            if expected_total != actual_total:
                print(f"警告: 统计数量不一致 - 期望: {expected_total}, 实际: {actual_total}")
                return False
            
            # 检查重叠比例范围
            for vehicle in result.vehicles:
                if not (0.0 <= vehicle.overlap_ratio <= 1.0):
                    print(f"警告: 重叠比例超出范围: {vehicle.overlap_ratio}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"验证分析结果时出错: {e}")
            return False
    
    def _calculate_water_coverage(self, water_mask: np.ndarray) -> float:
        """计算水面覆盖率"""
        total_pixels = water_mask.size
        water_pixels = np.sum(water_mask > 0)
        return (water_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    
    def _calculate_statistics_fast(
        self, 
        vehicle_results: List[VehicleResult], 
        image_shape: Tuple[int, int],
        water_coverage_percentage: float,
        processing_time: float
    ) -> Statistics:
        """
        快速计算统计信息
        
        Args:
            vehicle_results: 车辆分析结果列表
            image_shape: 图像尺寸 (height, width)
            water_coverage_percentage: 预计算的水面覆盖率
            processing_time: 处理时间
            
        Returns:
            Statistics: 统计信息
        """
        try:
            # 使用向量化操作统计不同淹没等级的车辆数量
            flood_levels = [v.flood_level for v in vehicle_results]
            
            light_count = flood_levels.count(FloodLevel.LIGHT)
            moderate_count = flood_levels.count(FloodLevel.MODERATE)
            severe_count = flood_levels.count(FloodLevel.SEVERE)
            
            statistics = Statistics(
                total_vehicles=len(vehicle_results),
                light_flood_count=light_count,
                moderate_flood_count=moderate_count,
                severe_flood_count=severe_count,
                water_coverage_percentage=water_coverage_percentage,
                processing_time=processing_time
            )
            
            return statistics
            
        except Exception as e:
            print(f"警告: 统计计算失败: {e}")
            # 返回默认统计信息
            return Statistics(
                total_vehicles=0,
                wheel_level_count=0,
                window_level_count=0,
                roof_level_count=0,
                water_coverage_percentage=0.0,
                processing_time=processing_time
            )
    

    def _calculate_water_coverage(self, water_mask: np.ndarray) -> float:
        """
        计算水面覆盖率
        
        Args:
            water_mask: 水面掩码
            
        Returns:
            float: 水面覆盖率百分比
        """
        try:
            total_pixels = water_mask.shape[0] * water_mask.shape[1]
            water_pixels = np.sum(water_mask > 0)
            coverage = (water_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
            return coverage
        except Exception as e:
            print(f"警告: 水面覆盖率计算失败: {e}")
            return 0.0
    
    def _extract_flood_level_from_detection(self, detection):
        """从检测结果中提取淹没等级"""
        
        # 根据检测的class_id映射到FloodLevel
        class_id = detection.class_id
        
        # 优先使用class_name进行映射
        if hasattr(detection, 'class_name'):
            class_name = detection.class_name.lower()
            if 'cc' in class_name or 'roof' in class_name:
                return FloodLevel.ROOF_LEVEL  # 车窗级
            elif 'cm' in class_name or 'window' in class_name:
                return FloodLevel.WINDOW_LEVEL  # 车门级
            elif 'lt' in class_name or 'wheel' in class_name:
                return FloodLevel.WHEEL_LEVEL  # 轮胎级
        
        # 如果没有明确的类别信息，根据class_id映射
        flood_level_mapping = {
            0: FloodLevel.WHEEL_LEVEL,   # 轮胎级（默认值）
            1: FloodLevel.WINDOW_LEVEL,  # 车门级
            2: FloodLevel.ROOF_LEVEL     # 车窗级
        }
        
        return flood_level_mapping.get(class_id, FloodLevel.WHEEL_LEVEL)