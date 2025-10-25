"""
可视化引擎
Visualization Engine for creating result images with annotations
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor

from .data_models import VehicleResult, AnalysisResult, FloodLevel
from .exceptions import ImageProcessingError
from .flood_analyzer import FloodAnalyzer


class VisualizationEngine:
    """可视化引擎类 - 性能优化版本"""
    
    def __init__(self):
        """初始化可视化引擎"""
        # 淹没部位等级颜色映射 (BGR格式) - 使用更鲜艳的颜色提高可见性
        self.colors = {
            FloodLevel.WHEEL_LEVEL: (0, 255, 0),    # 鲜绿色 - 轮胎级
            FloodLevel.WINDOW_LEVEL: (0, 165, 255), # 鲜橙色 - 车门级  
            FloodLevel.ROOF_LEVEL: (0, 0, 255)      # 鲜红色 - 车窗级
        }
        
        # 水面叠加颜色 (BGR格式) - 更明显的蓝色
        self.water_color = (255, 150, 50)  # 更鲜艳的青蓝色，提高可见度
        
        # 字体设置 - 增大以提高可见性
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8  # 增大字体
        self.font_thickness = 3  # 增加字体粗细
        
        # 边界框设置 - 增大以提高可见性
        self.bbox_thickness = 6  # 进一步增加边界框粗细
        
        # 创建FloodAnalyzer实例用于获取颜色和文本
        self.flood_analyzer = FloodAnalyzer()
        
        # 性能优化设置
        self._rendering_lock = threading.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # 预计算常用的文本尺寸
        self._text_cache = {}
    
    def draw_bounding_boxes(
        self, 
        image: np.ndarray, 
        vehicles: List[VehicleResult]
    ) -> np.ndarray:
        """
        在图像上绘制Vehicle边界框
        
        Args:
            image: 原始图像 (BGR格式)
            vehicles: Vehicle分析结果列表
            
        Returns:
            np.ndarray: 绘制了边界框的图像
        """
        try:
            result_image = image.copy()
            
            for vehicle in vehicles:
                bbox = vehicle.detection.bbox
                color = self.flood_analyzer.get_flood_level_color(vehicle.flood_level)
                
                # 绘制增强的边界框（双重边框效果）
                pt1 = (int(bbox.x1), int(bbox.y1))
                pt2 = (int(bbox.x2), int(bbox.y2))
                
                # 绘制外层黑色边框（增强对比度）
                cv2.rectangle(
                    result_image, 
                    pt1, 
                    pt2, 
                    (0, 0, 0),  # 黑色外框
                    self.bbox_thickness + 3
                )
                
                # 绘制内层彩色边框
                cv2.rectangle(
                    result_image, 
                    pt1, 
                    pt2, 
                    color, 
                    self.bbox_thickness
                )
                
                # 绘制置信度条
                self._draw_confidence_bar(
                    result_image, 
                    bbox, 
                    vehicle.detection.bbox.confidence,
                    color
                )
            
            return result_image
            
        except Exception as e:
            raise ImageProcessingError("边界框绘制", None, str(e))
    
    @lru_cache(maxsize=16)
    def _get_text_size_cached(self, text: str, font_scale: float) -> Tuple[int, int]:
        """缓存文本尺寸计算"""
        (text_width, text_height), _ = cv2.getTextSize(
            text, 
            self.font, 
            font_scale, 
            self.font_thickness
        )
        return text_width, text_height
    
    def overlay_water_mask_fast(
        self, 
        image: np.ndarray, 
        water_mask: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        快速在图像上叠加水面掩码
        
        Args:
            image: 原始图像 (BGR格式)
            water_mask: 水面二值掩码
            alpha: 透明度 (0.0-1.0)
            
        Returns:
            np.ndarray: 叠加了水面掩码的图像
        """
        try:
            # 使用就地操作减少内存分配
            result_image = image.copy()
            
            # 确保掩码尺寸匹配
            if water_mask.shape[:2] != image.shape[:2]:
                water_mask = cv2.resize(
                    water_mask, 
                    (image.shape[1], image.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            
            # 优化的掩码叠加 - 只处理有水的区域，增强可见度
            water_indices = np.where(water_mask > 0)
            if len(water_indices[0]) > 0:
                # 增强水面区域的对比度和饱和度
                water_overlay = np.array(self.water_color, dtype=np.float32)
                
                # 对水面区域进行更明显的着色
                result_image[water_indices] = (
                    result_image[water_indices] * (1 - alpha) + 
                    water_overlay * alpha
                ).astype(np.uint8)
                
                # 添加水面边缘高亮效果
                self._add_water_edge_highlight(result_image, water_mask)
            
            return result_image
            
        except Exception as e:
            raise ImageProcessingError("水面掩码叠加", None, str(e))
    
    def overlay_water_mask(
        self, 
        image: np.ndarray, 
        water_mask: np.ndarray, 
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        在图像上叠加水面掩码
        
        Args:
            image: 原始图像 (BGR格式)
            water_mask: 水面二值掩码
            alpha: 透明度 (0.0-1.0)
            
        Returns:
            np.ndarray: 叠加了水面掩码的图像
        """
        # 兼容性方法，调用快速版本
        return self.overlay_water_mask_fast(image, water_mask, alpha)
    
    def add_labels(
        self, 
        image: np.ndarray, 
        vehicles: List[VehicleResult]
    ) -> np.ndarray:
        """
        在图像上添加Vehicle标签
        
        Args:
            image: 图像
            vehicles: Vehicle分析结果列表
            
        Returns:
            np.ndarray: 添加了标签的图像
        """
        try:
            result_image = image.copy()
            
            for vehicle in vehicles:
                bbox = vehicle.detection.bbox
                color = self.flood_analyzer.get_flood_level_color(vehicle.flood_level)
                
                # 🔥 统一标签显示逻辑 - 只使用简洁的cc/cm/lt标签
                if vehicle.flood_level.value == 'WHEEL_LEVEL':
                    level_code = 'lt'  # 轮胎级 (车轮顶部及以下)
                elif vehicle.flood_level.value == 'WINDOW_LEVEL':
                    level_code = 'cm'  # 车门级 (车轮顶部至车窗下沿)
                elif vehicle.flood_level.value == 'ROOF_LEVEL':
                    level_code = 'cc'  # 车窗级 (车窗及以上)
                else:
                    level_code = 'uk'  # Unknown
                
                # 构建简洁的标签
                vehicle_id_str = str(vehicle.vehicle_id)
                
                # 🔥 修复单目标检测时的百分比显示
                if hasattr(vehicle, 'overlap_ratio') and not (np.isnan(vehicle.overlap_ratio) or np.isinf(vehicle.overlap_ratio)) and vehicle.overlap_ratio > 0:
                    # 有水面重叠数据时显示百分比
                    overlap_percent = int(max(0, min(100, vehicle.overlap_ratio * 100)))
                    detail_text = f"{overlap_percent}%"
                else:
                    # 单目标检测时显示置信度
                    confidence = int(vehicle.detection.bbox.confidence * 100) if hasattr(vehicle.detection.bbox, 'confidence') else 0
                    detail_text = f"Conf:{confidence}%"
                
                # 简化标签格式：Vehicle ID + 淹没等级
                label_text = f"V{vehicle_id_str}:{level_code}"
                
                # 计算标签位置
                label_x = int(bbox.x1)
                label_y = int(bbox.y1) - 10
                
                # 确保标签在图像范围内
                if label_y < 30:
                    label_y = int(bbox.y2) + 30
                
                # 绘制增强的标签背景
                text_size = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)[0]
                detail_size = cv2.getTextSize(detail_text, self.font, self.font_scale * 0.8, max(1, self.font_thickness - 1))[0]
                
                # 计算背景矩形大小（包含两行文字）
                bg_width = max(text_size[0], detail_size[0]) + 20
                bg_height = text_size[1] + detail_size[1] + 25
                
                # 绘制黑色外边框
                cv2.rectangle(
                    result_image,
                    (label_x - 2, label_y - bg_height - 2),
                    (label_x + bg_width + 2, label_y + 7),
                    (0, 0, 0),  # 黑色边框
                    -1
                )
                
                # 绘制彩色背景
                cv2.rectangle(
                    result_image,
                    (label_x, label_y - bg_height),
                    (label_x + bg_width, label_y + 5),
                    color,
                    -1  # 填充矩形
                )
                
                # 绘制文字阴影效果（增强可读性）
                shadow_offset = 2
                
                # 主标签阴影
                cv2.putText(
                    result_image,
                    label_text,
                    (label_x + 5 + shadow_offset, label_y - 5 + shadow_offset),
                    self.font,
                    self.font_scale,
                    (0, 0, 0),  # 黑色阴影
                    self.font_thickness + 1
                )
                
                # 主标签文字
                cv2.putText(
                    result_image,
                    label_text,
                    (label_x + 5, label_y - 5),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),  # 白色文字
                    self.font_thickness
                )
                
                # 详细信息阴影
                cv2.putText(
                    result_image,
                    detail_text,
                    (label_x + 5 + shadow_offset, label_y + 15 + shadow_offset),
                    self.font,
                    self.font_scale * 0.8,
                    (0, 0, 0),  # 黑色阴影
                    max(1, self.font_thickness)
                )
                
                # 详细信息文字
                cv2.putText(
                    result_image,
                    detail_text,
                    (label_x + 5, label_y + 15),
                    self.font,
                    self.font_scale * 0.8,
                    (255, 255, 255),  # 白色文字
                    max(1, self.font_thickness - 1)
                )
            
            return result_image
            
        except Exception as e:
            raise ImageProcessingError("标签添加", None, str(e))
    
    def create_result_image_fast(
        self, 
        original_image: np.ndarray, 
        analysis_result: AnalysisResult
    ) -> np.ndarray:
        """
        快速创建完整的结果图像
        
        Args:
            original_image: 原始图像
            analysis_result: 分析结果
            
        Returns:
            np.ndarray: 完整的结果图像
        """
        try:
            # 使用线程锁确保渲染安全
            with self._rendering_lock:
                # 1. 叠加水面掩码 - 使用更高的透明度使水面更明显
                result_image = self.overlay_water_mask_fast(
                    original_image, 
                    analysis_result.water_mask, 
                    alpha=0.6
                )
                
                # 2. 批量绘制边界框和标签（减少图像复制）
                result_image = self._draw_all_annotations_fast(
                    result_image, 
                    analysis_result.vehicles
                )
                
                # 3. 添加统计信息
                result_image = self.add_statistics_overlay_fast(
                    result_image, 
                    analysis_result.statistics
                )
                
                return result_image
            
        except Exception as e:
            raise ImageProcessingError("结果图像创建", None, str(e))
    
    def create_result_image(
        self, 
        original_image: np.ndarray, 
        analysis_result: AnalysisResult
    ) -> np.ndarray:
        """
        创建完整的结果图像
        
        Args:
            original_image: 原始图像
            analysis_result: 分析结果
            
        Returns:
            np.ndarray: 完整的结果图像
        """
        # 兼容性方法，调用快速版本
        return self.create_result_image_fast(original_image, analysis_result)
    
    def add_statistics_overlay(
        self, 
        image: np.ndarray, 
        statistics: Any
    ) -> np.ndarray:
        """
        在图像上添加统计信息叠加
        
        Args:
            image: 图像
            statistics: 统计信息
            
        Returns:
            np.ndarray: 添加了统计信息的图像
        """
        try:
            result_image = image.copy()
            h, w = image.shape[:2]
            
            # 统计信息文本
            stats_lines = [
                f"Total Vehicles: {statistics.total_vehicles}",
                f"Tire: {statistics.light_flood_count}  Door: {statistics.moderate_flood_count}  Window: {statistics.severe_flood_count}",
                f"Water Coverage: {statistics.water_coverage_percentage:.1f}%",
                f"Processing Time: {statistics.processing_time:.2f}s"
            ]
            
            # 计算统计面板尺寸
            panel_height = len(stats_lines) * 25 + 20
            panel_width = 350
            
            # 绘制半透明背景
            overlay = result_image.copy()
            cv2.rectangle(
                overlay,
                (10, h - panel_height - 10),
                (panel_width, h - 10),
                (0, 0, 0),  # 黑色背景
                -1
            )
            
            # 混合背景
            cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
            
            # 绘制统计文本
            for i, line in enumerate(stats_lines):
                y_pos = h - panel_height + 25 + i * 25
                cv2.putText(
                    result_image,
                    line,
                    (20, y_pos),
                    self.font,
                    0.5,
                    (255, 255, 255),  # 白色文字
                    1
                )
            
            return result_image
            
        except Exception as e:
            raise ImageProcessingError("统计信息叠加", None, str(e))
    
    def create_legend(
        self, 
        width: int = 200, 
        height: int = 150
    ) -> np.ndarray:
        """
        创建图例
        
        Args:
            width: 图例宽度
            height: 图例高度
            
        Returns:
            np.ndarray: 图例图像
        """
        try:
            legend = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
            
            # 绘制边框
            cv2.rectangle(legend, (0, 0), (width-1, height-1), (0, 0, 0), 2)
            
            # 标题
            cv2.putText(
                legend,
                "Flood Levels",
                (10, 25),
                self.font,
                0.6,
                (0, 0, 0),
                2
            )
            
            # 图例项
            legend_items = [
                (FloodLevel.LIGHT, "Light (<30%)"),
                (FloodLevel.MODERATE, "Moderate (30-60%)"),
                (FloodLevel.SEVERE, "Severe (>60%)")
            ]
            
            for i, (level, text) in enumerate(legend_items):
                y_pos = 50 + i * 30
                color = self.flood_analyzer.get_flood_level_color(level)
                
                # 绘制颜色块
                cv2.rectangle(
                    legend,
                    (10, y_pos - 10),
                    (30, y_pos + 5),
                    color,
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    legend,
                    text,
                    (40, y_pos),
                    self.font,
                    0.4,
                    (0, 0, 0),
                    1
                )
            
            return legend
            
        except Exception as e:
            raise ImageProcessingError("图例创建", None, str(e))
    
    def _draw_confidence_bar(
        self, 
        image: np.ndarray, 
        bbox: Any, 
        confidence: float, 
        color: Tuple[int, int, int]
    ):
        """绘制置信度条"""
        bar_width = int((bbox.x2 - bbox.x1) * 0.8)
        bar_height = 6
        
        bar_x = int(bbox.x1 + (bbox.x2 - bbox.x1 - bar_width) / 2)
        bar_y = int(bbox.y2) + 5
        
        # 背景条
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (128, 128, 128),
            -1
        )
        
        # 置信度条
        conf_width = int(bar_width * confidence)
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + conf_width, bar_y + bar_height),
            color,
            -1
        )
    
    def _draw_label_background(
        self, 
        image: np.ndarray, 
        text: str, 
        position: Tuple[int, int], 
        color: Tuple[int, int, int]
    ):
        """绘制标签背景"""
        (text_width, text_height), _ = cv2.getTextSize(
            text, 
            self.font, 
            self.font_scale, 
            self.font_thickness
        )
        
        x, y = position
        
        # 绘制背景矩形
        cv2.rectangle(
            image,
            (x, y - text_height - 10),
            (x + text_width + 10, y + 20),
            color,
            -1
        )
        
        # 绘制边框
        cv2.rectangle(
            image,
            (x, y - text_height - 10),
            (x + text_width + 10, y + 20),
            (0, 0, 0),
            1
        )
    
    def create_comparison_image(
        self, 
        original_image: np.ndarray, 
        result_image: np.ndarray
    ) -> np.ndarray:
        """
        创建对比图像（原图和结果图并排显示）
        
        Args:
            original_image: 原始图像
            result_image: 结果图像
            
        Returns:
            np.ndarray: 对比图像
        """
        try:
            # 确保两个图像尺寸相同
            if original_image.shape != result_image.shape:
                result_image = cv2.resize(
                    result_image, 
                    (original_image.shape[1], original_image.shape[0])
                )
            
            # 水平拼接
            comparison = np.hstack([original_image, result_image])
            
            # 添加分割线
            h, w = comparison.shape[:2]
            mid_x = w // 2
            cv2.line(comparison, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)
            
            # 添加标题
            cv2.putText(
                comparison,
                "Original",
                (20, 30),
                self.font,
                0.8,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                comparison,
                "Analysis Result",
                (mid_x + 20, 30),
                self.font,
                0.8,
                (255, 255, 255),
                2
            )
            
            return comparison
            
        except Exception as e:
            raise ImageProcessingError("对比图像创建", None, str(e))
    
    def _draw_all_annotations_fast(
        self, 
        image: np.ndarray, 
        vehicles: List[VehicleResult]
    ) -> np.ndarray:
        """
        快速批量绘制所有标注（边界框和标签）
        
        Args:
            image: 图像
            vehicles: Vehicle分析结果列表
            
        Returns:
            np.ndarray: 绘制了所有标注的图像
        """
        try:
            # 在同一个图像上进行所有绘制操作，减少内存分配
            for vehicle in vehicles:
                bbox = vehicle.detection.bbox
                color = self.flood_analyzer.get_flood_level_color(vehicle.flood_level)
                
                # 绘制边界框
                pt1 = (int(bbox.x1), int(bbox.y1))
                pt2 = (int(bbox.x2), int(bbox.y2))
                
                cv2.rectangle(image, pt1, pt2, color, self.bbox_thickness)
                
                # 绘制置信度条
                self._draw_confidence_bar_fast(image, bbox, vehicle.detection.bbox.confidence, color)
                
                # 绘制标签
                self._draw_label_fast(image, vehicle, color)
            
            return image
            
        except Exception as e:
            raise ImageProcessingError("批量标注绘制", None, str(e))
    
    def _draw_confidence_bar_fast(
        self, 
        image: np.ndarray, 
        bbox: Any, 
        confidence: float, 
        color: Tuple[int, int, int]
    ):
        """快速绘制置信度条"""
        bar_width = int((bbox.x2 - bbox.x1) * 0.8)
        bar_height = 6
        
        bar_x = int(bbox.x1 + (bbox.x2 - bbox.x1 - bar_width) / 2)
        bar_y = int(bbox.y2) + 5
        
        # 背景条
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (128, 128, 128), -1)
        
        # 置信度条
        conf_width = int(bar_width * confidence)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), color, -1)
    
    def _draw_label_fast(
        self, 
        image: np.ndarray, 
        vehicle: VehicleResult, 
        color: Tuple[int, int, int]
    ):
        """快速绘制Vehicle标签"""
        bbox = vehicle.detection.bbox
        
        # 创建标签文本（显示淹没等级）
        flood_text = self.flood_analyzer.get_flood_level_text(vehicle.flood_level)
        
        # 从class_name中提取淹没等级代码
        class_name = vehicle.detection.class_name
        if 'vehicle_' in class_name:
            flood_code = class_name.replace('vehicle_', '').upper()
        else:
            flood_code = 'UNK'
        
        label_text = f"{flood_code} {vehicle.vehicle_id}: {flood_text}"
        
        # 🔥 显示检测置信度而不是重叠比例
        if hasattr(vehicle.detection.bbox, 'confidence'):
            confidence = vehicle.detection.bbox.confidence
            detail_text = f"{confidence:.1f}"
        else:
            detail_text = "0.0"
        
        # 计算标签位置
        label_x = int(bbox.x1)
        label_y = int(bbox.y1) - 10
        
        # 确保标签在图像范围内
        if label_y < 30:
            label_y = int(bbox.y2) + 30
        
        # 使用缓存的文本尺寸
        text_width, text_height = self._get_text_size_cached(label_text, self.font_scale)
        
        # 绘制标签背景
        cv2.rectangle(
            image,
            (label_x, label_y - text_height - 10),
            (label_x + text_width + 10, label_y + 20),
            color,
            -1
        )
        
        # 绘制边框
        cv2.rectangle(
            image,
            (label_x, label_y - text_height - 10),
            (label_x + text_width + 10, label_y + 20),
            (0, 0, 0),
            1
        )
        
        # 绘制主标签
        cv2.putText(
            image,
            label_text,
            (label_x + 5, label_y - 5),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness
        )
        
        # 绘制详细信息
        cv2.putText(
            image,
            detail_text,
            (label_x + 5, label_y + 15),
            self.font,
            self.font_scale * 0.8,
            (255, 255, 255),
            max(1, self.font_thickness - 1)
        )
    
    def _add_water_edge_highlight(self, image: np.ndarray, water_mask: np.ndarray):
        """
        为水面区域添加边缘高亮效果，使水面检测更明显
        
        Args:
            image: 结果图像
            water_mask: 水面掩码
        """
        try:
            # 计算水面边缘
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(water_mask, cv2.MORPH_GRADIENT, kernel)
            
            # 在边缘位置添加亮蓝色高亮
            edge_indices = np.where(edges > 0)
            if len(edge_indices[0]) > 0:
                # 使用更亮的蓝色作为边缘高亮
                highlight_color = (255, 255, 100)  # 亮青色
                image[edge_indices] = highlight_color
                
        except Exception as e:
            print(f"水面边缘高亮添加失败: {e}")
    
    def add_statistics_overlay_fast(
        self, 
        image: np.ndarray, 
        statistics: Any
    ) -> np.ndarray:
        """
        快速在图像上添加统计信息叠加
        
        Args:
            image: 图像
            statistics: 统计信息
            
        Returns:
            np.ndarray: 添加了统计信息的图像
        """
        try:
            h, w = image.shape[:2]
            
            # 统计信息文本
            stats_lines = [
                f"Total Vehicles: {statistics.total_vehicles}",
                f"Tire: {statistics.light_flood_count}  Door: {statistics.moderate_flood_count}  Window: {statistics.severe_flood_count}",
                f"Water Coverage: {statistics.water_coverage_percentage:.1f}%",
                f"Processing Time: {statistics.processing_time:.2f}s"
            ]
            
            # 计算统计面板尺寸
            panel_height = len(stats_lines) * 25 + 20
            panel_width = 350
            
            # 创建半透明背景 - 直接在原图上操作
            overlay_region = image[h - panel_height - 10:h - 10, 10:panel_width].copy()
            overlay_region = (overlay_region * 0.3).astype(np.uint8)
            image[h - panel_height - 10:h - 10, 10:panel_width] = overlay_region
            
            # 绘制统计文本
            for i, line in enumerate(stats_lines):
                y_pos = h - panel_height + 25 + i * 25
                cv2.putText(
                    image,
                    line,
                    (20, y_pos),
                    self.font,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            return image
            
        except Exception as e:
            raise ImageProcessingError("统计信息叠加", None, str(e))