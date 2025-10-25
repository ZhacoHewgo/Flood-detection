#!/usr/bin/env python3
"""
性能测试脚本
Performance Testing Script for Flood Detection System
"""

import time
import numpy as np
import cv2
import psutil
import os
import sys
from typing import Dict, Any, List
import threading
from concurrent.futures import ThreadPoolExecutor
import gc

# 添加项目路径
sys.path.append('.')

from flood_detection_app.core.model_manager import ModelManager
from flood_detection_app.core.image_processor import ImageProcessor
from flood_detection_app.core.flood_analyzer import FloodAnalyzer
from flood_detection_app.core.visualization_engine import VisualizationEngine
from flood_detection_app.core.data_models import Detection, BoundingBox


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.results = {}
        self.test_image = self._create_test_image()
        
    def _create_test_image(self) -> np.ndarray:
        """创建测试图像"""
        # 创建一个1024x1024的测试图像
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        
        # 添加一些模拟的车辆区域
        cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.rectangle(image, (300, 300), (400, 400), (255, 0, 0), -1)
        cv2.rectangle(image, (500, 500), (600, 600), (0, 0, 255), -1)
        
        return image
    
    def _create_test_detections(self) -> List[Detection]:
        """创建测试检测结果"""
        detections = []
        
        # 创建一些测试边界框
        test_boxes = [
            (100, 100, 200, 200, 0.9),
            (300, 300, 400, 400, 0.8),
            (500, 500, 600, 600, 0.7),
            (700, 200, 800, 300, 0.85),
            (200, 700, 300, 800, 0.75)
        ]
        
        for i, (x1, y1, x2, y2, conf) in enumerate(test_boxes):
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
            detection = Detection(bbox=bbox, class_id=0, class_name="vehicle")
            detections.append(detection)
        
        return detections
    
    def _create_test_water_mask(self) -> np.ndarray:
        """创建测试水面掩码"""
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        
        # 添加一些水面区域
        cv2.circle(mask, (150, 150), 50, 1, -1)
        cv2.circle(mask, (350, 350), 60, 1, -1)
        cv2.rectangle(mask, (450, 450, 650, 650), 1, -1)
        
        return mask
    
    def test_component_initialization(self) -> Dict[str, float]:
        """测试组件初始化性能"""
        print("测试组件初始化性能...")
        
        results = {}
        
        # 测试ModelManager初始化
        start_time = time.time()
        model_manager = ModelManager()
        results['model_manager_init'] = time.time() - start_time
        
        # 测试ImageProcessor初始化
        start_time = time.time()
        image_processor = ImageProcessor()
        results['image_processor_init'] = time.time() - start_time
        
        # 测试FloodAnalyzer初始化
        start_time = time.time()
        flood_analyzer = FloodAnalyzer()
        results['flood_analyzer_init'] = time.time() - start_time
        
        # 测试VisualizationEngine初始化
        start_time = time.time()
        viz_engine = VisualizationEngine()
        results['viz_engine_init'] = time.time() - start_time
        
        return results
    
    def test_image_processing_performance(self) -> Dict[str, float]:
        """测试图像处理性能"""
        print("测试图像处理性能...")
        
        processor = ImageProcessor()
        results = {}
        
        # 测试图像缩放
        start_time = time.time()
        for _ in range(10):
            resized = processor.resize_with_aspect_ratio_fast(self.test_image, (640, 640))
        results['image_resize_10x'] = time.time() - start_time
        
        # 测试模型预处理
        start_time = time.time()
        for _ in range(10):
            tensor = processor.preprocess_for_model_fast(self.test_image, 'yolo', (640, 640))
        results['model_preprocess_10x'] = time.time() - start_time
        
        # 测试图像验证
        start_time = time.time()
        for _ in range(100):
            valid = processor.validate_image(self.test_image)
        results['image_validation_100x'] = time.time() - start_time
        
        return results
    
    def test_flood_analysis_performance(self) -> Dict[str, float]:
        """测试淹没分析性能"""
        print("测试淹没分析性能...")
        
        analyzer = FloodAnalyzer()
        results = {}
        
        # 创建测试数据
        detections = self._create_test_detections()
        water_mask = self._create_test_water_mask()
        
        # 测试重叠比例计算
        start_time = time.time()
        for detection in detections:
            for _ in range(20):
                ratio = analyzer.calculate_overlap_ratio(detection.bbox, water_mask)
        results['overlap_calculation_100x'] = time.time() - start_time
        
        # 测试场景分析
        start_time = time.time()
        for _ in range(10):
            result = analyzer.analyze_scene_batch(detections, water_mask)
        results['scene_analysis_10x'] = time.time() - start_time
        
        # 测试统计计算
        analysis_result = analyzer.analyze_scene_batch(detections, water_mask)
        start_time = time.time()
        for _ in range(50):
            stats = analyzer._calculate_statistics_fast(
                analysis_result.vehicles, 
                water_mask.shape, 
                50.0, 
                0.1
            )
        results['statistics_calculation_50x'] = time.time() - start_time
        
        return results
    
    def test_visualization_performance(self) -> Dict[str, float]:
        """测试可视化性能"""
        print("测试可视化性能...")
        
        viz_engine = VisualizationEngine()
        analyzer = FloodAnalyzer()
        results = {}
        
        # 创建测试数据
        detections = self._create_test_detections()
        water_mask = self._create_test_water_mask()
        analysis_result = analyzer.analyze_scene_batch(detections, water_mask)
        
        # 测试水面掩码叠加
        start_time = time.time()
        for _ in range(10):
            overlaid = viz_engine.overlay_water_mask_fast(self.test_image, water_mask)
        results['water_mask_overlay_10x'] = time.time() - start_time
        
        # 测试边界框绘制
        start_time = time.time()
        for _ in range(10):
            annotated = viz_engine._draw_all_annotations_fast(self.test_image.copy(), analysis_result.vehicles)
        results['bbox_drawing_10x'] = time.time() - start_time
        
        # 测试完整结果图像创建
        start_time = time.time()
        for _ in range(5):
            result_image = viz_engine.create_result_image_fast(self.test_image, analysis_result)
        results['result_image_creation_5x'] = time.time() - start_time
        
        return results
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用情况"""
        print("测试内存使用情况...")
        
        process = psutil.Process(os.getpid())
        
        # 初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建组件
        model_manager = ModelManager()
        image_processor = ImageProcessor()
        flood_analyzer = FloodAnalyzer()
        viz_engine = VisualizationEngine()
        
        # 组件创建后内存
        after_init_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行一些操作
        detections = self._create_test_detections()
        water_mask = self._create_test_water_mask()
        
        for _ in range(10):
            analysis_result = flood_analyzer.analyze_scene_batch(detections, water_mask)
            result_image = viz_engine.create_result_image_fast(self.test_image, analysis_result)
        
        # 操作后内存
        after_operations_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 清理内存
        if hasattr(model_manager, 'optimize_memory'):
            model_manager.optimize_memory()
        gc.collect()
        
        # 清理后内存
        after_cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'initial_memory_mb': initial_memory,
            'after_init_memory_mb': after_init_memory,
            'after_operations_memory_mb': after_operations_memory,
            'after_cleanup_memory_mb': after_cleanup_memory,
            'memory_increase_init_mb': after_init_memory - initial_memory,
            'memory_increase_operations_mb': after_operations_memory - after_init_memory,
            'memory_freed_cleanup_mb': after_operations_memory - after_cleanup_memory
        }
    
    def test_concurrent_performance(self) -> Dict[str, float]:
        """测试并发性能"""
        print("测试并发性能...")
        
        analyzer = FloodAnalyzer()
        detections = self._create_test_detections()
        water_mask = self._create_test_water_mask()
        
        results = {}
        
        def analyze_task():
            return analyzer.analyze_scene_batch(detections, water_mask)
        
        # 测试单线程性能
        start_time = time.time()
        for _ in range(20):
            analyze_task()
        results['single_thread_20x'] = time.time() - start_time
        
        # 测试多线程性能
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_task) for _ in range(20)]
            for future in futures:
                future.result()
        results['multi_thread_20x'] = time.time() - start_time
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有性能测试"""
        print("开始性能测试...")
        print("=" * 50)
        
        all_results = {}
        
        # 运行各项测试
        all_results['initialization'] = self.test_component_initialization()
        all_results['image_processing'] = self.test_image_processing_performance()
        all_results['flood_analysis'] = self.test_flood_analysis_performance()
        all_results['visualization'] = self.test_visualization_performance()
        all_results['memory_usage'] = self.test_memory_usage()
        all_results['concurrent'] = self.test_concurrent_performance()
        
        return all_results
    
    def print_results(self, results: Dict[str, Any]):
        """打印测试结果"""
        print("\n" + "=" * 50)
        print("性能测试结果")
        print("=" * 50)
        
        for category, category_results in results.items():
            print(f"\n{category.upper()}:")
            print("-" * 30)
            
            if isinstance(category_results, dict):
                for test_name, value in category_results.items():
                    if isinstance(value, float):
                        if 'memory' in test_name.lower():
                            print(f"  {test_name}: {value:.2f} MB")
                        else:
                            print(f"  {test_name}: {value:.4f} 秒")
                    else:
                        print(f"  {test_name}: {value}")
            else:
                print(f"  结果: {category_results}")
        
        print("\n" + "=" * 50)
        print("性能测试完成")
        print("=" * 50)


def main():
    """主函数"""
    tester = PerformanceTester()
    results = tester.run_all_tests()
    tester.print_results(results)
    
    # 保存结果到文件
    import json
    with open('performance_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 performance_test_results.json")


if __name__ == "__main__":
    main()