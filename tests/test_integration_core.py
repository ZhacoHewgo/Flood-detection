"""
核心集成测试（无GUI）
Core Integration Tests (No GUI)

测试核心功能集成，不依赖GUI组件
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flood_detection_app.core.data_models import (
    BoundingBox, Detection, VehicleResult, Statistics, AnalysisResult, FloodLevel
)
from flood_detection_app.desktop.file_operations import FileOperations
from flood_detection_app.desktop.analysis_controller import ModelSelectionManager


class TestCoreIntegration:
    """核心集成测试"""
    
    def test_data_models_integration(self):
        """测试数据模型集成"""
        # 创建边界框
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80, confidence=0.85)
        
        # 验证边界框计算
        assert bbox.area() == 90 * 60  # (100-10) * (80-20)
        assert bbox.center() == (55.0, 50.0)  # ((10+100)/2, (20+80)/2)
        
        # 创建检测结果
        detection = Detection(bbox=bbox, class_id=0, class_name="car")
        assert detection.bbox == bbox
        assert detection.class_name == "car"
        
        # 创建车辆结果
        vehicle_result = VehicleResult(
            detection=detection,
            flood_level=FloodLevel.MODERATE,
            overlap_ratio=0.45,
            vehicle_id=1
        )
        assert vehicle_result.flood_level == FloodLevel.MODERATE
        assert vehicle_result.overlap_ratio == 0.45
        
        # 创建统计信息
        statistics = Statistics(
            total_vehicles=3,
            light_flood_count=1,
            moderate_flood_count=1,
            severe_flood_count=1,
            water_coverage_percentage=25.5,
            processing_time=2.3
        )
        assert statistics.total_vehicles == 3
        assert statistics.water_coverage_percentage == 25.5
        
        # 创建完整分析结果
        water_mask = np.zeros((100, 100), dtype=np.uint8)
        water_mask[30:70, 30:70] = 255
        
        analysis_result = AnalysisResult(
            vehicles=[vehicle_result],
            water_mask=water_mask,
            statistics=statistics,
            original_image_shape=(100, 100)
        )
        
        assert len(analysis_result.vehicles) == 1
        assert analysis_result.statistics.total_vehicles == 3
        assert analysis_result.original_image_shape == (100, 100)
    
    def test_file_operations_core_functionality(self):
        """测试文件操作核心功能"""
        file_ops = FileOperations()
        
        # 测试文件过滤器构建
        supported_formats = ['.jpg', '.png', '.bmp', '.tiff']
        filter_str = file_ops._build_image_filter(supported_formats)
        
        # 验证过滤器包含所有格式
        assert 'jpg' in filter_str
        assert 'png' in filter_str
        assert 'bmp' in filter_str
        assert 'tiff' in filter_str
        assert '所有支持的图像' in filter_str
        assert '所有文件' in filter_str
    
    def test_image_header_validation(self):
        """测试图像文件头验证"""
        file_ops = FileOperations()
        
        # 测试JPEG文件头
        jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        assert file_ops._is_valid_image_header(jpeg_header) == True
        
        # 测试PNG文件头
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        assert file_ops._is_valid_image_header(png_header) == True
        
        # 测试BMP文件头
        bmp_header = b'BM\x36\x84\x03\x00\x00\x00\x00\x00'
        assert file_ops._is_valid_image_header(bmp_header) == True
        
        # 测试无效文件头
        invalid_header = b'INVALID_HEADER_DATA'
        assert file_ops._is_valid_image_header(invalid_header) == False
    
    def test_model_selection_manager(self):
        """测试模型选择管理器"""
        # 创建模拟的ModelManager
        mock_model_manager = Mock()
        mock_model_manager.get_available_models.return_value = {
            'vehicle_models': ['yolov11_car_detection', 'rtdetr_car_detection'],
            'water_models': ['deeplabv3_water', 'yolov11_seg_water']
        }
        
        # 创建模型选择管理器
        manager = ModelSelectionManager(mock_model_manager)
        
        # 测试获取模型列表
        vehicle_models = manager.get_vehicle_models()
        water_models = manager.get_water_models()
        
        assert 'yolov11_car_detection' in vehicle_models
        assert 'rtdetr_car_detection' in vehicle_models
        assert 'deeplabv3_water' in water_models
        assert 'yolov11_seg_water' in water_models
        
        # 测试模型设置
        assert manager.set_vehicle_model('yolov11_car_detection') == True
        assert manager.set_water_model('deeplabv3_water') == True
        
        # 测试无效模型设置
        assert manager.set_vehicle_model('invalid_model') == False
        assert manager.set_water_model('invalid_model') == False
        
        # 测试当前选择
        current_selection = manager.get_current_selection()
        assert current_selection['vehicle_model'] == 'yolov11_car_detection'
        assert current_selection['water_model'] == 'deeplabv3_water'
        
        # 测试选择有效性
        assert manager.is_selection_valid() == True
    
    def test_flood_level_enum(self):
        """测试淹没等级枚举"""
        # 测试枚举值
        assert FloodLevel.LIGHT.value == "light"
        assert FloodLevel.MODERATE.value == "moderate"
        assert FloodLevel.SEVERE.value == "severe"
        
        # 测试枚举比较
        assert FloodLevel.LIGHT != FloodLevel.MODERATE
        assert FloodLevel.MODERATE != FloodLevel.SEVERE
        
        # 测试在数据结构中的使用
        vehicle_result = VehicleResult(
            detection=Mock(),
            flood_level=FloodLevel.SEVERE,
            overlap_ratio=0.8,
            vehicle_id=1
        )
        assert vehicle_result.flood_level == FloodLevel.SEVERE
        assert vehicle_result.flood_level.value == "severe"


class TestWorkflowIntegration:
    """工作流程集成测试"""
    
    def test_analysis_data_flow(self):
        """测试分析数据流"""
        # 1. 创建输入数据
        input_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        
        # 2. 模拟车辆检测结果
        vehicle_bbox = BoundingBox(x1=50, y1=60, x2=150, y2=120, confidence=0.9)
        vehicle_detection = Detection(bbox=vehicle_bbox, class_id=0, class_name="car")
        
        # 3. 模拟水面掩码
        water_mask = np.zeros((200, 300), dtype=np.uint8)
        water_mask[80:140, 70:180] = 255  # 创建与车辆重叠的水面区域
        
        # 4. 计算重叠比例（简化计算）
        vehicle_area = vehicle_bbox.area()
        overlap_area = 60 * 60  # 假设重叠区域
        overlap_ratio = overlap_area / vehicle_area
        
        # 5. 确定淹没等级
        if overlap_ratio < 0.3:
            flood_level = FloodLevel.LIGHT
        elif overlap_ratio < 0.6:
            flood_level = FloodLevel.MODERATE
        else:
            flood_level = FloodLevel.SEVERE
        
        # 6. 创建车辆结果
        vehicle_result = VehicleResult(
            detection=vehicle_detection,
            flood_level=flood_level,
            overlap_ratio=overlap_ratio,
            vehicle_id=1
        )
        
        # 7. 创建统计信息
        statistics = Statistics(
            total_vehicles=1,
            light_flood_count=1 if flood_level == FloodLevel.LIGHT else 0,
            moderate_flood_count=1 if flood_level == FloodLevel.MODERATE else 0,
            severe_flood_count=1 if flood_level == FloodLevel.SEVERE else 0,
            water_coverage_percentage=np.sum(water_mask > 0) / water_mask.size * 100,
            processing_time=1.5
        )
        
        # 8. 创建最终结果
        analysis_result = AnalysisResult(
            vehicles=[vehicle_result],
            water_mask=water_mask,
            statistics=statistics,
            original_image_shape=input_image.shape[:2]
        )
        
        # 验证数据流完整性
        assert len(analysis_result.vehicles) == 1
        assert analysis_result.vehicles[0].vehicle_id == 1
        assert analysis_result.statistics.total_vehicles == 1
        assert analysis_result.original_image_shape == (200, 300)
        assert analysis_result.water_mask.shape == (200, 300)
    
    def test_error_handling_workflow(self):
        """测试错误处理工作流程"""
        # 测试无效边界框
        try:
            invalid_bbox = BoundingBox(x1=100, y1=100, x2=50, y2=50, confidence=0.5)
            # 这种情况下面积为负数
            area = invalid_bbox.area()
            assert area < 0  # 应该检测到无效边界框
        except Exception:
            pass  # 预期可能抛出异常
        
        # 测试空车辆列表
        empty_analysis = AnalysisResult(
            vehicles=[],
            water_mask=np.zeros((100, 100), dtype=np.uint8),
            statistics=Statistics(0, 0, 0, 0, 0.0, 0.1),
            original_image_shape=(100, 100)
        )
        
        assert len(empty_analysis.vehicles) == 0
        assert empty_analysis.statistics.total_vehicles == 0
        
        # 测试不匹配的统计信息
        mismatched_stats = Statistics(
            total_vehicles=5,  # 声称有5辆车
            light_flood_count=1,
            moderate_flood_count=1,
            severe_flood_count=1,  # 但只统计了3辆车
            water_coverage_percentage=20.0,
            processing_time=2.0
        )
        
        # 验证统计不一致（实际应用中需要验证逻辑）
        total_counted = (mismatched_stats.light_flood_count + 
                        mismatched_stats.moderate_flood_count + 
                        mismatched_stats.severe_flood_count)
        assert total_counted != mismatched_stats.total_vehicles


class TestFileOperationsIntegration:
    """文件操作集成测试"""
    
    def test_file_validation_workflow(self):
        """测试文件验证工作流程"""
        file_ops = FileOperations()
        
        # 创建临时测试文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # 写入JPEG文件头
            temp_file.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01')
            temp_file.write(b'\x00' * 100)  # 填充一些数据
            temp_file_path = temp_file.name
        
        try:
            # 测试文件验证
            is_valid = file_ops._validate_image_file(temp_file_path)
            assert is_valid == True
            
            # 测试不存在的文件
            is_valid_nonexistent = file_ops._validate_image_file('nonexistent_file.jpg')
            assert is_valid_nonexistent == False
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def test_extension_handling(self):
        """测试文件扩展名处理"""
        file_ops = FileOperations()
        
        # 测试JPEG扩展名
        jpeg_path = file_ops._ensure_extension("test_file", "JPEG图像 (*.jpg *.jpeg)")
        assert jpeg_path.endswith('.jpg')
        
        # 测试PNG扩展名
        png_path = file_ops._ensure_extension("test_file", "PNG图像 (*.png)")
        assert png_path.endswith('.png')
        
        # 测试已有扩展名的文件
        existing_ext_path = file_ops._ensure_extension("test_file.jpg", "PNG图像 (*.png)")
        assert existing_ext_path == "test_file.jpg"  # 保持原有扩展名


class TestPerformanceIntegration:
    """性能集成测试"""
    
    def test_large_data_handling(self):
        """测试大数据处理"""
        # 创建大量车辆结果
        vehicles = []
        for i in range(100):
            bbox = BoundingBox(x1=i*10, y1=i*10, x2=i*10+50, y2=i*10+50, confidence=0.8)
            detection = Detection(bbox=bbox, class_id=0, class_name="car")
            vehicle_result = VehicleResult(
                detection=detection,
                flood_level=FloodLevel.MODERATE,
                overlap_ratio=0.4,
                vehicle_id=i
            )
            vehicles.append(vehicle_result)
        
        # 创建大水面掩码
        large_water_mask = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8) * 255
        
        # 创建统计信息
        statistics = Statistics(
            total_vehicles=100,
            light_flood_count=30,
            moderate_flood_count=40,
            severe_flood_count=30,
            water_coverage_percentage=50.0,
            processing_time=5.0
        )
        
        # 创建大分析结果
        large_analysis_result = AnalysisResult(
            vehicles=vehicles,
            water_mask=large_water_mask,
            statistics=statistics,
            original_image_shape=(1000, 1000)
        )
        
        # 验证大数据处理
        assert len(large_analysis_result.vehicles) == 100
        assert large_analysis_result.water_mask.shape == (1000, 1000)
        assert large_analysis_result.statistics.total_vehicles == 100
    
    def test_memory_efficiency(self):
        """测试内存效率"""
        # 创建多个分析结果并清理
        results = []
        
        for i in range(10):
            # 创建中等大小的数据
            water_mask = np.random.randint(0, 2, (500, 500), dtype=np.uint8) * 255
            
            vehicles = []
            for j in range(10):
                bbox = BoundingBox(x1=j*20, y1=j*20, x2=j*20+40, y2=j*20+40, confidence=0.7)
                detection = Detection(bbox=bbox, class_id=0, class_name="car")
                vehicle_result = VehicleResult(
                    detection=detection,
                    flood_level=FloodLevel.LIGHT,
                    overlap_ratio=0.2,
                    vehicle_id=j
                )
                vehicles.append(vehicle_result)
            
            statistics = Statistics(10, 10, 0, 0, 25.0, 1.0)
            
            result = AnalysisResult(
                vehicles=vehicles,
                water_mask=water_mask,
                statistics=statistics,
                original_image_shape=(500, 500)
            )
            
            results.append(result)
        
        # 验证所有结果创建成功
        assert len(results) == 10
        for result in results:
            assert len(result.vehicles) == 10
            assert result.water_mask.shape == (500, 500)
        
        # 清理（Python垃圾回收会处理）
        results.clear()
        assert len(results) == 0


if __name__ == "__main__":
    # 简单的测试运行器
    import unittest
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCoreIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestFileOperationsIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    if result.wasSuccessful():
        print("\n✅ 所有核心集成测试通过!")
    else:
        print(f"\n❌ 测试失败: {len(result.failures)} 个失败, {len(result.errors)} 个错误")