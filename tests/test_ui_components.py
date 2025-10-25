"""
UI组件集成测试
UI Components Integration Tests

专门测试各个UI组件的交互和状态管理
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from flood_detection_app.desktop.image_display_widget import ImageDisplayPanel
from flood_detection_app.desktop.statistics_widget import StatisticsWidget, CompactStatisticsWidget
from flood_detection_app.core.data_models import Statistics, FloodLevel


class TestImageDisplayWidget:
    """图像显示组件测试"""
    
    def test_image_display_initialization(self, qapp):
        """测试图像显示组件初始化"""
        widget = ImageDisplayPanel("测试图像")
        
        # 验证初始化
        assert widget.title == "测试图像"
        assert hasattr(widget, 'image_label')
        assert hasattr(widget, 'title_label')
        
        # 验证初始状态
        assert widget.current_image is None
        
        widget.close()
    
    def test_image_setting_and_display(self, qapp, sample_image):
        """测试图像设置和显示"""
        widget = ImageDisplayPanel("测试图像")
        
        # 设置图像
        widget.set_image(sample_image)
        
        # 验证图像设置
        assert widget.current_image is not None
        assert np.array_equal(widget.current_image, sample_image)
        
        # 验证显示更新
        pixmap = widget.image_label.pixmap()
        assert pixmap is not None
        assert not pixmap.isNull()
        
        widget.close()
    
    def test_image_clearing(self, qapp, sample_image):
        """测试图像清除"""
        widget = ImageDisplayPanel("测试图像")
        
        # 先设置图像
        widget.set_image(sample_image)
        assert widget.current_image is not None
        
        # 清除图像
        widget.clear_image()
        
        # 验证清除结果
        assert widget.current_image is None
        
        widget.close()
    
    def test_image_scaling(self, qapp):
        """测试图像缩放功能"""
        widget = ImageDisplayPanel("测试图像")
        
        # 创建大图像
        large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        # 设置图像
        widget.set_image(large_image)
        
        # 验证图像被正确处理（不会导致UI问题）
        assert widget.current_image is not None
        
        widget.close()


class TestStatisticsWidget:
    """统计信息组件测试"""
    
    def test_statistics_widget_initialization(self, qapp):
        """测试统计信息组件初始化"""
        widget = StatisticsWidget()
        
        # 验证初始化
        assert hasattr(widget, 'total_vehicles_label')
        assert hasattr(widget, 'light_flood_label')
        assert hasattr(widget, 'moderate_flood_label')
        assert hasattr(widget, 'severe_flood_label')
        assert hasattr(widget, 'water_coverage_label')
        assert hasattr(widget, 'processing_time_label')
        
        widget.close()
    
    def test_statistics_update(self, qapp, sample_statistics):
        """测试统计信息更新"""
        widget = StatisticsWidget()
        
        # 准备附加信息
        additional_info = {
            'image_size_label': '100x100',
            'vehicle_model_label': 'YOLOv11',
            'water_model_label': 'DeepLabV3',
            'analysis_time_label': '2.5秒'
        }
        
        # 更新统计信息
        widget.update_statistics(sample_statistics, additional_info)
        
        # 验证更新结果
        assert str(sample_statistics.total_vehicles) in widget.total_vehicles_label.text()
        assert str(sample_statistics.light_flood_count) in widget.light_flood_label.text()
        assert str(sample_statistics.moderate_flood_count) in widget.moderate_flood_label.text()
        assert str(sample_statistics.severe_flood_count) in widget.severe_flood_label.text()
        
        widget.close()
    
    def test_statistics_clearing(self, qapp, sample_statistics):
        """测试统计信息清除"""
        widget = StatisticsWidget()
        
        # 先设置统计信息
        widget.update_statistics(sample_statistics, {})
        
        # 清除统计信息
        widget.clear_statistics()
        
        # 验证清除结果
        assert "0" in widget.total_vehicles_label.text()
        assert "0" in widget.light_flood_label.text()
        assert "0" in widget.moderate_flood_label.text()
        assert "0" in widget.severe_flood_label.text()
        
        widget.close()


class TestCompactStatisticsWidget:
    """紧凑型统计信息组件测试"""
    
    def test_compact_statistics_initialization(self, qapp):
        """测试紧凑型统计信息组件初始化"""
        widget = CompactStatisticsWidget()
        
        # 验证初始化
        assert hasattr(widget, 'stats_label')
        
        widget.close()
    
    def test_compact_statistics_update(self, qapp, sample_statistics):
        """测试紧凑型统计信息更新"""
        widget = CompactStatisticsWidget()
        
        # 更新统计信息
        widget.update_statistics(sample_statistics)
        
        # 验证更新结果（检查标签文本包含关键信息）
        label_text = widget.stats_label.text()
        assert str(sample_statistics.total_vehicles) in label_text
        
        widget.close()
    
    def test_compact_statistics_clearing(self, qapp):
        """测试紧凑型统计信息清除"""
        widget = CompactStatisticsWidget()
        
        # 清除统计信息
        widget.clear_statistics()
        
        # 验证清除结果
        assert widget.stats_label.text() == ""
        
        widget.close()


class TestUIComponentInteractions:
    """UI组件交互测试"""
    
    def test_image_and_statistics_coordination(self, qapp, sample_image, sample_statistics):
        """测试图像显示和统计信息的协调"""
        image_widget = ImageDisplayPanel("原始图像")
        stats_widget = StatisticsWidget()
        
        # 设置图像
        image_widget.set_image(sample_image)
        
        # 更新统计信息
        additional_info = {
            'image_size_label': f'{sample_image.shape[1]}x{sample_image.shape[0]}'
        }
        stats_widget.update_statistics(sample_statistics, additional_info)
        
        # 验证协调性
        assert image_widget.current_image is not None
        assert str(sample_image.shape[1]) in stats_widget.image_size_label.text()
        
        # 清除测试
        image_widget.clear_image()
        stats_widget.clear_statistics()
        
        assert image_widget.current_image is None
        assert "0" in stats_widget.total_vehicles_label.text()
        
        image_widget.close()
        stats_widget.close()
    
    def test_multiple_image_updates(self, qapp):
        """测试多次图像更新"""
        widget = ImageDisplayPanel("测试图像")
        
        # 创建多个不同的图像
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8),
            np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        ]
        
        # 依次设置图像
        for i, image in enumerate(images):
            widget.set_image(image)
            
            # 验证当前图像正确
            assert widget.current_image is not None
            assert widget.current_image.shape == image.shape
            
            # 验证显示更新
            pixmap = widget.image_label.pixmap()
            assert pixmap is not None
            assert not pixmap.isNull()
        
        widget.close()
    
    def test_statistics_progressive_updates(self, qapp):
        """测试统计信息渐进式更新"""
        widget = StatisticsWidget()
        
        # 创建不同的统计数据
        stats_list = [
            Statistics(1, 1, 0, 0, 10.0, 1.0),
            Statistics(2, 1, 1, 0, 15.0, 1.5),
            Statistics(3, 1, 1, 1, 20.0, 2.0)
        ]
        
        # 依次更新统计信息
        for stats in stats_list:
            widget.update_statistics(stats, {})
            
            # 验证更新正确
            assert str(stats.total_vehicles) in widget.total_vehicles_label.text()
            assert str(stats.light_flood_count) in widget.light_flood_label.text()
            assert str(stats.moderate_flood_count) in widget.moderate_flood_label.text()
            assert str(stats.severe_flood_count) in widget.severe_flood_label.text()
        
        widget.close()


class TestUIResponsiveness:
    """UI响应性测试"""
    
    def test_widget_resize_handling(self, qapp, sample_image):
        """测试组件大小调整处理"""
        widget = ImageDisplayPanel("测试图像")
        
        # 设置初始大小
        widget.resize(400, 300)
        QTest.qWait(50)
        
        # 设置图像
        widget.set_image(sample_image)
        
        # 调整大小
        widget.resize(800, 600)
        QTest.qWait(50)
        
        # 验证图像仍然正确显示
        assert widget.current_image is not None
        pixmap = widget.image_label.pixmap()
        assert pixmap is not None
        
        widget.close()
    
    def test_rapid_updates(self, qapp):
        """测试快速更新处理"""
        stats_widget = StatisticsWidget()
        
        # 快速连续更新
        for i in range(10):
            stats = Statistics(i, i//3, i//3, i//3, float(i*5), float(i*0.1))
            stats_widget.update_statistics(stats, {})
            
            # 短暂等待
            QTest.qWait(10)
        
        # 验证最终状态正确
        final_stats = Statistics(9, 3, 3, 3, 45.0, 0.9)
        stats_widget.update_statistics(final_stats, {})
        
        assert "9" in stats_widget.total_vehicles_label.text()
        
        stats_widget.close()


class TestErrorHandlingInUI:
    """UI错误处理测试"""
    
    def test_invalid_image_handling(self, qapp):
        """测试无效图像处理"""
        widget = ImageDisplayPanel("测试图像")
        
        # 尝试设置None图像
        widget.set_image(None)
        
        # 验证不会崩溃，且状态正确
        assert widget.current_image is None
        
        # 尝试设置无效形状的数组
        invalid_image = np.array([1, 2, 3])  # 1D数组
        widget.set_image(invalid_image)
        
        # 验证处理无效输入
        # 具体行为取决于实现，但不应该崩溃
        
        widget.close()
    
    def test_statistics_with_invalid_data(self, qapp):
        """测试无效统计数据处理"""
        widget = StatisticsWidget()
        
        # 创建包含异常值的统计数据
        invalid_stats = Statistics(-1, -1, -1, -1, -10.0, -1.0)
        
        # 更新统计信息（不应该崩溃）
        widget.update_statistics(invalid_stats, {})
        
        # 验证UI仍然可用
        assert widget.total_vehicles_label.text() is not None
        
        widget.close()
    
    def test_memory_management(self, qapp):
        """测试内存管理"""
        widgets = []
        
        # 创建多个组件
        for i in range(10):
            widget = ImageDisplayPanel(f"测试图像{i}")
            
            # 设置大图像
            large_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
            widget.set_image(large_image)
            
            widgets.append(widget)
        
        # 清理所有组件
        for widget in widgets:
            widget.clear_image()
            widget.close()
        
        # 验证清理完成（主要是确保不会内存泄漏）
        # 这个测试主要是确保代码执行不出错