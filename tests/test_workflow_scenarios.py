"""
工作流场景测试
Workflow Scenario Tests

测试真实用户使用场景和边缘情况
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest

from flood_detection_app.desktop.main_window import MainWindow
from flood_detection_app.core.data_models import (
    BoundingBox, Detection, VehicleResult, Statistics, AnalysisResult, FloodLevel
)


class TestRealWorldScenarios:
    """真实世界使用场景测试"""
    
    def test_typical_user_workflow(self, qapp, test_image_file, temp_dir):
        """测试典型用户工作流程"""
        with patch('flood_detection_app.desktop.main_window.ModelManager') as mock_model_manager_class:
            # 设置模拟
            mock_model_manager = Mock()
            mock_model_manager.load_models.return_value = True
            mock_model_manager.get_available_models.return_value = {
                'vehicle_models': ['yolov11_car_detection', 'rtdetr_car_detection'],
                'water_models': ['deeplabv3_water', 'yolov11_seg_water']
            }
            mock_model_manager_class.return_value = mock_model_manager
            
            window = MainWindow()
            
            # 1. 用户启动应用
            assert window.isVisible() == False  # 窗口创建但未显示
            
            # 2. 用户选择图像文件
            with patch.object(window.file_operations, 'select_image_file', return_value=test_image_file):
                with patch('flood_detection_app.core.image_processor.ImageProcessor.load_image') as mock_load:
                    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
                    mock_load.return_value = test_image
                    
                    window.select_image_file()
                    
                    # 验证图像加载
                    assert window.current_image is not None
                    assert window.analyze_action.isEnabled()
            
            # 3. 用户选择模型
            window.vehicle_model_combo.setCurrentText('yolov11_car_detection')
            window.water_model_combo.setCurrentText('deeplabv3_water')
            
            # 4. 用户开始分析
            with patch.object(window.analysis_controller, 'start_analysis', return_value=True) as mock_start:
                window.start_analysis()
                mock_start.assert_called_once()
            
            # 5. 模拟分析完成
            analysis_result = self._create_sample_analysis_result()
            result_data = {
                'analysis_result': analysis_result,
                'result_image': test_image
            }
            window.on_analysis_completed(result_data)
            
            # 验证结果显示
            assert window.analysis_result is not None
            assert window.save_action.isEnabled()
            
            # 6. 用户保存结果
            save_path = os.path.join(temp_dir, "result.jpg")
            with patch.object(window.file_operations, 'save_result_image', return_value=save_path):
                with patch('flood_detection_app.core.image_processor.ImageProcessor.save_image', return_value=True):
                    with patch('flood_detection_app.core.visualization_engine.VisualizationEngine.create_result_image') as mock_create:
                        mock_create.return_value = test_image
                        
                        window.save_result()
                        mock_create.assert_called_once()
            
            window.close()
    
    def test_multiple_image_analysis_session(self, qapp, temp_dir):
        """测试多图像分析会话"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 创建多个测试图像
            test_images = []
            for i in range(3):
                image = np.random.randint(0, 255, (200 + i*50, 300 + i*50, 3), dtype=np.uint8)
                test_images.append(image)
            
            # 依次分析每个图像
            for i, image in enumerate(test_images):
                # 加载图像
                window.current_image = image
                window.analysis_controller.set_image(image)
                window.analyze_action.setEnabled(True)
                
                # 模拟分析
                analysis_result = self._create_sample_analysis_result()
                result_data = {
                    'analysis_result': analysis_result,
                    'result_image': image
                }
                window.on_analysis_completed(result_data)
                
                # 验证每次分析结果
                assert window.analysis_result is not None
                assert window.save_action.isEnabled()
                
                # 清除结果准备下一次分析
                if i < len(test_images) - 1:
                    window.clear_result()
                    assert window.analysis_result is None
            
            window.close()
    
    def test_model_switching_workflow(self, qapp, sample_image):
        """测试模型切换工作流程"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 设置可用模型
            window.vehicle_model_combo.addItems(['yolov11_car_detection', 'rtdetr_car_detection'])
            window.water_model_combo.addItems(['deeplabv3_water', 'yolov11_seg_water'])
            
            # 加载图像
            window.current_image = sample_image
            window.analyze_action.setEnabled(True)
            
            # 第一次分析
            analysis_result1 = self._create_sample_analysis_result()
            result_data1 = {
                'analysis_result': analysis_result1,
                'result_image': sample_image
            }
            window.on_analysis_completed(result_data1)
            assert window.analysis_result is not None
            
            # 切换模型
            window.vehicle_model_combo.setCurrentText('rtdetr_car_detection')
            # 模型切换应该清除之前的结果
            assert window.analysis_result is None
            
            # 第二次分析
            analysis_result2 = self._create_sample_analysis_result()
            result_data2 = {
                'analysis_result': analysis_result2,
                'result_image': sample_image
            }
            window.on_analysis_completed(result_data2)
            assert window.analysis_result is not None
            
            window.close()
    
    def _create_sample_analysis_result(self):
        """创建示例分析结果"""
        # 创建车辆检测结果
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=100, confidence=0.85)
        detection = Detection(bbox=bbox, class_id=0, class_name="car")
        vehicle_result = VehicleResult(
            detection=detection,
            flood_level=FloodLevel.MODERATE,
            overlap_ratio=0.4,
            vehicle_id=1
        )
        
        # 创建统计信息
        statistics = Statistics(
            total_vehicles=1,
            light_flood_count=0,
            moderate_flood_count=1,
            severe_flood_count=0,
            water_coverage_percentage=15.5,
            processing_time=1.8
        )
        
        # 创建水面掩码
        water_mask = np.zeros((200, 300), dtype=np.uint8)
        water_mask[100:150, 50:200] = 255
        
        return AnalysisResult(
            vehicles=[vehicle_result],
            water_mask=water_mask,
            statistics=statistics,
            original_image_shape=(200, 300)
        )


class TestEdgeCases:
    """边缘情况测试"""
    
    def test_empty_image_analysis(self, qapp):
        """测试空图像分析"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 创建空图像（全黑）
            empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
            window.current_image = empty_image
            window.analysis_controller.set_image(empty_image)
            
            # 模拟分析结果（无车辆检测）
            empty_result = AnalysisResult(
                vehicles=[],
                water_mask=np.zeros((100, 100), dtype=np.uint8),
                statistics=Statistics(0, 0, 0, 0, 0.0, 0.5),
                original_image_shape=(100, 100)
            )
            
            result_data = {
                'analysis_result': empty_result,
                'result_image': empty_image
            }
            window.on_analysis_completed(result_data)
            
            # 验证空结果处理
            assert window.analysis_result is not None
            assert window.analysis_result.statistics.total_vehicles == 0
            
            window.close()
    
    def test_large_image_handling(self, qapp):
        """测试大图像处理"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 创建大图像
            large_image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
            
            # 设置图像（应该能够处理而不崩溃）
            window.current_image = large_image
            window.original_image_panel.set_image(large_image)
            
            # 验证图像设置成功
            assert window.current_image is not None
            assert window.current_image.shape == large_image.shape
            
            window.close()
    
    def test_rapid_user_interactions(self, qapp, sample_image):
        """测试快速用户交互"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 快速连续操作
            for i in range(5):
                # 快速设置图像
                window.current_image = sample_image
                window.analyze_action.setEnabled(True)
                
                # 快速清除
                window.clear_result()
                
                # 短暂等待
                QTest.qWait(10)
            
            # 验证最终状态正确
            assert window.analysis_result is None
            assert not window.save_action.isEnabled()
            
            window.close()
    
    def test_analysis_interruption(self, qapp, sample_image):
        """测试分析中断"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 设置图像
            window.current_image = sample_image
            window.analyze_action.setEnabled(True)
            
            # 开始分析
            window.on_analysis_started()
            assert not window.analyze_action.isEnabled()
            
            # 模拟分析失败/中断
            window.on_analysis_failed("分析被中断")
            
            # 验证状态恢复
            assert window.analyze_action.isEnabled()
            assert window.select_file_action.isEnabled()
            assert not window.progress_bar.isVisible()
            
            window.close()


class TestPerformanceScenarios:
    """性能场景测试"""
    
    def test_memory_usage_with_multiple_images(self, qapp):
        """测试多图像内存使用"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 处理多个图像
            for i in range(10):
                # 创建不同大小的图像
                size = 100 + i * 50
                image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                
                # 设置图像
                window.current_image = image
                window.original_image_panel.set_image(image)
                
                # 模拟分析结果
                result = self._create_minimal_analysis_result(size)
                result_data = {
                    'analysis_result': result,
                    'result_image': image
                }
                window.on_analysis_completed(result_data)
                
                # 清除结果
                window.clear_result()
            
            # 验证最终状态
            assert window.analysis_result is None
            
            window.close()
    
    def test_ui_responsiveness_during_updates(self, qapp):
        """测试更新期间UI响应性"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 快速连续更新统计信息
            for i in range(20):
                stats = Statistics(i, i//3, i//3, i//3, float(i*2), float(i*0.1))
                
                # 更新统计信息
                window.statistics_widget.update_statistics(stats, {})
                window.compact_stats.update_statistics(stats)
                
                # 更新状态
                window.update_status(f"处理第{i+1}个结果")
                
                # 短暂等待
                QTest.qWait(5)
            
            # 验证UI仍然响应
            assert window.status_label.text() == "处理第20个结果"
            
            window.close()
    
    def _create_minimal_analysis_result(self, image_size):
        """创建最小分析结果"""
        return AnalysisResult(
            vehicles=[],
            water_mask=np.zeros((image_size, image_size), dtype=np.uint8),
            statistics=Statistics(0, 0, 0, 0, 0.0, 0.1),
            original_image_shape=(image_size, image_size)
        )


class TestErrorRecoveryScenarios:
    """错误恢复场景测试"""
    
    def test_recovery_from_model_loading_failure(self, qapp):
        """测试模型加载失败后的恢复"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 模拟模型加载失败
            with patch.object(window.analysis_controller, 'load_models', return_value=False):
                with patch('PyQt6.QtWidgets.QMessageBox.warning') as mock_warning:
                    window.load_models()
                    mock_warning.assert_called()
            
            # 验证应用仍然可用
            assert window.vehicle_model_combo is not None
            assert window.water_model_combo is not None
            
            # 模拟重新加载成功
            with patch.object(window.analysis_controller, 'load_models', return_value=True):
                with patch.object(window.analysis_controller, 'get_available_models', 
                                return_value={'vehicle_models': ['yolov11'], 'water_models': ['deeplabv3']}):
                    window.load_models()
                    
                    # 验证恢复
                    assert window.vehicle_model_combo.count() > 0
                    assert window.water_model_combo.count() > 0
            
            window.close()
    
    def test_recovery_from_file_operation_errors(self, qapp, sample_image):
        """测试文件操作错误后的恢复"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 模拟文件选择失败
            with patch.object(window.file_operations, 'select_image_file', return_value=None):
                window.select_image_file()
                
                # 验证状态未改变
                assert window.current_image is None
                assert not window.analyze_action.isEnabled()
            
            # 模拟文件选择成功
            with patch.object(window.file_operations, 'select_image_file', return_value="test.jpg"):
                with patch('flood_detection_app.core.image_processor.ImageProcessor.load_image', return_value=sample_image):
                    window.select_image_file()
                    
                    # 验证恢复
                    assert window.current_image is not None
                    assert window.analyze_action.isEnabled()
            
            window.close()
    
    def test_graceful_shutdown_during_analysis(self, qapp, sample_image):
        """测试分析期间优雅关闭"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 设置分析状态
            window.current_image = sample_image
            window.on_analysis_started()
            
            # 验证分析状态
            assert not window.analyze_action.isEnabled()
            assert window.progress_bar.isVisible()
            
            # 模拟关闭窗口
            # 这应该能够优雅处理而不崩溃
            window.close()
            
            # 验证关闭成功（主要是不抛出异常）