"""
桌面应用集成测试
Desktop Application Integration Tests

测试完整的图像分析工作流程、UI组件交互、状态管理、文件操作和错误处理
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QPixmap

from flood_detection_app.desktop.main_window import MainWindow
from flood_detection_app.desktop.analysis_controller import AnalysisController
from flood_detection_app.desktop.file_operations import FileOperations
from flood_detection_app.core.data_models import FloodLevel, Statistics, AnalysisResult


class TestDesktopIntegration:
    """桌面应用集成测试类"""
    
    def test_main_window_initialization(self, qapp):
        """测试主窗口初始化"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 验证窗口基本属性
            assert window.windowTitle() == "积水车辆检测系统 - Flood Vehicle Detection"
            assert window.minimumSize().width() == 1200
            assert window.minimumSize().height() == 800
            
            # 验证UI组件存在
            assert hasattr(window, 'original_image_panel')
            assert hasattr(window, 'result_image_panel')
            assert hasattr(window, 'statistics_widget')
            assert hasattr(window, 'vehicle_model_combo')
            assert hasattr(window, 'water_model_combo')
            assert hasattr(window, 'analyze_action')
            assert hasattr(window, 'save_action')
            
            # 验证初始状态
            assert not window.analyze_action.isEnabled()
            assert not window.save_action.isEnabled()
            assert window.current_image is None
            assert window.analysis_result is None
            
            window.close()
    
    def test_complete_analysis_workflow(self, qapp, sample_image, sample_analysis_result):
        """测试完整的图像分析工作流程"""
        with patch('flood_detection_app.desktop.main_window.ModelManager') as mock_model_manager_class:
            # 设置模拟
            mock_model_manager = Mock()
            mock_model_manager.load_models.return_value = True
            mock_model_manager.get_available_models.return_value = {
                'vehicle_models': ['yolov11_car_detection'],
                'water_models': ['deeplabv3_water']
            }
            mock_model_manager_class.return_value = mock_model_manager
            
            window = MainWindow()
            
            # 模拟分析控制器
            with patch.object(window.analysis_controller, 'start_analysis') as mock_start_analysis:
                with patch.object(window.analysis_controller, 'load_models', return_value=True):
                    with patch.object(window.analysis_controller, 'get_available_models', 
                                    return_value={'vehicle_models': ['yolov11'], 'water_models': ['deeplabv3']}):
                        
                        # 1. 加载图像
                        window.current_image = sample_image
                        window.analysis_controller.set_image(sample_image)
                        window.original_image_panel.set_image(sample_image)
                        window.analyze_action.setEnabled(True)
                        
                        # 验证图像加载后的状态
                        assert window.current_image is not None
                        assert window.analyze_action.isEnabled()
                        assert not window.save_action.isEnabled()
                        
                        # 2. 开始分析
                        window.start_analysis()
                        mock_start_analysis.assert_called_once()
                        
                        # 3. 模拟分析完成
                        result_data = {
                            'analysis_result': sample_analysis_result,
                            'result_image': sample_image
                        }
                        window.on_analysis_completed(result_data)
                        
                        # 验证分析完成后的状态
                        assert window.analysis_result == sample_analysis_result
                        assert window.save_action.isEnabled()
            
            window.close()
    
    def test_ui_component_interactions(self, qapp):
        """测试UI组件交互和状态管理"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 测试模型选择变化
            window.vehicle_model_combo.addItems(['model1', 'model2'])
            window.water_model_combo.addItems(['water1', 'water2'])
            
            # 模拟模型选择变化
            with patch.object(window, 'clear_result') as mock_clear:
                window.analysis_result = Mock()  # 设置有结果
                window.on_model_selection_changed()
                mock_clear.assert_called_once()
            
            # 测试按钮状态管理
            window.current_image = np.zeros((100, 100, 3))
            window.analyze_action.setEnabled(True)
            
            # 模拟分析开始
            window.on_analysis_started()
            assert not window.analyze_action.isEnabled()
            assert not window.select_file_action.isEnabled()
            assert window.progress_bar.isVisible()
            
            # 模拟分析失败
            window.on_analysis_failed("测试错误")
            assert window.analyze_action.isEnabled()
            assert window.select_file_action.isEnabled()
            assert not window.progress_bar.isVisible()
            
            window.close()
    
    def test_file_operations_integration(self, qapp, test_image_file, temp_dir):
        """测试文件操作和错误处理"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 测试图像加载
            with patch.object(window.file_operations, 'select_image_file', return_value=test_image_file):
                with patch('flood_detection_app.core.image_processor.ImageProcessor.load_image') as mock_load:
                    mock_load.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    
                    window.select_image_file()
                    
                    # 验证图像加载
                    assert window.current_image is not None
                    assert window.analyze_action.isEnabled()
            
            # 测试保存结果
            window.analysis_result = Mock()
            save_path = os.path.join(temp_dir, "result.jpg")
            
            with patch.object(window.file_operations, 'save_result_image', return_value=save_path):
                with patch('flood_detection_app.core.image_processor.ImageProcessor.save_image', return_value=True):
                    with patch('flood_detection_app.core.visualization_engine.VisualizationEngine.create_result_image') as mock_create:
                        mock_create.return_value = np.zeros((100, 100, 3))
                        
                        window.save_result()
                        mock_create.assert_called_once()
            
            window.close()
    
    def test_error_handling(self, qapp):
        """测试错误处理机制"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 测试模型加载失败
            with patch.object(window.analysis_controller, 'load_models', return_value=False):
                with patch('PyQt6.QtWidgets.QMessageBox.warning') as mock_warning:
                    window.load_models()
                    mock_warning.assert_called()
            
            # 测试图像加载失败
            with patch('flood_detection_app.core.image_processor.ImageProcessor.load_image', 
                      side_effect=Exception("加载失败")):
                with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_critical:
                    window.load_image("invalid_path.jpg")
                    mock_critical.assert_called()
            
            # 测试分析失败处理
            error_message = "分析过程中发生错误"
            window.on_analysis_failed(error_message)
            
            # 验证状态恢复
            assert window.analyze_action.isEnabled()
            assert window.select_file_action.isEnabled()
            assert not window.progress_bar.isVisible()
            
            window.close()
    
    def test_statistics_display_integration(self, qapp, sample_statistics, sample_analysis_result):
        """测试统计信息显示集成"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 设置分析结果
            window.analysis_result = sample_analysis_result
            window.current_image = np.zeros((100, 100, 3))
            
            # 模拟当前模型信息
            with patch.object(window.analysis_controller, 'get_current_models', 
                            return_value={'vehicle_model': 'yolov11', 'water_model': 'deeplabv3'}):
                
                # 更新统计信息
                window.update_statistics()
                
                # 验证统计信息更新
                # 这里主要验证方法调用不会出错
                assert window.analysis_result is not None
            
            window.close()
    
    def test_progress_tracking(self, qapp):
        """测试进度跟踪功能"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 测试进度更新
            window.on_analysis_progress(50, "正在处理...")
            assert window.progress_bar.value() == 50
            
            # 测试状态更新
            test_message = "测试状态消息"
            window.update_status(test_message)
            assert window.status_label.text() == test_message
            
            window.close()


class TestAnalysisController:
    """分析控制器集成测试"""
    
    def test_analysis_controller_initialization(self, qapp):
        """测试分析控制器初始化"""
        controller = AnalysisController()
        
        # 验证组件初始化
        assert hasattr(controller, 'model_manager')
        assert hasattr(controller, 'image_processor')
        assert hasattr(controller, 'flood_analyzer')
        assert hasattr(controller, 'viz_engine')
        assert hasattr(controller, 'model_selection')
        
        # 验证初始状态
        assert not controller.is_analyzing
        assert controller.current_image is None
    
    def test_model_management(self, qapp, mock_model_manager):
        """测试模型管理功能"""
        with patch('flood_detection_app.desktop.analysis_controller.ModelManager', return_value=mock_model_manager):
            controller = AnalysisController()
            
            # 测试模型加载
            result = controller.load_models()
            assert result == True
            
            # 测试获取可用模型
            models = controller.get_available_models()
            assert 'vehicle_models' in models
            assert 'water_models' in models
            
            # 测试模型设置
            assert controller.set_vehicle_model('yolov11_car_detection')
            assert controller.set_water_model('deeplabv3_water')
    
    def test_analysis_workflow(self, qapp, sample_image):
        """测试分析工作流程"""
        with patch('flood_detection_app.desktop.analysis_controller.ModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.load_models.return_value = True
            mock_manager.get_available_models.return_value = {
                'vehicle_models': ['yolov11'],
                'water_models': ['deeplabv3']
            }
            mock_manager_class.return_value = mock_manager
            
            controller = AnalysisController()
            controller.set_image(sample_image)
            controller.set_vehicle_model('yolov11')
            controller.set_water_model('deeplabv3')
            
            # 测试分析前检查
            can_start, message = controller.can_start_analysis()
            assert can_start == True
            
            # 测试分析状态
            status = controller.get_analysis_status()
            assert status['has_image'] == True
            assert status['models_ready'] == True
            assert status['is_analyzing'] == False


class TestFileOperations:
    """文件操作集成测试"""
    
    def test_file_operations_initialization(self, qapp):
        """测试文件操作初始化"""
        file_ops = FileOperations()
        
        # 验证初始化
        assert hasattr(file_ops, 'last_directory')
        assert os.path.exists(file_ops.last_directory)
    
    def test_image_validation(self, qapp, test_image_file):
        """测试图像文件验证"""
        file_ops = FileOperations()
        
        # 测试有效图像文件
        assert file_ops._validate_image_file(test_image_file) == True
        
        # 测试无效文件路径
        assert file_ops._validate_image_file("nonexistent.jpg") == False
    
    def test_file_filter_building(self, qapp):
        """测试文件过滤器构建"""
        file_ops = FileOperations()
        
        supported_formats = ['.jpg', '.png', '.bmp']
        filter_str = file_ops._build_image_filter(supported_formats)
        
        # 验证过滤器包含支持的格式
        assert 'jpg' in filter_str
        assert 'png' in filter_str
        assert 'bmp' in filter_str
        assert '所有支持的图像' in filter_str


class TestUserInterfaceUsability:
    """用户界面易用性测试"""
    
    def test_window_responsiveness(self, qapp):
        """测试窗口响应性"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 测试窗口大小调整
            original_size = window.size()
            window.resize(1800, 1200)
            QTest.qWait(100)  # 等待UI更新
            
            new_size = window.size()
            assert new_size.width() >= 1800
            assert new_size.height() >= 1200
            
            window.close()
    
    def test_keyboard_shortcuts(self, qapp):
        """测试键盘快捷键（如果有的话）"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 验证动作存在（可以扩展添加快捷键测试）
            assert window.select_file_action is not None
            assert window.analyze_action is not None
            assert window.save_action is not None
            
            window.close()
    
    def test_tooltip_and_status_messages(self, qapp):
        """测试工具提示和状态消息"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 验证状态提示存在
            assert window.select_file_action.statusTip() != ""
            assert window.analyze_action.statusTip() != ""
            assert window.save_action.statusTip() != ""
            
            # 测试状态消息更新
            test_message = "测试状态消息"
            window.update_status(test_message)
            assert window.status_label.text() == test_message
            
            window.close()
    
    def test_error_message_display(self, qapp):
        """测试错误消息显示"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 测试错误处理不会崩溃应用
            with patch('PyQt6.QtWidgets.QMessageBox.critical') as mock_critical:
                window.on_analysis_failed("测试错误消息")
                # 验证错误消息会被显示（通过mock验证）
                # 实际的消息框显示由信号处理
            
            window.close()


class TestDataFlowIntegration:
    """数据流集成测试"""
    
    def test_image_data_flow(self, qapp, sample_image):
        """测试图像数据流"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 设置图像
            window.current_image = sample_image
            window.analysis_controller.set_image(sample_image)
            
            # 验证数据传递
            assert window.current_image is not None
            assert window.analysis_controller.current_image is not None
            assert np.array_equal(window.current_image, window.analysis_controller.current_image)
            
            window.close()
    
    def test_result_data_flow(self, qapp, sample_analysis_result, sample_image):
        """测试结果数据流"""
        with patch('flood_detection_app.desktop.main_window.ModelManager'):
            window = MainWindow()
            
            # 模拟分析完成
            result_data = {
                'analysis_result': sample_analysis_result,
                'result_image': sample_image
            }
            
            window.on_analysis_completed(result_data)
            
            # 验证结果数据传递
            assert window.analysis_result == sample_analysis_result
            
            window.close()