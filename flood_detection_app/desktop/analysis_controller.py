"""
分析控制器
Analysis Controller for managing model selection and analysis process
"""

import time
from typing import Optional, Dict, Any, Callable
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from ..core import ModelManager, ImageProcessor, FloodAnalyzer, VisualizationEngine
from ..core.data_models import AnalysisResult
from ..core.exceptions import FloodDetectionError, ModelLoadError, InferenceError
import numpy as np


class AnalysisWorker(QThread):
    """分析工作线程"""
    
    # 信号定义
    progress_updated = pyqtSignal(int, str)  # 进度值, 状态消息
    analysis_completed = pyqtSignal(object)  # 分析结果
    analysis_failed = pyqtSignal(str)        # 错误消息
    
    def __init__(self, 
                 image: np.ndarray,
                 model_manager: ModelManager,
                 flood_analyzer: FloodAnalyzer,
                 viz_engine: VisualizationEngine,
                 vehicle_model: str,
                 water_model: str):
        super().__init__()
        
        self.image = image
        self.model_manager = model_manager
        self.flood_analyzer = flood_analyzer
        self.viz_engine = viz_engine
        self.vehicle_model = vehicle_model
        self.water_model = water_model
        
        self._is_cancelled = False
    
    def run(self):
        """执行分析"""
        try:
            # 确定任务模式
            has_vehicle_model = self.vehicle_model is not None
            has_water_model = self.water_model is not None
            
            vehicles = []
            water_mask = None
            
            # 1. 车辆检测（如果启用）
            if has_vehicle_model:
                self.progress_updated.emit(20, "Setting vehicle detection model...")
                self.model_manager.set_active_models(self.vehicle_model, None)
                
                if self._is_cancelled:
                    return
                
                self.progress_updated.emit(30, "Detecting vehicles...")
                vehicles = self.model_manager.predict_vehicles(self.image)
                
                if self._is_cancelled:
                    return
            
            # 2. 水面分割（如果启用）
            if has_water_model:
                self.progress_updated.emit(50, "Setting water segmentation model...")
                self.model_manager.set_active_models(None, self.water_model)
                
                if self._is_cancelled:
                    return
                
                self.progress_updated.emit(60, "Segmenting water...")
                water_mask = self.model_manager.predict_water(self.image)
                
                if self._is_cancelled:
                    return
            
            # 3. 淹没分析（仅当两个模型都启用时）
            if has_vehicle_model and has_water_model:
                self.progress_updated.emit(80, "Analyzing flood levels...")
                analysis_result = self.flood_analyzer.analyze_scene(vehicles, water_mask)
            else:
                # 创建简化的分析结果
                from ..core.data_models import Statistics, AnalysisResult
                import time
                
                # 确保变量都已初始化
                if 'vehicles' not in locals():
                    vehicles = []
                if vehicles is None:
                    vehicles = []
                
                # 如果没有水面掩码，创建空掩码
                if water_mask is None:
                    water_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                
                # 计算真实的水面覆盖率
                water_coverage_percentage = 0.0
                if water_mask is not None and water_mask.size > 0:
                    water_pixels = np.sum(water_mask > 0)
                    total_pixels = water_mask.size
                    water_coverage_percentage = (water_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
                
                # 首先创建车辆结果（确保在所有情况下都被初始化）
                vehicle_results = []
                if has_vehicle_model and vehicles:
                    from ..core.data_models import VehicleResult, FloodLevel
                    
                    if has_water_model and water_mask is not None:
                        # 🔥 修复：有水面掩码时，仍然使用模型预测的淹没等级
                        for i, v in enumerate(vehicles):
                            # 使用模型预测的淹没等级
                            predicted_flood_level = self._extract_flood_level_from_detection(v)
                            
                            # 计算与水面的重叠比例
                            center_x = int((v.bbox.x1 + v.bbox.x2) / 2)
                            center_y = int((v.bbox.y1 + v.bbox.y2) / 2)
                            
                            # 确保坐标在掩码范围内
                            if (0 <= center_y < water_mask.shape[0] and 
                                0 <= center_x < water_mask.shape[1]):
                                is_in_water = water_mask[center_y, center_x] > 0
                                # 简化的重叠比例计算
                                overlap_ratio = 0.5 if is_in_water else 0.0
                            else:
                                overlap_ratio = 0.0
                            
                            vehicle_results.append(VehicleResult(
                                detection=v,
                                flood_level=predicted_flood_level,  # 🔥 使用模型预测的等级
                                overlap_ratio=overlap_ratio,
                                vehicle_id=i
                            ))
                    else:
                        # 只有车辆检测，使用模型预测的淹没等级
                        for i, v in enumerate(vehicles):
                            # 从检测结果中提取淹没等级
                            predicted_flood_level = self._extract_flood_level_from_detection(v)
                            
                            vehicle_results.append(VehicleResult(
                                detection=v,
                                flood_level=predicted_flood_level,
                                overlap_ratio=0.0,  # 没有水面信息时设为0
                                vehicle_id=i
                            ))
                
                # 现在计算淹没车辆统计（在vehicle_results创建之后）
                # 添加安全检查
                if 'vehicle_results' not in locals() or vehicle_results is None:
                    vehicle_results = []
                
                try:
                    # 🔥 修复统计计算 - 确保正确计算各级别车辆数量
                    wheel_count = 0
                    window_count = 0
                    roof_count = 0
                    
                    for vr in vehicle_results:
                        if hasattr(vr, 'flood_level') and vr.flood_level:
                            level_name = vr.flood_level.name if hasattr(vr.flood_level, 'name') else str(vr.flood_level.value)
                            if level_name == 'WHEEL_LEVEL':
                                wheel_count += 1
                            elif level_name == 'WINDOW_LEVEL':
                                window_count += 1
                            elif level_name == 'ROOF_LEVEL':
                                roof_count += 1
                    
                    print(f"🔍 统计计算结果: 车轮级={wheel_count}, 车窗级={window_count}, 车顶级={roof_count}")
                    
                except Exception as e:
                    print(f"警告: 统计计算失败: {e}")
                    wheel_count = window_count = roof_count = 0
                
                # 创建基本统计信息
                stats = Statistics(
                    total_vehicles=len(vehicles) if (has_vehicle_model and vehicles) else 0,
                    wheel_level_count=wheel_count,
                    window_level_count=window_count,
                    roof_level_count=roof_count,
                    water_coverage_percentage=water_coverage_percentage,
                    processing_time=0.0
                )
                
                # 🔍 调试信息
                print(f"🔍 创建分析结果:")
                print(f"  - 车辆结果数量: {len(vehicle_results)}")
                print(f"  - 统计信息: 总车辆={stats.total_vehicles}, 轮级={stats.wheel_level_count}, 窗级={stats.window_level_count}, 顶级={stats.roof_level_count}")
                print(f"  - 水面覆盖率: {stats.water_coverage_percentage:.2f}%")
                
                analysis_result = AnalysisResult(
                    vehicles=vehicle_results,
                    water_mask=water_mask,
                    statistics=stats,
                    original_image_shape=self.image.shape[:2]
                )
            
            if self._is_cancelled:
                return
            
            # 4. 生成结果图像
            self.progress_updated.emit(90, "Generating result image...")
            result_image = self.viz_engine.create_result_image(
                self.image, 
                analysis_result
            )
            
            # 5. 完成
            self.progress_updated.emit(100, "Analysis completed")
            
            # 返回结果
            result_data = {
                'analysis_result': analysis_result,
                'result_image': result_image,
                'original_image': self.image
            }
            
            self.analysis_completed.emit(result_data)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.analysis_failed.emit(error_msg)
    
    def cancel(self):
        """取消分析"""
        self._is_cancelled = True
    
    def _extract_flood_level_from_detection(self, detection):
        """从检测结果中提取淹没等级"""
        from ..core.data_models import FloodLevel
        
        # 🔍 调试信息
        print(f"🔍 提取淹没等级: class_id={detection.class_id}, class_name={getattr(detection, 'class_name', 'unknown')}")
        
        # 根据检测的class_id映射到FloodLevel
        class_id = detection.class_id
        
        # 🔥 修正映射关系 - 根据实际模型输出调整
        # 如果模型输出的class_id都是0，说明模型可能只检测车辆而不区分淹没等级
        # 在这种情况下，我们应该使用默认的轻度淹没等级
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
        
        result = flood_level_mapping.get(class_id, FloodLevel.WHEEL_LEVEL)
        print(f"🔍 映射结果: {result}")
        return result


class ModelSelectionManager:
    """模型选择管理器"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.available_models = {}
        self.current_selection = {
            'vehicle_model': None,
            'water_model': None
        }
        
        self.refresh_available_models()
    
    def refresh_available_models(self):
        """刷新可用模型列表"""
        try:
            self.available_models = self.model_manager.get_available_models()
            
            # 设置默认选择
            if (self.available_models['vehicle_models'] and 
                not self.current_selection['vehicle_model']):
                self.current_selection['vehicle_model'] = self.available_models['vehicle_models'][0]
            
            if (self.available_models['water_models'] and 
                not self.current_selection['water_model']):
                self.current_selection['water_model'] = self.available_models['water_models'][0]
                
        except Exception as e:
            print(f"刷新模型列表失败: {e}")
    
    def get_vehicle_models(self) -> list:
        """获取车辆检测模型列表"""
        return self.available_models.get('vehicle_models', [])
    
    def get_water_models(self) -> list:
        """获取水面分割模型列表"""
        return self.available_models.get('water_models', [])
    
    def set_vehicle_model(self, model_name: str) -> bool:
        """设置车辆检测模型"""
        if model_name in self.get_vehicle_models():
            self.current_selection['vehicle_model'] = model_name
            return True
        return False
    
    def set_water_model(self, model_name: str) -> bool:
        """设置水面分割模型"""
        if model_name in self.get_water_models():
            self.current_selection['water_model'] = model_name
            return True
        return False
    
    def get_current_selection(self) -> Dict[str, str]:
        """获取当前选择的模型"""
        return self.current_selection.copy()
    
    def is_selection_valid(self) -> bool:
        """检查当前选择是否有效"""
        return (self.current_selection['vehicle_model'] is not None and 
                self.current_selection['water_model'] is not None)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        # 这里可以扩展返回更详细的模型信息
        return {
            'name': model_name,
            'available': model_name in (self.get_vehicle_models() + self.get_water_models())
        }


class AnalysisController(QObject):
    """分析控制器"""
    
    # 信号定义
    analysis_started = pyqtSignal()
    analysis_progress = pyqtSignal(int, str)  # 进度, 消息
    analysis_completed = pyqtSignal(object)   # 结果数据
    analysis_failed = pyqtSignal(str)         # 错误消息
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        
        self.parent = parent
        
        # 核心组件
        self.model_manager = ModelManager()
        self.image_processor = ImageProcessor()
        self.flood_analyzer = FloodAnalyzer()
        self.viz_engine = VisualizationEngine()
        
        # 管理器
        self.model_selection = ModelSelectionManager(self.model_manager)
        
        # 工作线程
        self.analysis_worker = None
        self.progress_dialog = None
        
        # 状态
        self.is_analyzing = False
        self.current_image = None
    
    def load_models(self) -> bool:
        """加载模型"""
        try:
            success = self.model_manager.load_models()
            self.model_selection.refresh_available_models()
            return success
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def set_image(self, image: np.ndarray):
        """设置要分析的图像"""
        self.current_image = image
    
    def get_available_models(self) -> Dict[str, list]:
        """获取可用模型"""
        return {
            'vehicle_models': self.model_selection.get_vehicle_models(),
            'water_models': self.model_selection.get_water_models()
        }
    
    def set_vehicle_model(self, model_name: Optional[str]) -> bool:
        """设置车辆检测模型"""
        if model_name is None:
            self.model_selection.current_selection['vehicle_model'] = None
            return True
        return self.model_selection.set_vehicle_model(model_name)
    
    def set_water_model(self, model_name: Optional[str]) -> bool:
        """设置水面分割模型"""
        if model_name is None:
            self.model_selection.current_selection['water_model'] = None
            return True
        return self.model_selection.set_water_model(model_name)
    
    def get_current_models(self) -> Dict[str, str]:
        """获取当前选择的模型"""
        return self.model_selection.get_current_selection()
    
    def can_start_analysis(self):
        """检查是否可以开始分析"""
        if self.is_analyzing:
            return False, "Analysis in progress, please wait"
        
        if self.current_image is None:
            return False, "Please select an image to analyze"
        
        # 至少需要一个模型
        current_models = self.model_selection.get_current_selection()
        if not current_models['vehicle_model'] and not current_models['water_model']:
            return False, "Please select at least one model"
        
        return True, "Ready to analyze"
    
    def start_analysis(self) -> bool:
        """开始分析"""
        # 检查是否可以开始
        can_start, message = self.can_start_analysis()
        if not can_start:
            if self.parent:
                QMessageBox.warning(self.parent, "无法开始分析", message)
            return False
        
        try:
            # 获取当前选择的模型
            current_models = self.get_current_models()
            
            # 创建工作线程
            self.analysis_worker = AnalysisWorker(
                image=self.current_image,
                model_manager=self.model_manager,
                flood_analyzer=self.flood_analyzer,
                viz_engine=self.viz_engine,
                vehicle_model=current_models['vehicle_model'],
                water_model=current_models['water_model']
            )
            
            # 连接信号
            self.analysis_worker.progress_updated.connect(self._on_progress_updated)
            self.analysis_worker.analysis_completed.connect(self._on_analysis_completed)
            self.analysis_worker.analysis_failed.connect(self._on_analysis_failed)
            
            # 创建进度对话框
            self._create_progress_dialog()
            
            # 设置状态
            self.is_analyzing = True
            
            # 发送开始信号
            self.analysis_started.emit()
            
            # 启动线程
            self.analysis_worker.start()
            
            return True
            
        except Exception as e:
            error_msg = f"启动分析失败: {str(e)}"
            if self.parent:
                QMessageBox.critical(self.parent, "分析错误", error_msg)
            return False
    
    def cancel_analysis(self):
        """取消分析"""
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.cancel()
            self.analysis_worker.wait(3000)  # 等待3秒
            
        self._cleanup_analysis()
    
    def _create_progress_dialog(self):
        """创建进度对话框"""
        if self.parent:
            self.progress_dialog = QProgressDialog(
                "正在初始化分析...", 
                "取消", 
                0, 100, 
                self.parent
            )
            self.progress_dialog.setWindowTitle("图像分析进度")
            self.progress_dialog.setModal(True)
            self.progress_dialog.canceled.connect(self.cancel_analysis)
            self.progress_dialog.show()
    
    def _on_progress_updated(self, value: int, message: str):
        """进度更新处理"""
        if self.progress_dialog and hasattr(self.progress_dialog, 'setValue'):
            try:
                self.progress_dialog.setValue(value)
                self.progress_dialog.setLabelText(message)
            except Exception as e:
                print(f"进度对话框更新失败: {e}")
        
        # 发送进度信号
        self.analysis_progress.emit(value, message)
    
    def _on_analysis_completed(self, result_data: Dict[str, Any]):
        """分析完成处理"""
        self._cleanup_analysis()
        
        # 发送完成信号
        self.analysis_completed.emit(result_data)
    
    def _on_analysis_failed(self, error_message: str):
        """分析失败处理"""
        self._cleanup_analysis()
        
        # 显示错误消息
        if self.parent:
            QMessageBox.critical(self.parent, "分析失败", error_message)
        
        # 发送失败信号
        self.analysis_failed.emit(error_message)
    
    def _cleanup_analysis(self):
        """清理分析资源"""
        self.is_analyzing = False
        
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if self.analysis_worker:
            if self.analysis_worker.isRunning():
                self.analysis_worker.quit()
                self.analysis_worker.wait()
            self.analysis_worker = None
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """获取分析状态"""
        return {
            'is_analyzing': self.is_analyzing,
            'has_image': self.current_image is not None,
            'models_ready': self.model_selection.is_selection_valid(),
            'current_models': self.get_current_models()
        }