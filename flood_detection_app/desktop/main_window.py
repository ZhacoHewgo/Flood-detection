"""
主窗口
Main Window for Desktop GUI Application
"""

import sys
import os
import time
import numpy as np
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QStatusBar, QLabel, QProgressBar,
    QComboBox, QPushButton, QFileDialog, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QIcon, QPixmap, QMouseEvent


class ClickableFrame(QFrame):
    """可点击的Frame"""
    clicked = pyqtSignal()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

# 导入桌面组件
from .image_display_widget import ImageDisplayPanel
from .file_operations import FileOperations
from .analysis_controller import AnalysisController
from .statistics_widget import StatisticsWidget, CompactStatisticsWidget

# 导入核心模块
from ..core import (
    ModelManager, ImageProcessor, FloodAnalyzer, VisualizationEngine,
    config_manager
)
from ..core.exceptions import FloodDetectionError


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 桌面组件
        self.file_operations = FileOperations(self)
        self.analysis_controller = AnalysisController(self)
        
        # 状态变量
        self.current_image = None
        self.current_image_path = None
        self.analysis_result = None
        
        # 🔥 分析结果缓存 - 为每张图片保存分析结果
        self.analysis_cache = {}  # {file_path: {result, result_image, timestamp}}
        
        # 初始化UI
        self.setup_ui()
        self.setup_toolbar()
        self.setup_status_bar()
        
        # 连接信号（在UI创建之后）
        self.connect_signals()
        
        # 加载模型
        self.load_models()
        
        # 设置窗口属性
        self.setWindowTitle("Flood Vehicle Detection System")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        
        # 设置窗口居中显示
        self.center_window()
    
    def setup_ui(self):
        """设置现代化深色主题用户界面"""
        # 设置深色主题样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1d23;
                color: #ffffff;
            }
            QWidget {
                background-color: #1a1d23;
                color: #ffffff;
            }
            QSplitter::handle {
                background-color: #2d3748;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建左侧面板
        self.setup_left_panel()
        main_layout.addWidget(self.left_panel, 1)
        
        # 创建右侧面板
        self.setup_right_panel()
        main_layout.addWidget(self.right_panel, 2)
    
    def setup_left_panel(self):
        """设置左侧面板 - 上传区域"""
        self.left_panel = QWidget()
        self.left_panel.setStyleSheet("""
            QWidget {
                background-color: #252a31;
                border-right: 1px solid #3a4149;
            }
        """)
        self.left_panel.setMinimumWidth(350)
        self.left_panel.setMaximumWidth(450)
        
        layout = QVBoxLayout(self.left_panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # 标题区域
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(8)
        
        # Logo和标题行
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        
        logo_label = QLabel("🌊")
        logo_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                color: #4fc3f7;
                background-color: #2d5a87;
                border-radius: 18px;
                padding: 6px;
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
            }
        """)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(logo_label)
        
        title_text = QLabel("积水识别和车辆淹没部位判别系统")
        title_text.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        header_layout.addWidget(title_text)
        header_layout.addStretch()
        
        title_layout.addLayout(header_layout)
        
        # 副标题
        subtitle = QLabel("Vehicle Analysis System")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #8a9ba8;
                margin-left: 46px;
            }
        """)
        title_layout.addWidget(subtitle)
        
        layout.addWidget(title_container)
        
        # 上传区域
        self.setup_upload_area(layout)
        
        # 文件列表区域
        self.setup_file_list_area(layout)
        
        # 模型选择区域
        self.setup_model_selection_area(layout)
    
    def setup_upload_area(self, parent_layout):
        """设置上传区域"""
        upload_frame = QFrame()
        upload_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1d23;
                border: 2px dashed #4a5568;
                border-radius: 10px;
            }
            QFrame:hover {
                border-color: #4fc3f7;
                background-color: #1e2329;
            }
        """)
        upload_frame.setMinimumHeight(180)
        upload_frame.setMaximumHeight(220)
        
        upload_layout = QVBoxLayout(upload_frame)
        upload_layout.setContentsMargins(15, 15, 15, 15)
        upload_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.setSpacing(10)
        
        # 拖拽图标
        drag_icon = QLabel("📁")
        drag_icon.setStyleSheet("""
            QLabel {
                font-size: 40px;
                color: #4a5568;
            }
        """)
        drag_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upload_layout.addWidget(drag_icon)
        
        # 拖拽文本
        drag_text = QLabel("Drag & Drop image here\nor click to browse")
        drag_text.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #8a9ba8;
                font-weight: 500;
            }
        """)
        drag_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drag_text.setWordWrap(True)
        upload_layout.addWidget(drag_text)
        
        # 上传按钮容器
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        # 单张上传按钮
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4fc3f7;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #29b6f6;
            }
            QPushButton:pressed {
                background-color: #0288d1;
            }
        """)
        self.upload_btn.clicked.connect(self.select_image_file)
        button_layout.addWidget(self.upload_btn)
        
        # 批量上传按钮
        self.batch_upload_btn = QPushButton("Batch Upload")
        self.batch_upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #66bb6a;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4caf50;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
        """)
        self.batch_upload_btn.clicked.connect(self.select_multiple_images)
        button_layout.addWidget(self.batch_upload_btn)
        
        upload_layout.addLayout(button_layout)
        
        parent_layout.addWidget(upload_frame)
    
    def setup_file_list_area(self, parent_layout):
        """设置文件列表区域"""
        # 文件列表容器
        file_list_container = QWidget()
        file_list_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        file_list_container.setMaximumHeight(200)  # 增加高度从150到200
        
        container_layout = QVBoxLayout(file_list_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(5)
        
        # 滚动区域
        from PyQt6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background-color: #1a1d23;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a5568;
                border-radius: 4px;
            }
        """)
        
        self.file_list_widget = QWidget()
        self.file_list_widget.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        
        file_list_layout = QVBoxLayout(self.file_list_widget)
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.setSpacing(5)
        file_list_layout.addStretch()
        
        scroll_area.setWidget(self.file_list_widget)
        container_layout.addWidget(scroll_area)
        
        parent_layout.addWidget(file_list_container)
    
    def setup_model_selection_area(self, parent_layout):
        """设置模型选择区域"""
        model_frame = QFrame()
        model_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1d23;
                border: 1px solid #4a5568;
                border-radius: 8px;
            }
        """)
        
        model_layout = QVBoxLayout(model_frame)
        model_layout.setContentsMargins(12, 12, 12, 12)
        model_layout.setSpacing(12)
        
        # 任务模式选择
        mode_label = QLabel("Task Mode:")
        mode_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        model_layout.addWidget(mode_label)
        
        self.task_mode_combo = QComboBox()
        self.task_mode_combo.addItems([
            "Combined Analysis",
            "Vehicle Detection Only", 
            "Water Segmentation Only"
        ])
        self.task_mode_combo.setStyleSheet(self._get_combo_style())
        self.task_mode_combo.currentTextChanged.connect(self.on_task_mode_changed)
        model_layout.addWidget(self.task_mode_combo)
        
        # 车辆检测模型选择
        vehicle_label = QLabel("Vehicle Detection:")
        vehicle_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #8a9ba8;
                margin-top: 5px;
            }
        """)
        model_layout.addWidget(vehicle_label)
        
        self.vehicle_model_combo = QComboBox()
        self.vehicle_model_combo.setStyleSheet(self._get_combo_style())
        model_layout.addWidget(self.vehicle_model_combo)
        
        # 水面分割模型选择
        water_label = QLabel("Water Segmentation:")
        water_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #8a9ba8;
                margin-top: 5px;
            }
        """)
        model_layout.addWidget(water_label)
        
        self.water_model_combo = QComboBox()
        self.water_model_combo.setStyleSheet(self._get_combo_style())
        model_layout.addWidget(self.water_model_combo)
        
        parent_layout.addWidget(model_frame)
        
        # 操作按钮区域
        button_container = QWidget()
        button_container.setStyleSheet("QWidget { background-color: transparent; }")
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 0)
        button_layout.setSpacing(8)
        
        # 分析按钮
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4fc3f7;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #29b6f6;
            }
            QPushButton:disabled {
                background-color: #4a5568;
                color: #8a9ba8;
            }
        """)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.analyze_btn)
        
        # 批量分析按钮
        self.batch_analyze_btn = QPushButton("Batch Analyze")
        self.batch_analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
            QPushButton:disabled {
                background-color: #4a5568;
                color: #8a9ba8;
            }
        """)
        self.batch_analyze_btn.setEnabled(False)
        self.batch_analyze_btn.clicked.connect(self.start_batch_analysis)
        button_layout.addWidget(self.batch_analyze_btn)
        
        # 🔥 添加全选/取消全选按钮
        select_all_layout = QHBoxLayout()
        select_all_layout.setSpacing(5)
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        self.select_all_btn.clicked.connect(self.select_all_files)
        select_all_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.deselect_all_btn.clicked.connect(self.deselect_all_files)
        select_all_layout.addWidget(self.deselect_all_btn)
        
        button_layout.addLayout(select_all_layout)
        
        # 按钮行
        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        
        # 清除按钮
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #f44336;
                color: #f44336;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #f44336;
                color: #ffffff;
            }
        """)
        button_row.addWidget(self.clear_btn)
        
        # 下载结果按钮 (支持批量下载)
        self.download_btn = QPushButton("Download Results")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #4fc3f7;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #29b6f6;
            }
            QPushButton:disabled {
                background-color: #4a5568;
                color: #8a9ba8;
            }
        """)
        self.download_btn.setEnabled(False)
        button_row.addWidget(self.download_btn)
        
        button_layout.addLayout(button_row)
        parent_layout.addWidget(button_container)
        
        # 添加弹性空间
        parent_layout.addStretch()
    
    def _get_combo_style(self):
        """获取下拉框统一样式"""
        return """
            QComboBox {
                background-color: #252a31;
                border: 1px solid #4a5568;
                border-radius: 6px;
                padding: 8px 10px;
                color: #ffffff;
                font-size: 11px;
                min-height: 16px;
            }
            QComboBox:hover {
                border-color: #4fc3f7;
            }
            QComboBox:disabled {
                background-color: #1a1d23;
                color: #4a5568;
                border-color: #3a4149;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #8a9ba8;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #252a31;
                border: 1px solid #4a5568;
                selection-background-color: #4fc3f7;
                color: #ffffff;
            }
        """
    
    def on_task_mode_changed(self, mode: str):
        """任务模式改变处理"""
        if mode == "Vehicle Detection Only":
            self.vehicle_model_combo.setEnabled(True)
            self.water_model_combo.setEnabled(False)
        elif mode == "Water Segmentation Only":
            self.vehicle_model_combo.setEnabled(False)
            self.water_model_combo.setEnabled(True)
        else:  # Combined Analysis
            self.vehicle_model_combo.setEnabled(True)
            self.water_model_combo.setEnabled(True)
    
    def setup_right_panel(self):
        """设置右侧面板 - 结果显示区域"""
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
            }
        """)
        
        layout = QVBoxLayout(self.right_panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 图像显示区域（移除了标签页）
        self.setup_image_display_area(layout)
    

    
    def setup_image_display_area(self, parent_layout):
        """设置图像显示区域 - 左右分屏"""
        # 图像容器
        image_container = QWidget()
        image_container.setStyleSheet("""
            QWidget {
                background-color: #252a31;
                border-radius: 10px;
            }
        """)
        
        container_layout = QVBoxLayout(image_container)
        container_layout.setContentsMargins(15, 15, 15, 15)
        container_layout.setSpacing(12)
        
        # 🔥 创建左右分屏的图像显示区域
        image_splitter = QSplitter(Qt.Orientation.Horizontal)
        image_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #4a5568;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #4fc3f7;
            }
        """)
        
        # 左侧：原图显示
        left_panel = QWidget()
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(5)
        
        # 原图标题
        original_title = QLabel("Original Image")
        original_title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(original_title)
        
        # 原图显示面板
        self.original_image_panel = ImageDisplayPanel("Original")
        self.original_image_panel.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border: 1px solid #4a5568;
                border-radius: 6px;
            }
        """)
        left_layout.addWidget(self.original_image_panel, 1)
        
        # 右侧：分析结果显示
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border-radius: 8px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(5)
        
        # 结果标题
        result_title = QLabel("Analysis Result")
        result_title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(result_title)
        
        # 分析结果显示面板
        self.result_image_panel = ImageDisplayPanel("Result")
        self.result_image_panel.setStyleSheet("""
            QWidget {
                background-color: #1a1d23;
                border: 1px solid #4a5568;
                border-radius: 6px;
            }
        """)
        right_layout.addWidget(self.result_image_panel, 1)
        
        # 添加到分割器
        image_splitter.addWidget(left_panel)
        image_splitter.addWidget(right_panel)
        
        # 设置分割比例（左右各50%）
        image_splitter.setSizes([1, 1])
        
        container_layout.addWidget(image_splitter, 1)
        
        # 分析摘要面板
        self.setup_analysis_summary(container_layout)
        
        # 🔥 移除图像控制按钮（放大、缩小、适应窗口）
        # self.setup_image_controls(container_layout)
        
        parent_layout.addWidget(image_container, 1)
    
    def setup_analysis_summary(self, parent_layout):
        """设置分析摘要面板"""
        summary_frame = QFrame()
        summary_frame.setStyleSheet("""
            QFrame {
                background-color: #1e2329;
                border: 1px solid #3a4149;
                border-radius: 8px;
            }
        """)
        
        summary_layout = QVBoxLayout(summary_frame)
        summary_layout.setContentsMargins(12, 10, 12, 10)
        summary_layout.setSpacing(8)
        
        # 分析摘要标题
        summary_title = QLabel("Analysis Summary")
        summary_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        summary_layout.addWidget(summary_title)
        
        # 统计数据行
        stats_row = QHBoxLayout()
        stats_row.setSpacing(20)
        
        # 检测到的对象数量
        objects_container = QWidget()
        objects_layout = QHBoxLayout(objects_container)
        objects_layout.setContentsMargins(0, 0, 0, 0)
        objects_layout.setSpacing(8)
        
        objects_text = QLabel("Objects:")
        objects_text.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #8a9ba8;
            }
        """)
        self.objects_count_label = QLabel("0")
        self.objects_count_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #4fc3f7;
            }
        """)
        objects_layout.addWidget(objects_text)
        objects_layout.addWidget(self.objects_count_label)
        stats_row.addWidget(objects_container)
        
        # 水位等级
        water_container = QWidget()
        water_layout = QHBoxLayout(water_container)
        water_layout.setContentsMargins(0, 0, 0, 0)
        water_layout.setSpacing(8)
        
        water_text = QLabel("Water Level:")
        water_text.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #8a9ba8;
            }
        """)
        self.water_level_label = QLabel("--")
        self.water_level_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #8a9ba8;
            }
        """)
        water_layout.addWidget(water_text)
        water_layout.addWidget(self.water_level_label)
        stats_row.addWidget(water_container)
        
        stats_row.addStretch()
        summary_layout.addLayout(stats_row)
        
        parent_layout.addWidget(summary_frame)
    
    # 🔥 已移除图像控制按钮方法
    

    
    def clear_all(self):
        """清除所有内容"""
        self.clear_result()
        self.current_image = None
        self.current_image_path = None
        self.analyze_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        
        # 清除文件列表
        for i in reversed(range(self.file_list_widget.layout().count())):
            child = self.file_list_widget.layout().itemAt(i).widget()
            if child:
                child.setParent(None)
    
    def add_file_to_list(self, file_path: str):
        """添加文件到列表"""
        import os
        
        # 🔥 检查文件是否已经存在，避免重复添加
        existing_files = self.get_uploaded_file_paths()
        if file_path in existing_files:
            print(f"文件已存在，跳过添加: {file_path}")
            return
        
        file_item = ClickableFrame()
        file_item.setStyleSheet("""
            QFrame {
                background-color: #1a1d23;
                border: 1px solid #4a5568;
                border-radius: 6px;
            }
            QFrame:hover {
                background-color: #2d3748;
                border-color: #4fc3f7;
            }
        """)
        file_item.setMaximumHeight(60)
        file_item.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # 🔥 存储文件路径，用于批量分析
        file_item.file_path = file_path
        
        # 连接点击事件
        file_item.clicked.connect(lambda: self.load_image_from_path(file_path))
        
        item_layout = QHBoxLayout(file_item)
        item_layout.setContentsMargins(8, 6, 8, 6)
        item_layout.setSpacing(8)
        
        # 🔥 添加选择框
        from PyQt6.QtWidgets import QCheckBox
        checkbox = QCheckBox()
        checkbox.setStyleSheet("""
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #4a5568;
                background-color: transparent;
            }
            QCheckBox::indicator:checked {
                background-color: #4fc3f7;
                border-color: #4fc3f7;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #29b6f6;
            }
        """)
        checkbox.setChecked(True)  # 默认选中
        checkbox.stateChanged.connect(self.update_batch_analyze_button)
        item_layout.addWidget(checkbox)
        
        # 存储checkbox引用
        file_item.checkbox = checkbox
        
        # 文件图标
        file_icon = QLabel("📷")
        file_icon.setStyleSheet("font-size: 14px;")
        item_layout.addWidget(file_icon)
        
        # 文件信息
        file_info_layout = QVBoxLayout()
        file_info_layout.setSpacing(3)
        
        file_name = QLabel(os.path.basename(file_path)[:25] + "..." if len(os.path.basename(file_path)) > 25 else os.path.basename(file_path))
        file_name.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 11px;
                font-weight: 500;
            }
        """)
        
        # 进度条
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(75)
        progress.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #3a4149;
                border-radius: 2px;
                height: 3px;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
                border-radius: 2px;
            }
        """)
        progress.setTextVisible(False)
        progress.setMaximumHeight(3)
        
        file_status = QLabel("Uploading...")
        file_status.setStyleSheet("""
            QLabel {
                color: #8a9ba8;
                font-size: 9px;
            }
        """)
        # 🔥 设置对象名称，用于后续状态更新
        file_status.setObjectName("status_label")
        
        file_info_layout.addWidget(file_name)
        file_info_layout.addWidget(progress)
        file_info_layout.addWidget(file_status)
        
        item_layout.addLayout(file_info_layout, 1)
        
        # 删除按钮
        delete_btn = QPushButton("×")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #f44336;
                font-size: 16px;
                padding: 2px;
                border-radius: 3px;
                min-width: 20px;
                max-width: 20px;
                min-height: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background-color: #f44336;
                color: #ffffff;
            }
        """)
        
        # 连接删除按钮的点击事件
        delete_btn.clicked.connect(lambda: self.remove_file_item(file_item, file_path))
        
        item_layout.addWidget(delete_btn)
        
        # 添加到文件列表（在stretch之前）
        layout = self.file_list_widget.layout()
        if layout:
            # 移除stretch
            count = layout.count()
            if count > 0:
                last_item = layout.itemAt(count - 1)
                if last_item.spacerItem():
                    layout.removeItem(last_item)
            
            layout.addWidget(file_item)
            layout.addStretch()
        
        # 模拟上传完成
        QTimer.singleShot(1500, lambda: self.complete_file_upload(progress, file_status))
    
    def complete_file_upload(self, progress, status_label):
        """完成文件上传"""
        progress.setValue(100)
        if status_label:
            status_label.setText("Completed")
            status_label.setStyleSheet("""
                QLabel {
                    color: #4caf50;
                    font-size: 9px;
                }
            """)
    
    def remove_file_item(self, file_item, file_path):
        """删除文件项"""
        try:
            # 如果删除的是当前显示的图片，清除显示
            if self.current_image_path == file_path:
                self.clear_current_image()
            
            # 从布局中移除文件项
            layout = self.file_list_widget.layout()
            if layout:
                layout.removeWidget(file_item)
                file_item.deleteLater()
            
            # 更新状态栏
            self.status_bar.showMessage(f"已删除文件: {os.path.basename(file_path)}", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "删除失败", f"无法删除文件: {str(e)}")
    
    def clear_current_image(self):
        """清除当前图片和分析结果"""
        self.current_image = None
        self.current_image_path = None
        self.analysis_result = None
        
        # 清除图像显示
        if hasattr(self, 'result_image_panel'):
            self.result_image_panel.clear_image()
        
        # 清除统计信息
        if hasattr(self, 'compact_stats'):
            self.compact_stats.clear_statistics()
        
        # 更新UI状态
        self.update_ui_state()
    
    def load_image_from_path(self, file_path):
        """从路径加载图片"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "文件不存在", f"文件不存在: {file_path}")
                return
            
            # 加载图像
            image = self.file_operations.load_image(file_path)
            if image is None:
                QMessageBox.warning(self, "加载失败", f"无法加载图像: {file_path}")
                return
            
            # 设置当前图像
            self.current_image = image
            self.current_image_path = file_path
            
            # 🔥 关键修复：设置到分析控制器
            self.analysis_controller.set_image(image)
            
            # 🔥 显示原图像（左侧）
            self.original_image_panel.set_image(image)
            
            # 🔥 检查是否有缓存的分析结果
            cached_result_loaded = False
            try:
                # 首先检查单张图片的缓存结果
                if file_path in self.analysis_cache:
                    cached_data = self.analysis_cache[file_path]
                    self.analysis_result = cached_data['analysis_result']
                    result_image = cached_data.get('result_image')
                    
                    if result_image is not None:
                        # 显示缓存的结果图像
                        self.result_image_panel.set_image(result_image)
                        cached_result_loaded = True
                        
                        # 🔥 更新所有统计信息显示
                        if hasattr(self, 'compact_stats') and self.analysis_result:
                            self.compact_stats.update_statistics(self.analysis_result.statistics)
                        
                        # 🔥 更新Analysis Summary面板
                        self.update_analysis_summary()
                        
                        # 启用下载按钮
                        if hasattr(self, 'download_btn'):
                            self.download_btn.setEnabled(True)
                        
                        print(f"✅ 从缓存加载分析结果: {os.path.basename(file_path)}")
                
                # 如果没有缓存结果，检查批量分析结果
                if not cached_result_loaded:
                    if hasattr(self, 'load_batch_result') and self.load_batch_result(file_path):
                        # 如果有批量分析结果，在右侧显示结果
                        cached_result_loaded = True
                    else:
                        # 没有任何分析结果，右侧也显示原图像
                        self.result_image_panel.set_image(image)
                        # 清除之前的分析结果
                        self.clear_analysis_result()
                        
            except Exception as e:
                print(f"结果加载错误: {e}")
                # 右侧显示原图像
                self.result_image_panel.set_image(image)
                # 清除之前的分析结果
                self.clear_analysis_result()
            
            # 更新UI状态
            self.update_ui_state()
            
            # 更新状态栏
            self.status_bar.showMessage(f"已加载图像: {os.path.basename(file_path)}", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "加载错误", f"加载图像时发生错误: {str(e)}")
    
    def clear_analysis_result(self):
        """清除分析结果"""
        self.analysis_result = None
        
        # 清除结果图像显示
        if hasattr(self, 'result_image_panel'):
            self.result_image_panel.clear_image()
        
        # 清除统计信息
        if hasattr(self, 'compact_stats'):
            self.compact_stats.clear_statistics()
        
        # 禁用下载按钮
        if hasattr(self, 'download_btn'):
            self.download_btn.setEnabled(False)
    
    def update_ui_state(self):
        """更新UI状态"""
        try:
            # 更新分析按钮状态
            if hasattr(self, 'analyze_btn'):
                self.analyze_btn.setEnabled(self.current_image is not None)
            
            # 更新批量分析按钮状态
            if hasattr(self, 'batch_analyze_btn'):
                self.update_batch_analyze_button()  # 使用新的更新方法
            
            # 更新下载按钮状态 - 支持批量下载
            if hasattr(self, 'download_btn'):
                # 检查是否有当前分析结果或已分析的选中文件
                has_current_result = self.analysis_result is not None
                analyzed_files = self.get_analyzed_selected_files()
                has_analyzed_files = len(analyzed_files) > 0
                
                self.download_btn.setEnabled(has_current_result or has_analyzed_files)
                
                # 更新按钮文本显示可下载数量
                if len(analyzed_files) > 1:
                    self.download_btn.setText(f"Download Results ({len(analyzed_files)})")
                elif len(analyzed_files) == 1:
                    self.download_btn.setText("Download Result")
                else:
                    self.download_btn.setText("Download Results")
            
            # 更新清除按钮状态
            if hasattr(self, 'clear_btn'):
                self.clear_btn.setEnabled(self.current_image is not None or self.analysis_result is not None)
            
            # 更新状态栏
            if hasattr(self, 'status_bar'):
                if self.current_image is not None:
                    image_name = os.path.basename(self.current_image_path) if self.current_image_path else "未知图像"
                    self.status_bar.showMessage(f"当前图像: {image_name}")
                else:
                    self.status_bar.showMessage("就绪")
                    
        except Exception as e:
            print(f"更新UI状态失败: {e}")
    

    
    def setup_toolbar(self):
        """设置现代化工具栏"""
        # 隐藏默认工具栏，使用自定义标题栏
        pass
    
    def setup_status_bar(self):
        """设置现代化状态栏"""
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #252a31;
                border-top: 1px solid #3a4149;
                color: #8a9ba8;
                font-size: 12px;
            }
        """)
        self.setStatusBar(self.status_bar)
        
        # 状态标签
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #8a9ba8;
                padding: 5px 10px;
            }
        """)
        self.status_bar.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #4a5568;
                border-radius: 4px;
                background-color: #1a1d23;
                text-align: center;
                color: #ffffff;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
                border-radius: 3px;
            }
        """)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # 紧凑型统计信息
        self.compact_stats = CompactStatisticsWidget()
        self.status_bar.addPermanentWidget(self.compact_stats)
    
    def connect_signals(self):
        """连接信号和槽"""
        # 模型选择变化
        self.vehicle_model_combo.currentTextChanged.connect(self.on_model_selection_changed)
        self.water_model_combo.currentTextChanged.connect(self.on_model_selection_changed)
        
        # 按钮信号
        self.clear_btn.clicked.connect(self.clear_all)
        self.download_btn.clicked.connect(self.save_result)
        
        # 分析控制器信号
        self.analysis_controller.analysis_started.connect(self.on_analysis_started)
        self.analysis_controller.analysis_progress.connect(self.on_analysis_progress)
        self.analysis_controller.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_controller.analysis_failed.connect(self.on_analysis_failed)
    
    def load_models(self):
        """加载深度学习模型"""
        self.update_status("Loading models... (This may take a moment for RT-DETR)")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        # 强制更新UI
        QApplication.processEvents()
        
        try:
            # 使用分析控制器加载模型
            success = self.analysis_controller.load_models()
            
            if success:
                # 更新模型选择下拉框
                available_models = self.analysis_controller.get_available_models()
                
                # 填充车辆检测模型列表
                self.vehicle_model_combo.clear()
                vehicle_models = available_models.get('vehicle_models', [])
                if vehicle_models:
                    self.vehicle_model_combo.addItems(vehicle_models)
                else:
                    self.vehicle_model_combo.addItem("No models available")
                
                # 填充水面分割模型列表
                self.water_model_combo.clear()
                water_models = available_models.get('water_models', [])
                if water_models:
                    self.water_model_combo.addItems(water_models)
                else:
                    self.water_model_combo.addItem("No models available")
                
                self.update_status("Models loaded successfully")
                
            else:
                self.update_status("Some models failed to load")
                QMessageBox.warning(
                    self,
                    "Model Loading Warning",
                    "Some models failed to load. Some features may not be available.\nPlease check the model files in the models directory."
                )
        
        except Exception as e:
            self.update_status(f"Model loading failed: {str(e)}")
            QMessageBox.critical(
                self,
                "Model Loading Error",
                f"Model loading failed:\n{str(e)}\n\nPlease check if model files exist and are in correct format."
            )
        
        finally:
            self.progress_bar.setVisible(False)
    
    def select_image_file(self):
        """选择图像文件"""
        file_path = self.file_operations.select_image_file()
        if file_path:
            self.load_image(file_path)
    
    def select_multiple_images(self):
        """批量选择图像文件"""
        from PyQt6.QtWidgets import QFileDialog
        
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Multiple Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;All Files (*)"
        )
        
        if file_paths:
            self.load_multiple_images(file_paths)
    
    def load_multiple_images(self, file_paths: list):
        """批量加载图像"""
        try:
            self.update_status(f"Loading {len(file_paths)} images...")
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(file_paths))
            
            loaded_count = 0
            failed_files = []
            
            for i, file_path in enumerate(file_paths):
                try:
                    # 更新进度
                    self.progress_bar.setValue(i)
                    self.update_status(f"Loading image {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                    
                    # 使用ImageProcessor加载图像
                    from ..core import ImageProcessor
                    image_processor = ImageProcessor()
                    image = image_processor.load_image(file_path)
                    
                    # 添加到文件列表（不设置为当前图像）
                    self.add_file_to_list(file_path)
                    loaded_count += 1
                    
                    # 处理UI事件，保持界面响应
                    QApplication.processEvents()
                    
                except Exception as e:
                    failed_files.append(f"{os.path.basename(file_path)}: {str(e)}")
                    continue
            
            # 完成加载
            self.progress_bar.setValue(len(file_paths))
            self.progress_bar.setVisible(False)
            
            # 显示结果
            if loaded_count > 0:
                self.update_status(f"Successfully loaded {loaded_count} images")
                
                # 如果没有当前图像，加载第一张
                if self.current_image is None and file_paths:
                    self.load_image(file_paths[0])
            
            # 显示失败的文件
            if failed_files:
                from PyQt6.QtWidgets import QMessageBox
                failed_text = "\n".join(failed_files[:10])  # 最多显示10个失败文件
                if len(failed_files) > 10:
                    failed_text += f"\n... and {len(failed_files) - 10} more files"
                
                QMessageBox.warning(
                    self,
                    "Some Files Failed to Load",
                    f"Failed to load {len(failed_files)} files:\n\n{failed_text}"
                )
                
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.update_status(f"Batch loading failed: {str(e)}")
            QMessageBox.critical(
                self,
                "Batch Loading Error",
                f"Failed to load images:\n{str(e)}"
            )
    
    def load_image(self, file_path: str):
        """加载图像"""
        try:
            self.update_status("Loading image...")
            
            # 使用ImageProcessor加载图像
            from ..core import ImageProcessor
            image_processor = ImageProcessor()
            image = image_processor.load_image(file_path)
            
            # 保存当前图像
            self.current_image = image
            self.current_image_path = file_path
            
            # 设置到分析控制器
            self.analysis_controller.set_image(image)
            
            # 🔥 显示图像到左右两侧
            self.original_image_panel.set_image(image)  # 左侧显示原图
            self.result_image_panel.set_image(image)    # 右侧初始也显示原图
            
            # 添加到文件列表
            self.add_file_to_list(file_path)
            
            # 更新状态
            image_info = image_processor.get_image_info(image)
            self.update_status(f"Image loaded: {image_info['width']}x{image_info['height']}")
            
            # 启用分析按钮
            self.analyze_btn.setEnabled(True)
            
            # 清除之前的结果
            self.clear_result()
            
        except Exception as e:
            self.update_status(f"Failed to load image: {str(e)}")
            QMessageBox.critical(
                self,
                "Image Loading Error",
                f"Unable to load image:\n{str(e)}"
            )
    

    
    def start_analysis(self):
        """开始分析"""
        # 获取任务模式
        task_mode = self.task_mode_combo.currentText()
        
        # 根据任务模式设置模型
        if task_mode == "Vehicle Detection Only":
            # 仅车辆检测
            vehicle_model = self.vehicle_model_combo.currentText()
            if vehicle_model and vehicle_model != "No models available":
                self.analysis_controller.set_vehicle_model(vehicle_model)
                self.analysis_controller.set_water_model(None)  # 不使用水面分割
            else:
                QMessageBox.warning(self, "Warning", "Please select a valid vehicle detection model.")
                return
                
        elif task_mode == "Water Segmentation Only":
            # 仅水面分割
            water_model = self.water_model_combo.currentText()
            if water_model and water_model != "No models available":
                self.analysis_controller.set_vehicle_model(None)  # 不使用车辆检测
                self.analysis_controller.set_water_model(water_model)
            else:
                QMessageBox.warning(self, "Warning", "Please select a valid water segmentation model.")
                return
                
        else:  # Combined Analysis
            # 组合分析
            vehicle_model = self.vehicle_model_combo.currentText()
            water_model = self.water_model_combo.currentText()
            
            if (vehicle_model and vehicle_model != "No models available" and
                water_model and water_model != "No models available"):
                self.analysis_controller.set_vehicle_model(vehicle_model)
                self.analysis_controller.set_water_model(water_model)
            else:
                QMessageBox.warning(self, "Warning", "Please select valid models for both tasks.")
                return
        
        # 启动分析
        self.analysis_controller.start_analysis()
    
    def start_batch_analysis(self):
        """开始批量分析"""
        # 🔥 获取选中的文件路径
        selected_files = self.get_selected_file_paths()
        
        if not selected_files:
            QMessageBox.warning(self, "Warning", "No images selected for batch analysis.")
            return
        
        if len(selected_files) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 images for batch analysis.")
            return
        
        # 确认批量分析
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Batch Analysis",
            f"Analyze {len(selected_files)} selected images?\n\nThis may take some time.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 开始批量分析
        self.perform_batch_analysis(selected_files)
    
    def get_uploaded_file_paths(self):
        """获取所有上传的文件路径"""
        file_paths = []
        
        # 遍历文件列表组件，提取文件路径
        layout = self.file_list_widget.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    # 从widget中提取文件路径（需要在add_file_to_list中存储）
                    if hasattr(widget, 'file_path'):
                        file_paths.append(widget.file_path)
        
        return file_paths
    
    def get_selected_file_paths(self):
        """获取选中的文件路径"""
        selected_paths = []
        
        # 遍历文件列表组件，提取选中的文件路径
        layout = self.file_list_widget.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    # 检查是否有文件路径和选择框
                    if hasattr(widget, 'file_path') and hasattr(widget, 'checkbox'):
                        if widget.checkbox.isChecked():
                            selected_paths.append(widget.file_path)
        
        return selected_paths
    
    def update_batch_analyze_button(self):
        """更新批量分析按钮和下载按钮状态"""
        selected_files = self.get_selected_file_paths()
        
        # 更新批量分析按钮
        if hasattr(self, 'batch_analyze_btn'):
            self.batch_analyze_btn.setEnabled(len(selected_files) > 1)
            # 更新按钮文本显示选中数量
            if len(selected_files) > 1:
                self.batch_analyze_btn.setText(f"Batch Analyze ({len(selected_files)})")
            else:
                self.batch_analyze_btn.setText("Batch Analyze")
        
        # 🔥 同时更新下载按钮状态
        if hasattr(self, 'download_btn'):
            analyzed_files = self.get_analyzed_selected_files()
            has_current_result = self.analysis_result is not None
            has_analyzed_files = len(analyzed_files) > 0
            
            self.download_btn.setEnabled(has_current_result or has_analyzed_files)
            
            # 更新下载按钮文本
            if len(analyzed_files) > 1:
                self.download_btn.setText(f"Download Results ({len(analyzed_files)})")
            elif len(analyzed_files) == 1:
                self.download_btn.setText("Download Result")
            else:
                self.download_btn.setText("Download Results")
    
    def select_all_files(self):
        """全选所有文件"""
        layout = self.file_list_widget.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    if hasattr(widget, 'checkbox'):
                        widget.checkbox.setChecked(True)
        self.update_batch_analyze_button()
    
    def deselect_all_files(self):
        """取消全选所有文件"""
        layout = self.file_list_widget.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    if hasattr(widget, 'checkbox'):
                        widget.checkbox.setChecked(False)
        self.update_batch_analyze_button()
    
    def perform_batch_analysis(self, file_paths):
        """执行批量分析"""
        try:
            self.update_status(f"Starting batch analysis of {len(file_paths)} images...")
            
            # 禁用按钮
            self.batch_analyze_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.upload_btn.setEnabled(False)
            self.batch_upload_btn.setEnabled(False)
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(file_paths))
            
            # 🔥 创建批量分析结果存储
            if not hasattr(self, 'batch_results'):
                self.batch_results = {}
            
            successful_analyses = []
            failed_analyses = []
            
            for i, file_path in enumerate(file_paths):
                try:
                    # 更新进度
                    self.progress_bar.setValue(i)
                    self.update_status(f"Analyzing {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                    
                    # 加载图像
                    self.load_image(file_path)
                    
                    # 等待图像加载完成
                    QApplication.processEvents()
                    
                    # 🔥 执行同步分析
                    if self.current_image is not None:
                        analysis_result = self.perform_single_analysis_sync()
                        
                        if analysis_result:
                            # 保存分析结果到批量结果
                            self.batch_results[file_path] = {
                                'analysis_result': analysis_result['analysis_result'],
                                'result_image': analysis_result['result_image'],
                                'timestamp': time.time()
                            }
                            
                            # 🔥 同时保存到统一缓存，确保切换图片时能加载结果
                            self.analysis_cache[file_path] = {
                                'analysis_result': analysis_result['analysis_result'],
                                'result_image': analysis_result['result_image'],
                                'timestamp': time.time()
                            }
                            
                            successful_analyses.append(file_path)
                            
                            # 🔥 更新文件列表项的状态，显示已分析
                            self.update_file_item_status(file_path, "analyzed")
                            
                            print(f"✅ 批量分析结果已保存到缓存: {os.path.basename(file_path)}")
                        else:
                            failed_analyses.append(f"{os.path.basename(file_path)}: Analysis failed")
                    else:
                        failed_analyses.append(f"{os.path.basename(file_path)}: Failed to load image")
                    
                except Exception as e:
                    failed_analyses.append(f"{os.path.basename(file_path)}: {str(e)}")
                    continue
            
            # 完成批量分析
            self.progress_bar.setValue(len(file_paths))
            self.progress_bar.setVisible(False)
            
            # 恢复按钮状态
            self.batch_analyze_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.upload_btn.setEnabled(True)
            self.batch_upload_btn.setEnabled(True)
            
            # 显示结果
            success_count = len(successful_analyses)
            self.update_status(f"Batch analysis completed: {success_count} successful")
            
            if failed_analyses:
                failed_text = "\n".join(failed_analyses[:5])  # 显示前5个失败项
                if len(failed_analyses) > 5:
                    failed_text += f"\n... and {len(failed_analyses) - 5} more"
                
                QMessageBox.warning(
                    self,
                    "Batch Analysis Results",
                    f"Completed: {success_count}/{len(file_paths)}\n\nFailed analyses:\n{failed_text}"
                )
            else:
                QMessageBox.information(
                    self,
                    "Batch Analysis Complete",
                    f"Successfully analyzed all {success_count} images!"
                )
                
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.batch_analyze_btn.setEnabled(True)
            self.analyze_btn.setEnabled(True)
            self.upload_btn.setEnabled(True)
            self.batch_upload_btn.setEnabled(True)
            
            QMessageBox.critical(
                self,
                "Batch Analysis Error",
                f"Batch analysis failed:\n{str(e)}"
            )
    
    # 分析相关的事件处理方法
    def on_analysis_started(self):
        """分析开始处理"""
        self.analyze_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
    
    def on_analysis_progress(self, value: int, message: str):
        """分析进度处理"""
        self.progress_bar.setValue(value)
        self.update_status(message)
    
    def on_analysis_completed(self, result_data):
        """分析完成处理"""
        try:
            # 获取结果数据
            self.analysis_result = result_data['analysis_result']
            result_image = result_data['result_image']
            
            # 🔥 保存分析结果到缓存
            if self.current_image_path:
                self.analysis_cache[self.current_image_path] = {
                    'analysis_result': self.analysis_result,
                    'result_image': result_image,
                    'timestamp': time.time()
                }
                print(f"✅ 保存分析结果到缓存: {os.path.basename(self.current_image_path)}")
            
            # 显示结果图像
            self.result_image_panel.set_image(result_image)
            
            # 更新统计信息
            self.update_analysis_summary()
            
            # 启用保存按钮
            self.download_btn.setEnabled(True)
            
            self.update_status("Analysis completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Result Processing Error", f"Error processing analysis results:\n{str(e)}")
        
        finally:
            # 恢复按钮状态
            self.analyze_btn.setEnabled(True)
            self.upload_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def on_analysis_failed(self, error_message: str):
        """分析失败处理"""
        self.update_status(f"Analysis failed: {error_message}")
        
        # 恢复按钮状态
        self.analyze_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def save_result(self):
        """保存结果 - 支持单张和批量下载"""
        # 🔥 检查是否有已分析的图片可以批量下载
        analyzed_files = self.get_analyzed_selected_files()
        
        if len(analyzed_files) > 1:
            # 多张已分析图片，提供批量下载选项
            self.show_download_options(analyzed_files)
        elif len(analyzed_files) == 1:
            # 单张图片，直接下载
            self.save_single_result(analyzed_files[0])
        elif self.analysis_result is not None and self.current_image_path:
            # 当前图片有分析结果，下载当前图片
            self.save_single_result(self.current_image_path)
        else:
            QMessageBox.warning(self, "警告", "没有可保存的分析结果")
    
    def get_analyzed_selected_files(self):
        """获取已分析且选中的文件列表"""
        analyzed_files = []
        selected_files = self.get_selected_file_paths()
        
        for file_path in selected_files:
            if file_path in self.analysis_cache:
                analyzed_files.append(file_path)
        
        return analyzed_files
    
    def show_download_options(self, analyzed_files):
        """显示下载选项对话框"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("批量下载选项")
        dialog.setModal(True)
        dialog.resize(450, 350)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 标题
        title_label = QLabel(f"发现 {len(analyzed_files)} 张已分析的图片")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 16px; 
                font-weight: bold; 
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                border-left: 4px solid #4fc3f7;
            }
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 说明文字
        info_label = QLabel("请选择下载方式:")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: 500;
                color: #2c3e50;
                padding: 8px 12px;
                background-color: rgba(79, 195, 247, 0.1);
                border-radius: 6px;
                border: 1px solid rgba(79, 195, 247, 0.3);
            }
        """)
        layout.addWidget(info_label)
        
        # 选项按钮
        single_btn = QPushButton("📄 仅下载当前图片")
        single_btn.clicked.connect(lambda: self.handle_download_choice(dialog, "single"))
        
        batch_btn = QPushButton(f"📦 批量下载全部 {len(analyzed_files)} 张")
        batch_btn.clicked.connect(lambda: self.handle_download_choice(dialog, "batch", analyzed_files))
        
        cancel_btn = QPushButton("❌ 取消")
        cancel_btn.clicked.connect(dialog.reject)
        
        # 按钮样式
        base_button_style = """
            QPushButton {
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 8px;
                color: #333;
                background-color: #ffffff;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border-color: #bbb;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """
        
        batch_button_style = """
            QPushButton {
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #4fc3f7;
                border-radius: 8px;
                color: white;
                background-color: #4fc3f7;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #29b6f6;
                border-color: #29b6f6;
            }
            QPushButton:pressed {
                background-color: #0288d1;
            }
        """
        
        cancel_button_style = """
            QPushButton {
                padding: 12px 20px;
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #f44336;
                border-radius: 8px;
                color: #f44336;
                background-color: #ffffff;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #ffebee;
                border-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #ffcdd2;
            }
        """
        
        single_btn.setStyleSheet(base_button_style)
        batch_btn.setStyleSheet(batch_button_style)
        cancel_btn.setStyleSheet(cancel_button_style)
        
        # 添加按钮到布局，带间距
        layout.addWidget(single_btn)
        layout.addSpacing(10)
        layout.addWidget(batch_btn)
        layout.addSpacing(15)
        layout.addWidget(cancel_btn)
        
        # 添加弹性空间
        layout.addStretch()
        
        dialog.exec()
    
    def handle_download_choice(self, dialog, choice, analyzed_files=None):
        """处理下载选择"""
        dialog.accept()
        
        if choice == "single":
            if self.current_image_path and self.analysis_result:
                self.save_single_result(self.current_image_path)
            else:
                QMessageBox.warning(self, "警告", "当前图片没有分析结果")
        elif choice == "batch" and analyzed_files:
            self.save_batch_results(analyzed_files)
    
    def save_single_result(self, file_path):
        """保存单张图片的分析结果"""
        try:
            # 获取分析结果
            if file_path == self.current_image_path and self.analysis_result:
                analysis_result = self.analysis_result
                original_image = self.current_image
            elif file_path in self.analysis_cache:
                cached_data = self.analysis_cache[file_path]
                analysis_result = cached_data['analysis_result']
                # 重新加载原图
                from ..core import ImageProcessor
                image_processor = ImageProcessor()
                original_image = image_processor.load_image(file_path)
            else:
                QMessageBox.warning(self, "警告", "没有找到该图片的分析结果")
                return
            
            # 选择保存位置
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            suggested_name = f"{base_name}_analysis_result.jpg"
            
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存分析结果",
                suggested_name,
                "图像文件 (*.jpg *.jpeg *.png *.bmp)"
            )
            
            if save_path:
                # 创建结果图像
                from ..core import VisualizationEngine, ImageProcessor
                viz_engine = VisualizationEngine()
                result_image = viz_engine.create_result_image(original_image, analysis_result)
                
                # 保存图像
                image_processor = ImageProcessor()
                success = image_processor.save_image(result_image, save_path)
                
                if success:
                    self.update_status(f"结果已保存: {save_path}")
                    QMessageBox.information(self, "保存成功", f"分析结果已保存到:\n{save_path}")
                else:
                    QMessageBox.critical(self, "保存失败", "无法保存结果图像")
        
        except Exception as e:
            QMessageBox.critical(self, "保存错误", f"保存过程中发生错误:\n{str(e)}")
    
    def save_batch_results(self, analyzed_files):
        """批量保存分析结果"""
        try:
            # 选择保存目录
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "选择批量保存目录",
                os.path.expanduser("~/Desktop")
            )
            
            if not save_dir:
                return
            
            # 显示进度
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, len(analyzed_files))
            
            successful_saves = []
            failed_saves = []
            
            from ..core import VisualizationEngine, ImageProcessor
            viz_engine = VisualizationEngine()
            image_processor = ImageProcessor()
            
            for i, file_path in enumerate(analyzed_files):
                try:
                    # 更新进度
                    self.progress_bar.setValue(i)
                    self.update_status(f"保存 {i+1}/{len(analyzed_files)}: {os.path.basename(file_path)}")
                    QApplication.processEvents()
                    
                    # 获取分析结果
                    if file_path in self.analysis_cache:
                        cached_data = self.analysis_cache[file_path]
                        analysis_result = cached_data['analysis_result']
                        
                        # 加载原图
                        original_image = image_processor.load_image(file_path)
                        
                        # 创建结果图像
                        result_image = viz_engine.create_result_image(original_image, analysis_result)
                        
                        # 生成保存文件名
                        base_name = os.path.splitext(os.path.basename(file_path))[0]
                        save_filename = f"{base_name}_analysis_result.jpg"
                        save_path = os.path.join(save_dir, save_filename)
                        
                        # 保存图像
                        success = image_processor.save_image(result_image, save_path)
                        
                        if success:
                            successful_saves.append(os.path.basename(file_path))
                        else:
                            failed_saves.append(f"{os.path.basename(file_path)}: 保存失败")
                    else:
                        failed_saves.append(f"{os.path.basename(file_path)}: 没有分析结果")
                
                except Exception as e:
                    failed_saves.append(f"{os.path.basename(file_path)}: {str(e)}")
            
            # 完成批量保存
            self.progress_bar.setValue(len(analyzed_files))
            self.progress_bar.setVisible(False)
            
            # 显示结果
            success_count = len(successful_saves)
            self.update_status(f"批量保存完成: {success_count} 个成功")
            
            if failed_saves:
                failed_text = "\n".join(failed_saves[:5])  # 只显示前5个失败项
                if len(failed_saves) > 5:
                    failed_text += f"\n... 还有 {len(failed_saves) - 5} 个失败"
                
                QMessageBox.warning(
                    self,
                    "批量保存结果",
                    f"完成: {success_count}/{len(analyzed_files)}\n\n失败项目:\n{failed_text}"
                )
            else:
                QMessageBox.information(
                    self,
                    "批量保存完成",
                    f"成功保存全部 {success_count} 张分析结果到:\n{save_dir}"
                )
        
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "批量保存错误", f"批量保存失败:\n{str(e)}")
    
    def update_analysis_summary(self):
        """更新分析摘要"""
        if self.analysis_result is None:
            return
        
        stats = self.analysis_result.statistics
        
        # 🔍 调试信息
        print(f"🔍 更新分析摘要:")
        print(f"  - 总车辆数: {stats.total_vehicles}")
        print(f"  - 轮胎级淹没: {stats.light_flood_count}")
        print(f"  - 车门级淹没: {stats.moderate_flood_count}")
        print(f"  - 车窗级淹没: {stats.severe_flood_count}")
        
        # 更新检测到的对象数量
        self.objects_count_label.setText(str(stats.total_vehicles))
        
        # 更新淹没级别
        if stats.severe_flood_count > 0:
            self.water_level_label.setText("车窗级")
            self.water_level_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #f44336;
                }
            """)
        elif stats.moderate_flood_count > 0:
            self.water_level_label.setText("车门级")
            self.water_level_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #ff9800;
                }
            """)
        else:
            self.water_level_label.setText("轮胎级")
            self.water_level_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #4caf50;
                }
            """)
    
    def on_model_selection_changed(self):
        """模型选择变化处理"""
        # 清除之前的分析结果
        if self.analysis_result is not None:
            self.clear_result()
    
    def clear_result(self):
        """清除分析结果"""
        self.analysis_result = None
        # 🔥 只清除右侧结果面板，左侧原图保持不变
        if hasattr(self, 'result_image_panel'):
            # 如果有当前图像，右侧显示原图；否则清空
            if hasattr(self, 'current_image') and self.current_image is not None:
                self.result_image_panel.set_image(self.current_image)
            else:
                self.result_image_panel.clear_image()
        if hasattr(self, 'compact_stats'):
            self.compact_stats.clear_statistics()
        self.download_btn.setEnabled(False)
        
        # 重置分析摘要
        self.objects_count_label.setText("0")
        self.water_level_label.setText("--")
        self.water_level_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #8a9ba8;
            }
        """)
    
    def update_status(self, message: str):
        """更新状态信息"""
        self.status_label.setText(message)
        QApplication.processEvents()  # 立即更新UI
    

    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 确保分割器比例在窗口大小变化时保持合理
        if hasattr(self, 'image_splitter') and self.image_splitter:
            # 保持左右图像面板等宽
            total_width = self.image_splitter.width()
            if total_width > 0:
                half_width = total_width // 2
                self.image_splitter.setSizes([half_width, half_width])
        
        # 图像显示组件会自动处理内部的图像缩放
    
    def center_window(self):
        """将窗口居中显示"""
        try:
            from PyQt6.QtGui import QGuiApplication
            
            # 获取屏幕几何信息
            screen = QGuiApplication.primaryScreen()
            if screen:
                screen_geometry = screen.availableGeometry()
                window_geometry = self.frameGeometry()
                
                # 计算居中位置
                center_point = screen_geometry.center()
                window_geometry.moveCenter(center_point)
                
                # 移动窗口到居中位置
                self.move(window_geometry.topLeft())
        except Exception as e:
            # 如果居中失败，不影响程序运行
            print(f"窗口居中失败: {e}")

    def perform_single_analysis_sync(self):
        """执行单张图片的同步分析"""
        try:
            if self.current_image is None:
                return None
            
            # 获取任务模式和模型设置
            task_mode = self.task_mode_combo.currentText()
            vehicle_model = self.vehicle_model_combo.currentText()
            water_model = self.water_model_combo.currentText()
            
            print(f"🔄 执行同步分析: {task_mode}")
            
            # 🔥 直接执行同步分析，不使用异步控制器
            # 使用现有的分析控制器进行同步分析
            from ..core import ModelManager, FloodAnalyzer, VisualizationEngine
            from ..core.data_models import Statistics, AnalysisResult
            import numpy as np
            
            # 获取现有的模型管理器
            model_manager = self.analysis_controller.model_manager
            flood_analyzer = FloodAnalyzer()
            viz_engine = VisualizationEngine()
            
            # 设置模型
            if task_mode == "Vehicle Detection Only":
                if vehicle_model and vehicle_model != "No models available":
                    model_manager.set_active_models(vehicle_model, None)
                    vehicles = model_manager.predict_vehicles(self.current_image)
                    water_mask = None
                else:
                    print("❌ 无效的车辆检测模型")
                    return None
            elif task_mode == "Water Segmentation Only":
                if water_model and water_model != "No models available":
                    model_manager.set_active_models(None, water_model)
                    vehicles = []
                    water_mask = model_manager.predict_water(self.current_image)
                else:
                    print("❌ 无效的水面分割模型")
                    return None
            else:  # Combined Analysis
                if (vehicle_model and vehicle_model != "No models available" and 
                    water_model and water_model != "No models available"):
                    model_manager.set_active_models(vehicle_model, water_model)
                    vehicles = model_manager.predict_vehicles(self.current_image)
                    water_mask = model_manager.predict_water(self.current_image)
                else:
                    print("❌ 组合分析需要有效的车辆和水面模型")
                    return None
            
            # 分析淹没情况
            if vehicles and water_mask is not None:
                analysis_result = flood_analyzer.analyze_scene(vehicles, water_mask)
            else:
                # 创建简化的分析结果
                from ..core.data_models import VehicleResult
                
                vehicle_results = []
                if vehicles:
                    for i, v in enumerate(vehicles):
                        predicted_flood_level = flood_analyzer._extract_flood_level_from_detection(v)
                        vehicle_results.append(VehicleResult(
                            detection=v,
                            flood_level=predicted_flood_level,
                            overlap_ratio=0.0,
                            vehicle_id=i
                        ))
                
                # 计算统计信息
                wheel_count = sum(1 for vr in vehicle_results if vr.flood_level.name == 'WHEEL_LEVEL')
                window_count = sum(1 for vr in vehicle_results if vr.flood_level.name == 'WINDOW_LEVEL')
                roof_count = sum(1 for vr in vehicle_results if vr.flood_level.name == 'ROOF_LEVEL')
                
                water_coverage = 0.0
                if water_mask is not None:
                    water_pixels = np.sum(water_mask > 0)
                    total_pixels = water_mask.size
                    water_coverage = (water_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
                
                stats = Statistics(
                    total_vehicles=len(vehicle_results),
                    wheel_level_count=wheel_count,
                    window_level_count=window_count,
                    roof_level_count=roof_count,
                    water_coverage_percentage=water_coverage,
                    processing_time=0.0
                )
                
                if water_mask is None:
                    water_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                
                analysis_result = AnalysisResult(
                    vehicles=vehicle_results,
                    water_mask=water_mask,
                    statistics=stats,
                    original_image_shape=self.current_image.shape[:2]
                )
            
            # 生成结果图像
            result_image = viz_engine.create_result_image(self.current_image, analysis_result)
            
            return {
                'analysis_result': analysis_result,
                'result_image': result_image,
                'original_image': self.current_image
            }
            
        except Exception as e:
            print(f"同步分析失败: {e}")
            return None
    
    def update_file_item_status(self, file_path: str, status: str):
        """更新文件列表项的状态"""
        try:
            layout = self.file_list_widget.layout()
            if layout:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        widget = item.widget()
                        if hasattr(widget, 'file_path') and widget.file_path == file_path:
                            # 找到对应的状态标签并更新
                            for child in widget.findChildren(QLabel):
                                if hasattr(child, 'objectName') and child.objectName() == 'status_label':
                                    if status == "analyzed":
                                        child.setText("Analyzed")
                                        child.setStyleSheet("""
                                            QLabel {
                                                color: #4caf50;
                                                font-size: 9px;
                                            }
                                        """)
                                    break
                            break
        except Exception as e:
            print(f"更新文件状态失败: {e}")
    
    def load_batch_result(self, file_path: str):
        """加载批量分析的结果"""
        try:
            if hasattr(self, 'batch_results') and file_path in self.batch_results:
                result_data = self.batch_results[file_path]
                
                # 设置分析结果
                self.analysis_result = result_data['analysis_result']
                
                # 显示结果图像
                self.result_image_panel.set_image(result_data['result_image'])
                
                # 更新统计信息
                self.update_analysis_summary()
                
                # 启用下载按钮
                self.download_btn.setEnabled(True)
                
                # 更新状态
                self.update_status(f"Loaded batch analysis result: {os.path.basename(file_path)}")
                
                return True
            return False
        except Exception as e:
            print(f"加载批量结果失败: {e}")
            return False


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("积水车辆检测系统")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Flood Detection Team")
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
