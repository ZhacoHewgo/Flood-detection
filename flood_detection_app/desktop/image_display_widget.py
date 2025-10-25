"""
图像显示组件
Image Display Widget for showing images with proper scaling and interaction
"""

import numpy as np
from typing import Optional, Tuple
from PyQt6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, 
    QScrollArea, QFrame, QPushButton, QSlider,
    QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QRect
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor,
    QFont, QFontMetrics, QWheelEvent, QMouseEvent
)


class ImageDisplayWidget(QLabel):
    """图像显示组件"""
    
    # 信号定义
    imageClicked = pyqtSignal(int, int)  # 图像点击信号 (x, y)
    zoomChanged = pyqtSignal(float)      # 缩放变化信号
    
    def __init__(self, title: str = "图像"):
        super().__init__()
        
        self.title = title
        self.original_image = None
        self.current_pixmap = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.fit_to_window = True
        
        # 鼠标交互
        self.last_pan_point = None
        self.is_panning = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        # 设置基本属性
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        
        # 设置深色主题样式
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #4a5568;
                background-color: #1a1d23;
                border-radius: 8px;
                color: #ffffff;
            }
            QLabel:hover {
                border-color: #4fc3f7;
            }
        """)
        
        # 设置默认文本
        self.set_placeholder_text()
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
    
    def set_placeholder_text(self):
        """设置占位符文本"""
        placeholder_text = f"{self.title}\n\nClick to select image file\nor drag image here"
        self.setText(placeholder_text)
        
        # 设置深色主题文本样式
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)
        self.setStyleSheet(self.styleSheet() + """
            QLabel {
                color: #8a9ba8;
                font-size: 12px;
            }
        """)
    
    def set_image(self, image: np.ndarray):
        """
        设置要显示的图像
        
        Args:
            image: BGR格式的图像数组
        """
        try:
            if image is None or image.size == 0:
                self.clear_image()
                return
            
            # 保存原始图像
            self.original_image = image.copy()
            
            # 转换为Qt格式
            self.current_pixmap = self._numpy_to_pixmap(image)
            
            # 显示图像
            self.update_display()
            
            # 重置缩放
            if self.fit_to_window:
                self.fit_image_to_window()
            else:
                self.zoom_factor = 1.0
                self.update_display()
            
            # 发送缩放变化信号
            self.zoomChanged.emit(self.zoom_factor)
            
        except Exception as e:
            print(f"图像设置失败: {e}")
            self.set_error_text(f"图像显示失败: {str(e)}")
    
    def clear_image(self):
        """清除图像"""
        self.original_image = None
        self.current_pixmap = None
        self.zoom_factor = 1.0
        self.fit_to_window = True
        self.set_placeholder_text()
    
    def set_error_text(self, error_message: str):
        """设置错误文本"""
        self.setText(f"❌ {error_message}")
        self.setStyleSheet(self.styleSheet() + """
            QLabel {
                color: #f44336;
            }
        """)
    
    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """将numpy数组转换为QPixmap"""
        try:
            # 确保图像是3通道BGR格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                
                # BGR转RGB
                rgb_image = image[:, :, ::-1]
                
                # 确保数据是连续的并转换为bytes
                rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
                
                # 创建QImage
                q_image = QImage(
                    rgb_image.data.tobytes(),
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                
                # 转换为QPixmap
                return QPixmap.fromImage(q_image)
            
            elif len(image.shape) == 2:
                # 灰度图像
                height, width = image.shape
                
                # 确保数据是连续的并转换为bytes
                gray_image = np.ascontiguousarray(image, dtype=np.uint8)
                
                q_image = QImage(
                    gray_image.data.tobytes(),
                    width,
                    height,
                    width,
                    QImage.Format.Format_Grayscale8
                )
                return QPixmap.fromImage(q_image)
            
            else:
                raise ValueError(f"不支持的图像格式: {image.shape}")
                
        except Exception as e:
            raise Exception(f"图像转换失败: {str(e)}")
    
    def update_display(self):
        """更新图像显示"""
        if self.current_pixmap is None:
            return
        
        try:
            # 应用缩放
            if self.zoom_factor != 1.0:
                scaled_size = self.current_pixmap.size() * self.zoom_factor
                scaled_pixmap = self.current_pixmap.scaled(
                    scaled_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                scaled_pixmap = self.current_pixmap
            
            # 设置图像
            self.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"图像显示更新失败: {e}")
    
    def fit_image_to_window(self):
        """适应窗口大小"""
        if self.current_pixmap is None:
            return
        
        # 获取可用空间
        available_size = self.size()
        if available_size.width() <= 0 or available_size.height() <= 0:
            return
        
        # 计算缩放比例
        image_size = self.current_pixmap.size()
        scale_x = available_size.width() / image_size.width()
        scale_y = available_size.height() / image_size.height()
        
        # 选择较小的缩放比例以保持宽高比
        self.zoom_factor = min(scale_x, scale_y, 1.0)  # 不放大，只缩小
        
        # 限制缩放范围
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))
        
        # 更新显示
        self.update_display()
        self.zoomChanged.emit(self.zoom_factor)
    
    def zoom_in(self, factor: float = 1.2):
        """放大图像"""
        if self.current_pixmap is None:
            return
        
        # 安全检查
        if self.zoom_factor <= 0:
            self.zoom_factor = 1.0
        
        if factor <= 0:
            factor = 1.2
        
        new_zoom = self.zoom_factor * factor
        if new_zoom <= self.max_zoom:
            self.zoom_factor = new_zoom
            self.fit_to_window = False
            self.update_display()
            self.zoomChanged.emit(self.zoom_factor)
    
    def zoom_out(self, factor: float = 1.2):
        """缩小图像"""
        if self.current_pixmap is None:
            return
        
        # 安全检查，防止除零错误
        if self.zoom_factor <= 0:
            self.zoom_factor = 1.0
        
        if factor <= 0:
            factor = 1.2
        
        new_zoom = self.zoom_factor / factor
        if new_zoom >= self.min_zoom:
            self.zoom_factor = new_zoom
            self.fit_to_window = False
            self.update_display()
            self.zoomChanged.emit(self.zoom_factor)
    
    def reset_zoom(self):
        """重置缩放"""
        if self.current_pixmap is None:
            return
        
        self.zoom_factor = 1.0
        self.fit_to_window = False
        self.update_display()
        self.zoomChanged.emit(self.zoom_factor)
    
    def set_zoom(self, zoom: float):
        """设置指定的缩放比例"""
        if self.current_pixmap is None:
            return
        
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, zoom))
        self.fit_to_window = False
        self.update_display()
        self.zoomChanged.emit(self.zoom_factor)
    
    def toggle_fit_to_window(self):
        """切换适应窗口模式"""
        self.fit_to_window = not self.fit_to_window
        if self.fit_to_window:
            self.fit_image_to_window()
    
    def get_display_size(self) -> Tuple[int, int]:
        """获取当前显示尺寸"""
        if self.current_pixmap is None:
            return (0, 0)
        
        scaled_size = self.current_pixmap.size() * self.zoom_factor
        return (scaled_size.width(), scaled_size.height())
    
    def get_original_size(self) -> Tuple[int, int]:
        """获取原始图像尺寸"""
        if self.original_image is None:
            return (0, 0)
        
        if len(self.original_image.shape) == 3:
            height, width, _ = self.original_image.shape
        else:
            height, width = self.original_image.shape
        
        return (width, height)
    
    def get_image_info(self) -> dict:
        """获取图像信息"""
        if self.original_image is None:
            return {"has_image": False}
        
        original_size = self.get_original_size()
        display_size = self.get_display_size()
        
        return {
            "has_image": True,
            "original_size": original_size,
            "display_size": display_size,
            "zoom_factor": self.zoom_factor,
            "fit_to_window": self.fit_to_window,
            "channels": self.original_image.shape[2] if len(self.original_image.shape) == 3 else 1,
            "dtype": str(self.original_image.dtype)
        }
    
    # 事件处理
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件 - 缩放"""
        if self.current_pixmap is None:
            return
        
        # 获取滚轮增量
        delta = event.angleDelta().y()
        
        # 缩放
        if delta > 0:
            self.zoom_in(1.1)
        else:
            self.zoom_out(1.1)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_pixmap is not None:
                # 计算图像坐标
                image_pos = self._widget_to_image_coords(event.pos())
                if image_pos is not None:
                    self.imageClicked.emit(image_pos[0], image_pos[1])
            
            # 开始拖拽
            self.last_pan_point = event.pos()
            self.is_panning = True
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if self.is_panning and self.last_pan_point is not None:
            # 这里可以实现图像拖拽功能
            # 暂时跳过，因为QLabel不直接支持内容拖拽
            pass
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = False
            self.last_pan_point = None
        
        super().mouseReleaseEvent(event)
    
    def _widget_to_image_coords(self, widget_pos) -> Optional[Tuple[int, int]]:
        """将组件坐标转换为图像坐标"""
        if self.current_pixmap is None:
            return None
        
        try:
            # 获取显示的图像矩形
            pixmap_rect = self.pixmap().rect()
            widget_rect = self.rect()
            
            # 计算图像在组件中的位置（居中显示）
            x_offset = (widget_rect.width() - pixmap_rect.width()) // 2
            y_offset = (widget_rect.height() - pixmap_rect.height()) // 2
            
            # 转换坐标
            image_x = widget_pos.x() - x_offset
            image_y = widget_pos.y() - y_offset
            
            # 检查是否在图像范围内
            if (0 <= image_x < pixmap_rect.width() and 
                0 <= image_y < pixmap_rect.height()):
                
                # 转换为原始图像坐标
                original_x = int(image_x / self.zoom_factor)
                original_y = int(image_y / self.zoom_factor)
                
                return (original_x, original_y)
            
            return None
            
        except Exception as e:
            print(f"坐标转换失败: {e}")
            return None
    
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 如果是适应窗口模式，重新调整图像大小
        if self.fit_to_window and self.current_pixmap is not None:
            self.fit_image_to_window()


class ImageDisplayPanel(QWidget):
    """图像显示面板 - 包含图像显示和控制按钮"""
    
    def __init__(self, title: str = "图像"):
        super().__init__()
        
        self.title = title
        self.image_widget = ImageDisplayWidget(title)
        
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 图像显示区域（占据全部空间）
        layout.addWidget(self.image_widget, 1)
    
    def connect_signals(self):
        """连接信号和槽"""
        pass
    
    def set_image(self, image: np.ndarray):
        """设置图像"""
        self.image_widget.set_image(image)
    
    def clear_image(self):
        """清除图像"""
        self.image_widget.clear_image()
    
    def get_image_widget(self) -> ImageDisplayWidget:
        """获取图像显示组件"""
        return self.image_widget