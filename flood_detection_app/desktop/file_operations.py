"""
文件操作功能
File Operations for Desktop Application
"""

import os
from typing import Optional, List, Tuple
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget
from PyQt6.QtCore import QStandardPaths

from ..core.config import config_manager
from ..core.exceptions import FileOperationError, ValidationError


class FileOperations:
    """文件操作管理器"""
    
    def __init__(self, parent: QWidget = None):
        self.parent = parent
        self.last_directory = self._get_default_directory()
    
    def _get_default_directory(self) -> str:
        """获取默认目录"""
        # 尝试获取图片目录
        pictures_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        if pictures_dir and os.path.exists(pictures_dir):
            return pictures_dir
        
        # 回退到用户主目录
        home_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
        return home_dir if home_dir else os.getcwd()
    
    def select_image_file(self) -> Optional[str]:
        """选择图像文件"""
        try:
            # 构建文件过滤器
            supported_formats = config_manager.config.supported_image_formats
            format_filter = self._build_image_filter(supported_formats)
            
            # 打开文件对话框
            file_path, selected_filter = QFileDialog.getOpenFileName(
                self.parent,
                "选择图像文件",
                self.last_directory,
                format_filter
            )
            
            if file_path:
                # 验证文件
                if self._validate_image_file(file_path):
                    # 更新最后使用的目录
                    self.last_directory = os.path.dirname(file_path)
                    return file_path
                else:
                    self._show_error("文件验证失败", f"选择的文件不是有效的图像文件:\n{file_path}")
            
            return None
            
        except Exception as e:
            self._show_error("文件选择错误", f"选择文件时发生错误:\n{str(e)}")
            return None
    
    def save_result_image(self, default_name: str = "flood_analysis_result") -> Optional[str]:
        """保存结果图像"""
        try:
            # 构建保存文件名
            default_path = os.path.join(self.last_directory, f"{default_name}.jpg")
            
            # 构建保存过滤器
            save_filter = "JPEG图像 (*.jpg *.jpeg);;PNG图像 (*.png);;所有文件 (*)"
            
            # 打开保存对话框
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self.parent,
                "保存分析结果",
                default_path,
                save_filter
            )
            
            if file_path:
                # 确保有正确的扩展名
                file_path = self._ensure_extension(file_path, selected_filter)
                
                # 检查文件是否已存在
                if os.path.exists(file_path):
                    reply = QMessageBox.question(
                        self.parent,
                        "文件已存在",
                        f"文件已存在，是否覆盖?\n{file_path}",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    
                    if reply != QMessageBox.StandardButton.Yes:
                        return None
                
                # 更新最后使用的目录
                self.last_directory = os.path.dirname(file_path)
                return file_path
            
            return None
            
        except Exception as e:
            self._show_error("保存文件错误", f"保存文件时发生错误:\n{str(e)}")
            return None
    
    def _build_image_filter(self, supported_formats: List[str]) -> str:
        """构建图像文件过滤器"""
        # 创建格式组
        format_groups = {
            "JPEG图像": [".jpg", ".jpeg"],
            "PNG图像": [".png"],
            "BMP图像": [".bmp"],
            "TIFF图像": [".tiff", ".tif"]
        }
        
        # 构建过滤器字符串
        filters = []
        all_formats = []
        
        for group_name, extensions in format_groups.items():
            # 只包含支持的格式
            supported_exts = [ext for ext in extensions if ext in supported_formats]
            if supported_exts:
                pattern = " ".join([f"*{ext}" for ext in supported_exts])
                filters.append(f"{group_name} ({pattern})")
                all_formats.extend(supported_exts)
        
        # 添加"所有支持的图像"选项
        if all_formats:
            all_pattern = " ".join([f"*{ext}" for ext in all_formats])
            filters.insert(0, f"所有支持的图像 ({all_pattern})")
        
        # 添加"所有文件"选项
        filters.append("所有文件 (*)")
        
        return ";;".join(filters)
    
    def _validate_image_file(self, file_path: str) -> bool:
        """验证图像文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False
            
            # 检查文件扩展名
            if not config_manager.validate_image_format(file_path):
                return False
            
            # 检查文件是否可读
            try:
                with open(file_path, 'rb') as f:
                    # 读取文件头以验证是否为图像
                    header = f.read(16)
                    if len(header) < 4:
                        return False
                    
                    # 简单的文件头检查
                    if self._is_valid_image_header(header):
                        return True
            except:
                return False
            
            return False
            
        except Exception:
            return False
    
    def _is_valid_image_header(self, header: bytes) -> bool:
        """检查是否为有效的图像文件头"""
        # JPEG文件头
        if header.startswith(b'\xff\xd8\xff'):
            return True
        
        # PNG文件头
        if header.startswith(b'\x89PNG\r\n\x1a\n'):
            return True
        
        # BMP文件头
        if header.startswith(b'BM'):
            return True
        
        # TIFF文件头
        if header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):
            return True
        
        return False
    
    def _ensure_extension(self, file_path: str, selected_filter: str) -> str:
        """确保文件有正确的扩展名"""
        # 从过滤器中提取扩展名
        if "JPEG" in selected_filter:
            default_ext = ".jpg"
        elif "PNG" in selected_filter:
            default_ext = ".png"
        else:
            return file_path  # 保持原样
        
        # 检查是否已有扩展名
        _, current_ext = os.path.splitext(file_path)
        if not current_ext:
            return file_path + default_ext
        
        return file_path
    
    def _show_error(self, title: str, message: str):
        """显示错误消息"""
        if self.parent:
            QMessageBox.critical(self.parent, title, message)
        else:
            print(f"错误 - {title}: {message}")
    
    def get_recent_files(self, max_count: int = 10) -> List[str]:
        """获取最近使用的文件列表（简化实现）"""
        # 这里可以实现最近文件的持久化存储
        # 暂时返回空列表
        return []
    
    def add_to_recent_files(self, file_path: str):
        """添加到最近文件列表"""
        # 这里可以实现最近文件的持久化存储
        pass
    
    def load_image(self, file_path: str):
        """加载图像文件"""
        try:
            # 验证文件路径
            if not os.path.exists(file_path):
                raise FileOperationError("加载", file_path, "文件不存在")
            
            # 验证文件格式
            if not self._validate_image_file(file_path):
                raise ValidationError("图像格式", file_path, "不支持的图像格式")
            
            # 使用ImageProcessor加载图像
            from ..core.image_processor import ImageProcessor
            return ImageProcessor.load_image(file_path)
            
        except Exception as e:
            self._show_error("加载图像失败", f"无法加载图像 {file_path}:\n{str(e)}")
            return None


class DragDropHandler:
    """拖拽处理器"""
    
    def __init__(self, parent: QWidget, file_ops: FileOperations):
        self.parent = parent
        self.file_ops = file_ops
        
        # 启用拖拽
        self.parent.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否包含图像文件
            urls = event.mimeData().urls()
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if config_manager.validate_image_format(file_path):
                        event.acceptProposedAction()
                        return
        
        event.ignore()
    
    def dragMoveEvent(self, event):
        """拖拽移动事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """拖拽放下事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            
            # 处理第一个有效的图像文件
            for url in urls:
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if self.file_ops._validate_image_file(file_path):
                        # 触发文件加载
                        self._handle_dropped_file(file_path)
                        event.acceptProposedAction()
                        return
        
        event.ignore()
    
    def _handle_dropped_file(self, file_path: str):
        """处理拖拽的文件"""
        # 这个方法需要在具体使用时重写
        # 或者通过信号机制通知主窗口
        print(f"拖拽文件: {file_path}")


def create_file_operations(parent: QWidget = None) -> FileOperations:
    """创建文件操作实例的工厂函数"""
    return FileOperations(parent)