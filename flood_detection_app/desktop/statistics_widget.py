"""
统计信息显示组件
Statistics Display Widget for showing analysis results
"""

from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QFrame, QGridLayout, QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor

from ..core.data_models import Statistics, AnalysisResult, FloodLevel


class StatisticCard(QFrame):
    """统计卡片组件"""
    
    def __init__(self, title: str, value: str = "0", unit: str = "", color: str = "#2196F3"):
        super().__init__()
        
        self.title = title
        self.color = color
        
        self.setup_ui()
        self.set_value(value, unit)
    
    def setup_ui(self):
        """设置用户界面"""
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {self.color};
                border-radius: 8px;
                background-color: white;
                margin: 2px;
            }}
            QFrame:hover {{
                background-color: #f5f5f5;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        # 标题
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {self.color};
                font-size: 12px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.title_label)
        
        # 数值
        self.value_label = QLabel("0")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.value_label)
        
        # 单位
        self.unit_label = QLabel("")
        self.unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.unit_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.unit_label)
    
    def set_value(self, value: str, unit: str = ""):
        """设置数值"""
        self.value_label.setText(str(value))
        self.unit_label.setText(unit)
    
    def animate_value(self, target_value: int, duration: int = 1000):
        """数值动画效果"""
        # 简化实现，直接设置值
        self.set_value(str(target_value))


class FloodLevelBar(QWidget):
    """淹没等级条形图"""
    
    def __init__(self):
        super().__init__()
        
        self.data = {
            'wheel': 0,
            'window': 0,
            'roof': 0
        }
        
        self.colors = {
            'wheel': '#4CAF50',    # 绿色 - 车轮级
            'window': '#FF9800',   # 橙色 - 车窗级
            'roof': '#F44336'      # 红色 - 车顶级
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        title_label = QLabel("车辆淹没部位分布")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)
        
        # 创建进度条
        self.bars = {}
        self.labels = {}
        
        levels = [
            ('wheel', '车轮顶部及以下'),
            ('window', '车轮顶部至车窗下沿'),
            ('roof', '车窗及以上')
        ]
        
        for level_key, level_name in levels:
            # 水平布局
            h_layout = QHBoxLayout()
            
            # 标签
            label = QLabel(level_name)
            label.setMinimumWidth(80)
            label.setStyleSheet(f"""
                QLabel {{
                    color: {self.colors[level_key]};
                    font-weight: bold;
                }}
            """)
            h_layout.addWidget(label)
            
            # 进度条
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                    height: 20px;
                }}
                QProgressBar::chunk {{
                    background-color: {self.colors[level_key]};
                    border-radius: 2px;
                }}
            """)
            h_layout.addWidget(progress_bar, 1)
            
            # 数值标签
            count_label = QLabel("0")
            count_label.setMinimumWidth(30)
            count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            count_label.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    color: #333;
                }
            """)
            h_layout.addWidget(count_label)
            
            layout.addLayout(h_layout)
            
            self.bars[level_key] = progress_bar
            self.labels[level_key] = count_label
    
    def update_data(self, wheel: int, window: int, roof: int):
        """更新数据"""
        self.data = {
            'wheel': wheel,
            'window': window,
            'roof': roof
        }
        
        total = wheel + window + roof
        
        # 更新进度条和标签
        for level_key, count in self.data.items():
            percentage = (count / total * 100) if total > 0 else 0
            
            self.bars[level_key].setValue(int(percentage))
            self.labels[level_key].setText(str(count))


class StatisticsWidget(QWidget):
    """统计信息显示组件"""
    
    # 信号定义
    statistics_updated = pyqtSignal(object)  # 统计信息更新信号
    
    def __init__(self):
        super().__init__()
        
        self.current_statistics = None
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # 主要统计卡片
        self.setup_main_cards(main_layout)
        
        # 淹没等级分布
        self.setup_flood_distribution(main_layout)
        
        # 详细信息
        self.setup_detailed_info(main_layout)
    
    def setup_main_cards(self, parent_layout):
        """设置主要统计卡片"""
        cards_layout = QHBoxLayout()
        
        # 车辆总数卡片
        self.total_vehicles_card = StatisticCard(
            "车辆总数", "0", "辆", "#2196F3"
        )
        cards_layout.addWidget(self.total_vehicles_card)
        
        # 积水覆盖率卡片
        self.water_coverage_card = StatisticCard(
            "积水覆盖率", "0.0", "%", "#00BCD4"
        )
        cards_layout.addWidget(self.water_coverage_card)
        
        # 处理时间卡片
        self.processing_time_card = StatisticCard(
            "处理时间", "0.00", "秒", "#9C27B0"
        )
        cards_layout.addWidget(self.processing_time_card)
        
        parent_layout.addLayout(cards_layout)
    
    def setup_flood_distribution(self, parent_layout):
        """设置淹没等级分布"""
        # 创建分组框
        group_box = QGroupBox("淹没情况分析")
        group_box.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #333;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        group_layout = QVBoxLayout(group_box)
        
        # 淹没等级条形图
        self.flood_level_bar = FloodLevelBar()
        group_layout.addWidget(self.flood_level_bar)
        
        parent_layout.addWidget(group_box)
    
    def setup_detailed_info(self, parent_layout):
        """设置详细信息"""
        # 创建分组框
        group_box = QGroupBox("详细信息")
        group_box.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #333;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        group_layout = QGridLayout(group_box)
        
        # 详细信息标签
        info_items = [
            ("图像尺寸:", "image_size_label"),
            ("检测模型:", "vehicle_model_label"),
            ("分割模型:", "water_model_label"),
            ("分析时间:", "analysis_time_label")
        ]
        
        self.info_labels = {}
        
        for i, (title, label_key) in enumerate(info_items):
            # 标题标签
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    font-weight: bold;
                    color: #555;
                }
            """)
            group_layout.addWidget(title_label, i, 0)
            
            # 值标签
            value_label = QLabel("--")
            value_label.setStyleSheet("""
                QLabel {
                    color: #333;
                }
            """)
            group_layout.addWidget(value_label, i, 1)
            
            self.info_labels[label_key] = value_label
        
        parent_layout.addWidget(group_box)
    
    def update_statistics(self, statistics: Statistics, additional_info: Dict[str, Any] = None):
        """更新统计信息"""
        self.current_statistics = statistics
        
        # 更新主要卡片
        self.total_vehicles_card.set_value(str(statistics.total_vehicles), "辆")
        self.water_coverage_card.set_value(f"{statistics.water_coverage_percentage:.1f}", "%")
        self.processing_time_card.set_value(f"{statistics.processing_time:.2f}", "秒")
        
        # 更新淹没部位分布
        self.flood_level_bar.update_data(
            statistics.wheel_level_count,
            statistics.window_level_count,
            statistics.roof_level_count
        )
        
        # 更新详细信息
        if additional_info:
            for key, value in additional_info.items():
                if key in self.info_labels:
                    self.info_labels[key].setText(str(value))
        
        # 发送更新信号
        self.statistics_updated.emit(statistics)
    
    def clear_statistics(self):
        """清除统计信息"""
        self.current_statistics = None
        
        # 重置主要卡片
        self.total_vehicles_card.set_value("0", "辆")
        self.water_coverage_card.set_value("0.0", "%")
        self.processing_time_card.set_value("0.00", "秒")
        
        # 重置淹没部位分布
        self.flood_level_bar.update_data(0, 0, 0)
        
        # 重置详细信息
        for label in self.info_labels.values():
            label.setText("--")
    
    def get_statistics_summary(self) -> str:
        """获取统计信息摘要"""
        if not self.current_statistics:
            return "暂无统计信息"
        
        stats = self.current_statistics
        summary = (
            f"车辆总数: {stats.total_vehicles} | "
            f"车轮级: {stats.wheel_level_count} | "
            f"车窗级: {stats.window_level_count} | "
            f"车顶级: {stats.roof_level_count} | "
            f"积水: {stats.water_coverage_percentage:.1f}% | "
            f"用时: {stats.processing_time:.2f}s"
        )
        
        return summary
    
    def export_statistics(self) -> Dict[str, Any]:
        """导出统计信息"""
        if not self.current_statistics:
            return {}
        
        stats = self.current_statistics
        
        return {
            'total_vehicles': stats.total_vehicles,
            'light_flood_count': stats.light_flood_count,
            'moderate_flood_count': stats.moderate_flood_count,
            'severe_flood_count': stats.severe_flood_count,
            'water_coverage_percentage': stats.water_coverage_percentage,
            'processing_time': stats.processing_time,
            'summary': self.get_statistics_summary()
        }


class CompactStatisticsWidget(QWidget):
    """紧凑型统计信息组件（用于状态栏）"""
    
    def __init__(self):
        super().__init__()
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)
        
        # 统计标签
        self.stats_label = QLabel("Ready")
        self.stats_label.setStyleSheet("""
            QLabel {
                color: #8a9ba8;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.stats_label)
    
    def update_statistics(self, statistics: Statistics):
        """更新统计信息"""
        summary = (
            f"车辆: {statistics.total_vehicles} | "
            f"车轮级: {statistics.wheel_level_count} | "
            f"车窗级: {statistics.window_level_count} | "
            f"车顶级: {statistics.roof_level_count} | "
            f"积水: {statistics.water_coverage_percentage:.1f}% | "
            f"用时: {statistics.processing_time:.2f}s"
        )
        
        self.stats_label.setText(summary)
    
    def clear_statistics(self):
        """清除统计信息"""
        self.stats_label.setText("Ready")