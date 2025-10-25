"""
测试配置和共享fixtures
Test Configuration and Shared Fixtures
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flood_detection_app.core.data_models import (
    BoundingBox, Detection, VehicleResult, Statistics, AnalysisResult, FloodLevel
)


@pytest.fixture(scope="session")
def qapp():
    """创建QApplication实例"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # 不要退出应用，因为可能有其他测试需要使用


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_image():
    """创建示例图像"""
    # 创建一个简单的RGB图像 (100x100)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_detection():
    """创建示例检测结果"""
    bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9)
    return Detection(bbox=bbox, class_id=0, class_name="car")


@pytest.fixture
def sample_vehicle_result():
    """创建示例车辆结果"""
    bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50, confidence=0.9)
    detection = Detection(bbox=bbox, class_id=0, class_name="car")
    return VehicleResult(
        detection=detection,
        flood_level=FloodLevel.MODERATE,
        overlap_ratio=0.45,
        vehicle_id=1
    )


@pytest.fixture
def sample_statistics():
    """创建示例统计信息"""
    return Statistics(
        total_vehicles=3,
        light_flood_count=1,
        moderate_flood_count=1,
        severe_flood_count=1,
        water_coverage_percentage=25.5,
        processing_time=2.5
    )


@pytest.fixture
def sample_analysis_result(sample_vehicle_result, sample_statistics):
    """创建示例分析结果"""
    water_mask = np.zeros((100, 100), dtype=np.uint8)
    water_mask[20:80, 20:80] = 255  # 创建一个水面区域
    
    return AnalysisResult(
        vehicles=[sample_vehicle_result],
        water_mask=water_mask,
        statistics=sample_statistics,
        original_image_shape=(100, 100)
    )


@pytest.fixture
def mock_model_manager():
    """模拟ModelManager"""
    mock = Mock()
    mock.load_models.return_value = True
    mock.get_available_models.return_value = {
        'vehicle_models': ['yolov11_car_detection', 'rtdetr_car_detection'],
        'water_models': ['deeplabv3_water', 'yolov11_seg_water']
    }
    mock.set_active_models.return_value = True
    mock.predict_vehicles.return_value = []
    mock.predict_water.return_value = np.zeros((100, 100), dtype=np.uint8)
    return mock


@pytest.fixture
def mock_image_processor():
    """模拟ImageProcessor"""
    mock = Mock()
    mock.load_image.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mock.save_image.return_value = True
    mock.get_image_info.return_value = {'width': 100, 'height': 100}
    return mock


@pytest.fixture
def test_image_file(temp_dir):
    """创建测试图像文件"""
    import cv2
    
    # 创建测试图像
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 保存为文件
    image_path = os.path.join(temp_dir, "test_image.jpg")
    cv2.imwrite(image_path, image)
    
    return image_path


def wait_for_signal(signal, timeout=5000):
    """等待信号触发的辅助函数"""
    loop = QApplication.instance().processEvents
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: None)
    
    received = []
    signal.connect(lambda *args: received.append(args))
    
    timer.start(timeout)
    start_time = QTest.qGetTestTime()
    
    while not received and QTest.qGetTestTime() - start_time < timeout:
        loop()
        QTest.qWait(10)
    
    return len(received) > 0