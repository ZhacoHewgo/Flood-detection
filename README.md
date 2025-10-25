# 积水车辆检测系统

基于深度学习的积水车辆检测GUI应用程序，支持桌面版和Web版两种部署方式。

## 功能特性

- 🚗 **车辆检测**: 支持YOLOv11和RT-DETR两种检测模型
- 🌊 **水面分割**: 支持DeepLabV3和YOLOv11分割模型  
- 📊 **淹没分析**: 自动计算车辆淹没等级（轻度/中度/重度）
- 🖥️ **桌面应用**: PyQt6构建的用户友好界面
- 🌐 **Web应用**: React前端 + FastAPI后端
- 📈 **统计分析**: 详细的淹没情况统计和可视化
- 💾 **结果导出**: 支持保存分析结果图像

## 项目结构

```
flood_detection_app/
├── main.py                     # 应用程序入口
├── requirements.txt            # Python依赖
├── models/                     # 模型文件目录
│   ├── yolov11_car_detection.pt
│   ├── rtdetr_car_detection.pt
│   ├── deeplabv3_water.pt
│   └── yolov11_seg_water.pt
├── flood_detection_app/
│   ├── core/                   # 核心引擎
│   │   ├── data_models.py      # 数据模型
│   │   ├── exceptions.py       # 异常处理
│   │   └── config.py          # 配置管理
│   ├── desktop/               # 桌面GUI应用
│   └── web/                   # Web应用
│       ├── backend/           # FastAPI后端
│       └── frontend/          # React前端
└── README.md
```

## 安装和运行

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd flood-vehicle-detection-gui

# 安装Python依赖
pip install -r requirements.txt
```

### 2. 模型文件

请将以下预训练模型文件放置在 `models/` 目录中：
- `yolov11_car_detection.pt` - YOLOv11车辆检测模型
- `rtdetr_car_detection.pt` - RT-DETR车辆检测模型  
- `deeplabv3_water.pt` - DeepLabV3水面分割模型
- `yolov11_seg_water.pt` - YOLOv11水面分割模型

### 3. 运行应用

#### 桌面版
```bash
python main.py desktop
```

#### Web后端服务
```bash
python main.py web-backend
```

Web服务将在 http://localhost:8000 启动

## 开发状态

- ✅ 项目结构和数据模型
- 🚧 核心引擎模块 (进行中)
- ⏳ 桌面GUI应用
- ⏳ Web应用后端
- ⏳ Web应用前端
- ⏳ 部署配置

## 技术栈

- **深度学习**: PyTorch, YOLOv11, RT-DETR, DeepLabV3
- **图像处理**: OpenCV, PIL, NumPy
- **桌面GUI**: PyQt6
- **Web后端**: FastAPI, Uvicorn
- **Web前端**: React, TypeScript
- **部署**: Docker, Nginx

## 许可证

MIT License