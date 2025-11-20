# 🌊 积水车辆检测系统

基于深度学习的积水车辆检测系统，提供桌面版和Web版两种使用方式。Web版已实现与桌面版功能完全对等，支持批量处理、性能监控等高级功能。

## ✨ 核心功能

### 🔥 分析功能
- 🚗 **车辆检测**: YOLOv11、RT-DETR多种检测模型
- 🌊 **水面分割**: DeepLabV3、YOLOv11分割模型  
- 📊 **淹没分析**: 自动计算车辆淹没等级（轻度/中度/重度）
- 🔄 **任务模式**: 车辆检测/水面分割/组合分析

### 💻 应用版本
- 🖥️ **桌面应用**: PyQt6构建的专业界面
- 🌐 **Web应用**: React + FastAPI现代化架构
- 📱 **移动适配**: Web版支持手机平板访问

### ⚡ 高级功能
- 📦 **批量处理**: 支持多文件并行分析
- 🧠 **智能缓存**: 避免重复分析，提高效率
- 📊 **性能监控**: 实时监控系统资源使用
- 📈 **统计分析**: 详细的淹没情况统计和可视化
- 💾 **结果导出**: JSON格式结果导出

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

## 🚀 快速开始

### Web版（推荐）

```bash
# 1. 克隆项目
git clone <repository-url>
cd flood-detection-system

# 2. 一键启动Web应用
python start_web_complete.py

# 3. 访问应用
# 前端: http://localhost:3000
# 后端API: http://localhost:8000
# API文档: http://localhost:8000/docs
```

### 桌面版

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动桌面应用
python main.py
# 或使用脚本
./start_desktop.sh
```

## 📖 详细文档

- **[Web端完整指南](WEB_COMPLETE_GUIDE.md)** - Web版详细使用说明
- **[功能完善总结](WEB_FINAL_SUMMARY.md)** - Web端完善成果总览
- **[系统技术文档](SYSTEM_DOCUMENTATION.md)** - 技术架构和实现细节
- **[功能对比分析](WEB_DESKTOP_FEATURE_COMPARISON.md)** - Web版与桌面版对比
- **[文档索引](DOCUMENTATION_INDEX.md)** - 所有文档的导航指南

## 🛠️ 部署方案

### 开发环境
```bash
# 使用一键启动脚本（推荐）
python start_web_complete.py --dev

# 或手动启动
python deploy_web_complete.py local --mode development
```

### 生产环境
```bash
# Docker部署（推荐）
python deploy_web_complete.py docker --mode production

# 本地部署
python deploy_web_complete.py local --mode production
```

### 测试验证
```bash
# 运行完整功能测试
python test_web_complete.py

# 查看测试报告
cat web_test_report.json
```

## 📁 模型文件

请将以下预训练模型文件放置在 `models/` 目录中：
- `yolov11_car_detection.pt` - YOLOv11车辆检测模型
- `rtdetr_car_detection.pt` - RT-DETR车辆检测模型  
- `deeplabv3_water.pt` - DeepLabV3水面分割模型
- `yolov11_seg_water.pt` - YOLOv11水面分割模型

## ✅ 开发状态

### 已完成功能
- ✅ **核心引擎**: 完整的AI分析引擎
- ✅ **桌面应用**: 功能完整的PyQt6应用
- ✅ **Web后端**: FastAPI异步后端服务
- ✅ **Web前端**: React现代化前端界面
- ✅ **批量处理**: 多文件并行分析
- ✅ **性能监控**: 系统资源监控
- ✅ **智能缓存**: 结果缓存优化
- ✅ **部署配置**: Docker容器化部署
- ✅ **测试验证**: 完整的测试套件
- ✅ **文档完善**: 详细的使用和技术文档

### 功能对等性
- ✅ **Web端 = 桌面端**: 100%功能对等
- ✅ **跨平台支持**: Windows/macOS/Linux
- ✅ **移动端适配**: 响应式Web设计
- ✅ **部署便利**: 一键启动和部署

## 🔧 技术栈

### 🧠 AI引擎
- **深度学习**: PyTorch, YOLOv11, RT-DETR, DeepLabV3
- **图像处理**: OpenCV, PIL, NumPy, scikit-image
- **模型管理**: Ultralytics, 自定义模型加载器

### 💻 桌面应用
- **GUI框架**: PyQt6
- **异步处理**: QThread, 信号槽机制
- **图像显示**: 自定义图像组件

### 🌐 Web应用
- **后端**: FastAPI, Uvicorn, 异步处理
- **前端**: React 18, TypeScript, Ant Design
- **状态管理**: React Hooks, Context API
- **样式**: CSS3, 响应式设计

### 🚀 部署运维
- **容器化**: Docker, docker-compose
- **Web服务器**: Nginx
- **监控**: 性能监控, 健康检查
- **自动化**: 部署脚本, 测试脚本

## 🎯 使用场景

- **🏢 企业应用**: 洪水灾害评估和车辆损失统计
- **🏛️ 政府部门**: 城市防洪规划和应急响应
- **🎓 科研教育**: 计算机视觉和深度学习研究
- **👨‍💻 个人项目**: 图像分析和AI应用开发

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [Issue](../../issues)
- 发送邮件到项目维护者

---

⭐ 如果这个项目对你有帮助，请给它一个星标！