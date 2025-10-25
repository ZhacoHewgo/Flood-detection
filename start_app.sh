#!/bin/bash
# 启动积水车辆检测应用程序

echo "🚀 启动积水车辆检测应用程序..."

# 设置Qt环境变量
export QT_QPA_PLATFORM_PLUGIN_PATH="/Users/zhaco/miniconda3/envs/yolov11/lib/python3.10/site-packages/PyQt6/Qt6/plugins/platforms"
export QT_QPA_PLATFORM="cocoa"

echo "🔧 Qt环境已设置"
echo "📱 启动桌面应用..."

# 启动应用
python main.py desktop

echo "👋 应用已退出"