#!/bin/bash

# 积水车辆检测桌面应用启动脚本
# Desktop Application Startup Script for Flood Vehicle Detection

echo "启动积水车辆检测桌面应用..."

# 设置Qt插件路径
export QT_PLUGIN_PATH="$CONDA_PREFIX/lib/qt6/plugins:$CONDA_PREFIX/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_PLUGIN_PATH"

# 在macOS上强制使用cocoa平台
if [[ "$OSTYPE" == "darwin"* ]]; then
    export QT_QPA_PLATFORM="cocoa"
fi

echo "Qt环境变量已设置:"
echo "  QT_PLUGIN_PATH = $QT_PLUGIN_PATH"
echo "  QT_QPA_PLATFORM = $QT_QPA_PLATFORM"

# 启动应用
python main.py desktop