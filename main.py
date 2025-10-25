#!/usr/bin/env python3
"""
积水车辆检测应用程序主入口
Flood Vehicle Detection Application Main Entry Point
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_desktop_app():
    """运行桌面GUI应用"""
    try:
        from flood_detection_app.desktop.main_window import main as desktop_main
        desktop_main()
    except ImportError as e:
        print(f"桌面应用依赖缺失: {e}")
        print("请安装PyQt6: pip install PyQt6")
        sys.exit(1)
    except Exception as e:
        print(f"桌面应用启动失败: {e}")
        sys.exit(1)


def run_web_backend():
    """运行Web后端服务"""
    try:
        import uvicorn
        from flood_detection_app.web.backend.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except ImportError as e:
        print(f"Web后端依赖缺失: {e}")
        print("请安装FastAPI和Uvicorn: pip install fastapi uvicorn")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="积水车辆检测应用程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py desktop          # 运行桌面GUI应用
  python main.py web-backend      # 运行Web后端服务
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["desktop", "web-backend"],
        help="运行模式: desktop (桌面应用) 或 web-backend (Web后端)"
    )
    
    args = parser.parse_args()
    
    print(f"启动积水车辆检测应用程序 - {args.mode} 模式")
    
    if args.mode == "desktop":
        run_desktop_app()
    elif args.mode == "web-backend":
        run_web_backend()


if __name__ == "__main__":
    main()