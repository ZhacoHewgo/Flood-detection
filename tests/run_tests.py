#!/usr/bin/env python3
"""
测试运行器
Test Runner for Desktop Integration Tests
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


def main():
    """运行集成测试"""
    print("开始运行桌面应用集成测试...")
    
    # 设置测试参数
    test_args = [
        "-v",  # 详细输出
        "--tb=short",  # 简短的错误回溯
        "-x",  # 遇到第一个失败就停止
        str(Path(__file__).parent),  # 测试目录
    ]
    
    # 运行测试
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("\n✅ 所有集成测试通过!")
    else:
        print(f"\n❌ 测试失败，退出代码: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())