#!/usr/bin/env python3
"""
测试脚本：检查项目依赖是否可用
"""

import sys
print(f"Python version: {sys.version}")

# 测试基础模块
try:
    import json
    print("✓ JSON module loaded")
except ImportError as e:
    print(f"✗ JSON module failed: {e}")

try:
    import pathlib
    print("✓ pathlib module loaded")
except ImportError as e:
    print(f"✗ pathlib module failed: {e}")

# 测试第三方依赖
dependencies = [
    'networkx',
    'pandas', 
    'numpy',
    'collections',
    'logging',
    'random',
    'typing',
    'datetime',
    'argparse'
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep} loaded")
    except ImportError as e:
        print(f"✗ {dep} failed: {e}")

# 测试项目模块结构
print("\n=== 测试项目结构 ===")
try:
    from pathlib import Path
    
    # 检查目录结构
    dirs_to_check = ['core', 'utils', 'templates', 'input_texts', 'output_dataset']
    for dir_name in dirs_to_check:
        if Path(dir_name).exists():
            print(f"✓ Directory {dir_name} exists")
        else:
            print(f"✗ Directory {dir_name} missing")
    
    # 检查关键文件
    files_to_check = ['config.json', 'requirements.txt', 'main.py']
    for file_name in files_to_check:
        if Path(file_name).exists():
            print(f"✓ File {file_name} exists")
        else:
            print(f"✗ File {file_name} missing")
            
except Exception as e:
    print(f"✗ Structure check failed: {e}")

print("\n=== 测试完成 ===")