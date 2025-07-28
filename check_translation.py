#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
翻译质量检查工具
用于验证翻译的完整性和覆盖率
"""

import os
import sys
from pathlib import Path
import argparse

def check_translations(input_dir, output_dir, min_coverage=80):
    """检查翻译文件的完整性"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    if not output_path.exists():
        print(f"❌ 输出目录不存在: {output_dir}")
        return
    
    input_files = list(input_path.glob("*.txt"))
    
    if not input_files:
        print("❌ 输入目录中没有txt文件")
        return
    
    print(f"📊 翻译质量检查报告")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"最低覆盖率要求: {min_coverage}%")
    print("=" * 60)
    
    total_files = len(input_files)
    translated_files = 0
    low_coverage_files = []
    missing_files = []
    
    for input_file in sorted(input_files):
        output_file = output_path / f"{input_file.stem}_中文.txt"
        
        if output_file.exists():
            translated_files += 1
            
            # 计算覆盖率
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    input_content = f.read()
                    input_size = len(input_content)
                
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_content = f.read()
                    # 去除头部元信息
                    if output_content.startswith("#"):
                        lines = output_content.split('\n')
                        for i, line in enumerate(lines):
                            if not line.startswith("#") and line.strip():
                                output_content = '\n'.join(lines[i:])
                                break
                    output_size = len(output_content)
                
                if input_size > 0:
                    coverage = (output_size / input_size) * 100
                else:
                    coverage = 100
                
                status = "✅" if coverage >= min_coverage else "⚠️"
                print(f"{status} {input_file.name:<40} 覆盖率: {coverage:6.1f}%")
                
                if coverage < min_coverage:
                    low_coverage_files.append((input_file.name, coverage))
                    
            except Exception as e:
                print(f"❌ {input_file.name:<40} 读取错误: {e}")
        else:
            missing_files.append(input_file.name)
            print(f"❌ {input_file.name:<40} 未找到翻译文件")
    
    # 汇总报告
    print("\n" + "=" * 60)
    print("📈 汇总统计")
    print("=" * 60)
    print(f"总文件数: {total_files}")
    print(f"已翻译: {translated_files} ({translated_files/total_files*100:.1f}%)")
    print(f"未翻译: {len(missing_files)}")
    print(f"低覆盖率: {len(low_coverage_files)}")
    
    if missing_files:
        print(f"\n⚠️  未翻译的文件 ({len(missing_files)}个):")
        for f in missing_files[:10]:  # 只显示前10个
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... 还有 {len(missing_files)-10} 个文件")
    
    if low_coverage_files:
        print(f"\n⚠️  低覆盖率文件 (< {min_coverage}%):")
        for f, cov in sorted(low_coverage_files, key=lambda x: x[1])[:10]:
            print(f"  - {f:<40} {cov:6.1f}%")
        if len(low_coverage_files) > 10:
            print(f"  ... 还有 {len(low_coverage_files)-10} 个文件")
    
    # 生成需要重新翻译的文件列表
    if missing_files or low_coverage_files:
        retry_list = missing_files + [f[0] for f in low_coverage_files]
        retry_file = "retry_files.txt"
        with open(retry_file, 'w', encoding='utf-8') as f:
            for fname in retry_list:
                f.write(f"{fname}\n")
        print(f"\n💾 需要重新翻译的文件列表已保存到: {retry_file}")
    
    return translated_files == total_files and not low_coverage_files

def main():
    parser = argparse.ArgumentParser(description='检查翻译文件的完整性和覆盖率')
    parser.add_argument('input_dir', help='输入文件目录')
    parser.add_argument('output_dir', help='输出文件目录')
    parser.add_argument('--min-coverage', type=float, default=80, 
                        help='最低覆盖率要求 (默认: 80%)')
    
    args = parser.parse_args()
    
    success = check_translations(args.input_dir, args.output_dir, args.min_coverage)
    
    if success:
        print("\n✅ 所有文件翻译完成且质量合格!")
        sys.exit(0)
    else:
        print("\n⚠️  部分文件需要重新翻译或检查")
        sys.exit(1)

if __name__ == "__main__":
    main()