#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量翻译辅助工具
提供文件预处理、批量管理、智能调度等功能
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
import json
from datetime import datetime

class TranslationHelper:
    def __init__(self):
        self.config_file = "translation_config.json"
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "history": [],
                "preferences": {
                    "default_model": "2",  # 7B模型
                    "default_chunk_size": "2",  # 中等片段
                    "output_dir": "./chinese_translations"
                }
            }
    
    def save_config(self):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def preprocess_files(self, input_dir):
        """预处理文件"""
        print("🔍 预处理文件...")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 目录不存在: {input_dir}")
            return
        
        txt_files = list(input_path.glob("*.txt"))
        
        if not txt_files:
            print("❌ 没有找到txt文件")
            return
        
        print(f"📁 找到 {len(txt_files)} 个文件")
        
        # 文件分析
        file_stats = []
        total_size = 0
        encoding_issues = []
        
        for txt_file in txt_files:
            try:
                # 检查编码
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    size = len(content)
                    lines = content.count('\n')
                    
                file_stats.append({
                    "name": txt_file.name,
                    "size": size,
                    "lines": lines,
                    "path": str(txt_file)
                })
                total_size += size
                
            except UnicodeDecodeError:
                encoding_issues.append(txt_file.name)
                # 尝试其他编码
                for encoding in ['gbk', 'gb2312', 'latin-1']:
                    try:
                        with open(txt_file, 'r', encoding=encoding) as f:
                            content = f.read()
                        # 转换为UTF-8
                        backup_path = txt_file.with_suffix('.bak')
                        shutil.copy(txt_file, backup_path)
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"✅ 转换编码: {txt_file.name} ({encoding} → UTF-8)")
                        break
                    except:
                        continue
        
        # 显示统计
        print("\n📊 文件统计:")
        print(f"  总文件数: {len(txt_files)}")
        print(f"  总大小: {total_size:,} 字符 ({total_size/1024/1024:.1f} MB)")
        print(f"  平均大小: {total_size/len(txt_files):,.0f} 字符")
        
        if encoding_issues:
            print(f"\n⚠️  编码问题文件: {len(encoding_issues)}个")
            for f in encoding_issues[:5]:
                print(f"  - {f}")
        
        # 按大小分组
        small_files = [f for f in file_stats if f['size'] < 10000]
        medium_files = [f for f in file_stats if 10000 <= f['size'] < 100000]
        large_files = [f for f in file_stats if f['size'] >= 100000]
        
        print("\n📏 文件大小分布:")
        print(f"  小文件 (<10K字符): {len(small_files)}个")
        print(f"  中等文件 (10K-100K): {len(medium_files)}个")
        print(f"  大文件 (>100K): {len(large_files)}个")
        
        # 生成批处理建议
        print("\n💡 批处理建议:")
        if large_files:
            print("  - 大文件建议使用小片段(800字符)以确保完整翻译")
        if len(txt_files) > 100:
            print("  - 文件数量较多，建议使用后台运行或screen会话")
        if total_size > 10 * 1024 * 1024:  # 10MB
            print("  - 数据量较大，建议使用7B模型平衡速度和质量")
        
        # 保存文件列表
        file_list_path = "file_list.json"
        with open(file_list_path, 'w', encoding='utf-8') as f:
            json.dump(file_stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 文件列表已保存到: {file_list_path}")
        
        return file_stats
    
    def split_batch(self, input_dir, batch_size=50):
        """将文件分批处理"""
        print(f"📦 分批处理文件 (每批{batch_size}个)...")
        
        input_path = Path(input_dir)
        txt_files = list(input_path.glob("*.txt"))
        
        if not txt_files:
            print("❌ 没有找到txt文件")
            return
        
        # 创建批次目录
        batch_dir = Path("translation_batches")
        batch_dir.mkdir(exist_ok=True)
        
        # 清理旧批次
        for old_batch in batch_dir.glob("batch_*"):
            shutil.rmtree(old_batch)
        
        # 分批
        num_batches = (len(txt_files) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            batch_path = batch_dir / f"batch_{i+1:03d}"
            batch_path.mkdir(exist_ok=True)
            
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(txt_files))
            
            for j in range(start_idx, end_idx):
                # 创建符号链接而不是复制文件
                src = txt_files[j].absolute()
                dst = batch_path / txt_files[j].name
                if os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            
            batch_files = end_idx - start_idx
            print(f"  批次 {i+1}: {batch_files} 个文件")
        
        print(f"\n✅ 已创建 {num_batches} 个批次")
        print(f"📁 批次目录: {batch_dir}")
        
        # 生成批处理脚本
        script_path = "run_batches.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# 批量翻译脚本\n")
            f.write("# 自动处理所有批次\n\n")
            
            for i in range(num_batches):
                f.write(f"echo '处理批次 {i+1}/{num_batches}...'\n")
                f.write(f"python translate_deepseek_improved.py < batch_{i+1}_input.txt\n")
                f.write(f"echo '批次 {i+1} 完成'\n")
                f.write("echo '等待10秒...'\n")
                f.write("sleep 10\n\n")
            
            f.write("echo '✅ 所有批次处理完成!'\n")
        
        os.chmod(script_path, 0o755)
        print(f"💾 批处理脚本: {script_path}")
        
        # 为每个批次创建输入文件
        for i in range(num_batches):
            input_file = f"batch_{i+1}_input.txt"
            with open(input_file, 'w', encoding='utf-8') as f:
                # 模型选择
                f.write(f"{self.config['preferences']['default_model']}\n")
                # 输入目录
                f.write(f"{batch_dir}/batch_{i+1:03d}\n")
                # 输出目录
                f.write(f"{self.config['preferences']['output_dir']}/batch_{i+1:03d}\n")
                # 分片大小
                f.write(f"{self.config['preferences']['default_chunk_size']}\n")
        
        return num_batches
    
    def merge_results(self, output_base_dir="./chinese_translations"):
        """合并批次结果"""
        print("🔀 合并批次结果...")
        
        output_path = Path(output_base_dir)
        batch_dirs = sorted(output_path.glob("batch_*"))
        
        if not batch_dirs:
            print("❌ 没有找到批次目录")
            return
        
        # 创建合并目录
        merged_dir = output_path / "merged"
        merged_dir.mkdir(exist_ok=True)
        
        total_files = 0
        for batch_dir in batch_dirs:
            txt_files = list(batch_dir.glob("*_中文.txt"))
            for txt_file in txt_files:
                # 移动到合并目录
                dst = merged_dir / txt_file.name
                shutil.move(str(txt_file), str(dst))
                total_files += 1
        
        print(f"✅ 已合并 {total_files} 个文件到: {merged_dir}")
        
        # 清理批次目录
        for batch_dir in batch_dirs:
            if batch_dir.is_dir():
                shutil.rmtree(batch_dir)
        
        return total_files
    
    def create_report(self, input_dir, output_dir):
        """创建翻译报告"""
        print("📊 生成翻译报告...")
        
        report_path = "translation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DeepSeek 翻译报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {input_dir}\n")
            f.write(f"输出目录: {output_dir}\n\n")
            
            # 运行质量检查
            import subprocess
            result = subprocess.run(
                [sys.executable, "check_translation.py", input_dir, output_dir],
                capture_output=True,
                text=True
            )
            
            f.write(result.stdout)
            
            # 添加历史记录
            self.config["history"].append({
                "time": datetime.now().isoformat(),
                "input_dir": input_dir,
                "output_dir": output_dir
            })
            self.save_config()
        
        print(f"✅ 报告已生成: {report_path}")
        return report_path

def main():
    parser = argparse.ArgumentParser(description='批量翻译辅助工具')
    parser.add_argument('action', choices=['preprocess', 'split', 'merge', 'report'],
                        help='执行的操作')
    parser.add_argument('--input-dir', '-i', help='输入目录')
    parser.add_argument('--output-dir', '-o', help='输出目录')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                        help='每批文件数量 (默认: 50)')
    
    args = parser.parse_args()
    
    helper = TranslationHelper()
    
    if args.action == 'preprocess':
        if not args.input_dir:
            print("❌ 请指定输入目录")
            sys.exit(1)
        helper.preprocess_files(args.input_dir)
    
    elif args.action == 'split':
        if not args.input_dir:
            print("❌ 请指定输入目录")
            sys.exit(1)
        helper.split_batch(args.input_dir, args.batch_size)
    
    elif args.action == 'merge':
        output_dir = args.output_dir or "./chinese_translations"
        helper.merge_results(output_dir)
    
    elif args.action == 'report':
        if not args.input_dir or not args.output_dir:
            print("❌ 请指定输入和输出目录")
            sys.exit(1)
        helper.create_report(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()