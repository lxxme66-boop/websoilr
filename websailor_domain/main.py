"""
WebSailor TCL工业垂域数据集生成主程序
基于WebSailor方法论构建高质量的垂域数据集
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from core.data_synthesizer import DataSynthesizer

# 设置日志
def setup_logging(log_file='websailor_tcl.log', level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='WebSailor TCL工业垂域数据集生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置生成数据集
  python main.py --input_dir input_texts --output_dir output_dataset
  
  # 指定子图数量和每个子图的问题数
  python main.py --input_dir input_texts --output_dir output_dataset --num_subgraphs 500 --questions_per_subgraph 10
  
  # 使用自定义配置文件
  python main.py --config my_config.json --input_dir input_texts --output_dir output_dataset
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='配置文件路径 (默认: config.json)'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='输入文本目录路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出数据集目录路径'
    )
    
    parser.add_argument(
        '--num_subgraphs',
        type=int,
        default=1000,
        help='要采样的子图数量 (默认: 1000)'
    )
    
    parser.add_argument(
        '--questions_per_subgraph',
        type=int,
        default=5,
        help='每个子图生成的问题数 (默认: 5)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别 (默认: INFO)'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 检查输入目录
        input_path = Path(args.input_dir)
        if not input_path.exists():
            logger.error(f"输入目录不存在: {args.input_dir}")
            sys.exit(1)
            
        # 创建输出目录
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据综合器
        logger.info("初始化数据综合器...")
        synthesizer = DataSynthesizer(config)
        
        # 生成数据集
        logger.info("开始生成数据集...")
        stats = synthesizer.synthesize_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_subgraphs=args.num_subgraphs,
            questions_per_subgraph=args.questions_per_subgraph
        )
        
        # 打印统计信息
        logger.info("=" * 50)
        logger.info("数据集生成完成！统计信息：")
        logger.info(f"知识图谱: {stats['knowledge_graph']['num_nodes']}个节点, "
                   f"{stats['knowledge_graph']['num_edges']}条边")
        logger.info(f"子图: {stats['subgraphs']['total_subgraphs']}个")
        logger.info(f"问题: {stats['questions']['total_questions']}个")
        logger.info(f"数据集分割: 训练集{stats['splits']['train_size']}条, "
                   f"验证集{stats['splits']['val_size']}条, "
                   f"测试集{stats['splits']['test_size']}条")
        logger.info("=" * 50)
        
        # 保存运行配置
        run_config = {
            'config_file': args.config,
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'num_subgraphs': args.num_subgraphs,
            'questions_per_subgraph': args.questions_per_subgraph,
            'statistics': stats
        }
        
        run_config_path = output_path / 'run_config.json'
        with open(run_config_path, 'w', encoding='utf-8') as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"运行配置已保存到: {run_config_path}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()