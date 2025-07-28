#!/usr/bin/env python3
"""
WebSailor Domain Dataset Construction Tool
主入口文件 - 用于构造垂域（TCL工业）数据集
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

from core.knowledge_graph_builder import KnowledgeGraphBuilder
from core.subgraph_sampler import SubgraphSampler
from core.question_generator import QuestionGenerator
from core.obfuscation_processor import ObfuscationProcessor
from core.trajectory_generator import TrajectoryGenerator
from core.data_synthesizer import DataSynthesizer
from utils.text_utils import setup_logging


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='WebSailor Domain Dataset Construction Tool')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--input-dir', type=str, default='input_texts', help='输入文本目录')
    parser.add_argument('--output-dir', type=str, default='output_dataset', help='输出数据集目录')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    logger.info("加载配置文件...")
    config = load_config(args.config)
    
    # 初始化各个组件
    logger.info("初始化WebSailor组件...")
    
    # 1. 构建知识图谱
    kg_builder = KnowledgeGraphBuilder(config['knowledge_graph'])
    logger.info("构建领域知识图谱...")
    knowledge_graph = kg_builder.build_from_texts(args.input_dir)
    
    # 2. 子图采样（WebSailor核心思想）
    sampler = SubgraphSampler(config['subgraph_sampling'])
    logger.info("执行子图采样...")
    subgraphs = sampler.sample_subgraphs(knowledge_graph)
    
    # 3. 问题生成
    question_gen = QuestionGenerator(config['question_generation'])
    logger.info("生成问题...")
    qa_pairs = question_gen.generate_questions(subgraphs)
    
    # 4. 模糊化处理（WebSailor核心思想）
    obfuscator = ObfuscationProcessor(config['obfuscation'])
    logger.info("执行模糊化处理...")
    obfuscated_qa = obfuscator.process_qa_pairs(qa_pairs, knowledge_graph)
    
    # 5. 推理轨迹生成
    trajectory_gen = TrajectoryGenerator(config['trajectory_generation'])
    logger.info("生成推理轨迹...")
    trajectories = trajectory_gen.generate_trajectories(obfuscated_qa, knowledge_graph)
    
    # 6. 数据综合
    synthesizer = DataSynthesizer(config['data_synthesis'])
    logger.info("综合数据集...")
    final_dataset = synthesizer.synthesize(
        qa_pairs=obfuscated_qa,
        trajectories=trajectories,
        knowledge_graph=knowledge_graph,
        subgraphs=subgraphs
    )
    
    # 保存结果
    logger.info("保存数据集...")
    
    # 保存QA对
    with open(output_dir / 'qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(final_dataset['qa_pairs'], f, ensure_ascii=False, indent=2)
    
    # 保存推理轨迹
    with open(output_dir / 'trajectories.json', 'w', encoding='utf-8') as f:
        json.dump(final_dataset['trajectories'], f, ensure_ascii=False, indent=2)
    
    # 保存知识图谱
    with open(output_dir / 'knowledge_graphs.json', 'w', encoding='utf-8') as f:
        json.dump(final_dataset['knowledge_graph'], f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    statistics = {
        'timestamp': datetime.now().isoformat(),
        'total_qa_pairs': len(final_dataset['qa_pairs']),
        'total_trajectories': len(final_dataset['trajectories']),
        'total_subgraphs': len(subgraphs),
        'graph_nodes': len(knowledge_graph.nodes),
        'graph_edges': len(knowledge_graph.edges),
        'config': config
    }
    
    with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据集构建完成！")
    logger.info(f"- QA对数量: {statistics['total_qa_pairs']}")
    logger.info(f"- 推理轨迹数量: {statistics['total_trajectories']}")
    logger.info(f"- 子图数量: {statistics['total_subgraphs']}")
    logger.info(f"- 知识图谱节点数: {statistics['graph_nodes']}")
    logger.info(f"- 知识图谱边数: {statistics['graph_edges']}")


if __name__ == '__main__':
    main()