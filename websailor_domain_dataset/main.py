#!/usr/bin/env python3
"""
TCL工业垂域数据集构造主程序
基于WebSailor的核心思想：子图采样、问题生成、模糊化处理
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

from core.knowledge_graph_builder import KnowledgeGraphBuilder
from core.subgraph_sampler import SubgraphSampler
from core.question_generator import QuestionGenerator
from core.obfuscation_processor import ObfuscationProcessor
from core.trajectory_generator import TrajectoryGenerator
from core.data_synthesizer import DataSynthesizer

def setup_logging(log_level: str = "INFO") -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tcl_dataset_construction.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='TCL工业垂域数据集构造')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--input_dir', type=str, default='input_texts', help='输入文本目录')
    parser.add_argument('--output_dir', type=str, default='output_dataset', help='输出数据集目录')
    parser.add_argument('--log_level', type=str, default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("开始TCL工业垂域数据集构造...")
    
    try:
        # 加载配置
        config = load_config(args.config)
        logger.info(f"配置加载完成: {config}")
        
        # 创建输出目录
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. 构建知识图谱（基于TCL工业领域文本）
        logger.info("步骤1: 构建TCL工业领域知识图谱...")
        kg_builder = KnowledgeGraphBuilder(config['knowledge_graph'])
        knowledge_graph = kg_builder.build_from_texts(args.input_dir)
        logger.info(f"知识图谱构建完成，包含 {len(knowledge_graph.nodes)} 个节点")
        
        # 2. 子图采样 - WebSailor核心思想
        logger.info("步骤2: 执行子图采样（WebSailor核心）...")
        subgraph_sampler = SubgraphSampler(config['subgraph_sampling'])
        subgraphs = subgraph_sampler.sample_subgraphs(
            knowledge_graph, 
            num_subgraphs=config['dataset']['num_subgraphs']
        )
        logger.info(f"子图采样完成，生成 {len(subgraphs)} 个子图")
        
        # 3. 问题生成 - 基于子图的多样化问题
        logger.info("步骤3: 基于子图生成问题...")
        question_generator = QuestionGenerator(config['question_generation'])
        qa_pairs = []
        for i, subgraph in enumerate(subgraphs):
            questions = question_generator.generate_questions(subgraph)
            qa_pairs.extend(questions)
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(subgraphs)} 个子图")
        
        logger.info(f"问题生成完成，共生成 {len(qa_pairs)} 个QA对")
        
        # 4. 模糊化处理 - WebSailor的不确定性增强
        logger.info("步骤4: 执行模糊化处理...")
        obfuscation_processor = ObfuscationProcessor(config['obfuscation'])
        obfuscated_qa_pairs = obfuscation_processor.process_qa_pairs(qa_pairs)
        logger.info(f"模糊化处理完成，处理了 {len(obfuscated_qa_pairs)} 个QA对")
        
        # 5. 推理轨迹生成
        logger.info("步骤5: 生成推理轨迹...")
        trajectory_generator = TrajectoryGenerator(config['trajectory_generation'])
        trajectories = trajectory_generator.generate_trajectories(obfuscated_qa_pairs, subgraphs)
        logger.info(f"推理轨迹生成完成，共 {len(trajectories)} 条轨迹")
        
        # 6. 数据综合与输出
        logger.info("步骤6: 数据综合与输出...")
        data_synthesizer = DataSynthesizer(config['data_synthesis'])
        final_dataset = data_synthesizer.synthesize_dataset(
            qa_pairs=obfuscated_qa_pairs,
            trajectories=trajectories,
            knowledge_graphs=subgraphs
        )
        
        # 保存数据集
        output_files = {
            'qa_pairs.json': final_dataset['qa_pairs'],
            'trajectories.json': final_dataset['trajectories'], 
            'knowledge_graphs.json': final_dataset['knowledge_graphs'],
            'statistics.json': final_dataset['statistics']
        }
        
        for filename, data in output_files.items():
            filepath = output_path / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"保存文件: {filepath}")
        
        logger.info("TCL工业垂域数据集构造完成！")
        logger.info(f"数据集统计: {final_dataset['statistics']}")
        
    except Exception as e:
        logger.error(f"数据集构造过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()