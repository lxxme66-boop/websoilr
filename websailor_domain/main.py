#!/usr/bin/env python3
"""
WebSailor Domain-Specific Dataset Construction System
专门用于TCL工业垂域数据集构建的主入口文件

基于WebSailor核心思想:
1. 子图采样 - 从知识图中抽取不同拓扑的子图
2. 问题生成 - 基于子图设计多样化QA问题
3. 模糊化处理 - 添加干扰信息增加推理难度
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from core.knowledge_graph_builder import KnowledgeGraphBuilder
from core.subgraph_sampler import SubgraphSampler
from core.question_generator import QuestionGenerator
from core.obfuscation_processor import ObfuscationProcessor
from core.trajectory_generator import TrajectoryGenerator
from core.data_synthesizer import DataSynthesizer


def setup_logging(output_dir: Path):
    """设置日志系统"""
    log_file = output_dir / f"dataset_construction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """主函数 - 协调整个数据集构建流程"""
    parser = argparse.ArgumentParser(
        description="WebSailor Domain-Specific Dataset Construction System"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="配置文件路径"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_texts"),
        help="输入文本目录"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_dataset"),
        help="输出数据集目录"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="TCL_Industry",
        help="垂直领域名称"
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting WebSailor domain-specific dataset construction for {args.domain}")
    
    # 加载配置
    config = load_config(args.config)
    logger.info("Configuration loaded successfully")
    
    try:
        # 1. 构建知识图谱
        logger.info("Step 1: Building Knowledge Graph from domain texts...")
        kg_builder = KnowledgeGraphBuilder(config['knowledge_graph'])
        knowledge_graph = kg_builder.build_from_texts(args.input_dir)
        logger.info(f"Knowledge graph built with {len(knowledge_graph.nodes)} nodes and {len(knowledge_graph.edges)} edges")
        
        # 2. 子图采样 (WebSailor核心)
        logger.info("Step 2: Sampling subgraphs with different topologies...")
        sampler = SubgraphSampler(config['subgraph_sampling'])
        subgraphs = sampler.sample_diverse_subgraphs(knowledge_graph)
        logger.info(f"Sampled {len(subgraphs)} subgraphs representing different task scenarios")
        
        # 3. 问题生成 (WebSailor核心)
        logger.info("Step 3: Generating questions based on subgraphs...")
        question_gen = QuestionGenerator(config['question_generation'])
        qa_pairs = question_gen.generate_from_subgraphs(subgraphs)
        logger.info(f"Generated {len(qa_pairs)} QA pairs covering multiple question types")
        
        # 4. 模糊化处理 (WebSailor核心)
        logger.info("Step 4: Applying obfuscation to increase reasoning difficulty...")
        obfuscator = ObfuscationProcessor(config['obfuscation'])
        obfuscated_qa_pairs = obfuscator.process_qa_pairs(qa_pairs, knowledge_graph)
        logger.info("Obfuscation completed - added ambiguity and distractors")
        
        # 5. 推理轨迹生成
        logger.info("Step 5: Generating reasoning trajectories...")
        trajectory_gen = TrajectoryGenerator(config['trajectory_generation'])
        trajectories = trajectory_gen.generate_trajectories(obfuscated_qa_pairs, knowledge_graph)
        logger.info(f"Generated {len(trajectories)} reasoning trajectories")
        
        # 6. 数据综合与输出
        logger.info("Step 6: Synthesizing final dataset...")
        synthesizer = DataSynthesizer(config['data_synthesis'])
        dataset = synthesizer.synthesize(
            knowledge_graph=knowledge_graph,
            qa_pairs=obfuscated_qa_pairs,
            trajectories=trajectories,
            domain=args.domain
        )
        
        # 保存数据集
        output_path = args.output_dir / f"{args.domain}_dataset_{datetime.now().strftime('%Y%m%d')}.json"
        synthesizer.save_dataset(dataset, output_path)
        logger.info(f"Dataset saved to {output_path}")
        
        # 生成统计信息
        stats = synthesizer.generate_statistics(dataset)
        stats_path = args.output_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Statistics saved to {stats_path}")
        
        logger.info("Dataset construction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset construction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()