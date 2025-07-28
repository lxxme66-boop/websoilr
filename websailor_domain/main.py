#!/usr/bin/env python3
"""
WebSailor-based TCL Industrial Domain Dataset Construction System
基于WebSailor方法的TCL工业垂域数据集构建系统

主要功能：
1. 从TCL工业领域文本构建知识图谱
2. 使用子图采样生成多样化的任务场景
3. 基于子图生成复杂问题
4. 应用模糊化处理增加问题难度
5. 生成推理轨迹
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from core.knowledge_graph_builder import KnowledgeGraphBuilder
from core.subgraph_sampler import SubgraphSampler
from core.question_generator import QuestionGenerator
from core.obfuscation_processor import ObfuscationProcessor
from core.trajectory_generator import TrajectoryGenerator
from core.data_synthesizer import DataSynthesizer
from utils.nlp_utils import setup_nlp_models
from utils.text_utils import load_config


def setup_logging(log_level="INFO"):
    """设置日志系统"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('websailor_tcl.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="WebSailor-based TCL Industrial Domain Dataset Construction"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "build_kg", "sample", "generate", "synthesize"],
        default="full",
        help="运行模式"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="input_texts",
        help="输入文本目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dataset",
        help="输出数据集目录"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=== WebSailor TCL工业垂域数据集构建系统启动 ===")
    
    # 加载配置
    config = load_config(args.config)
    logger.info(f"加载配置文件: {args.config}")
    
    # 设置NLP模型
    logger.info("初始化NLP模型...")
    setup_nlp_models(config)
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    if args.mode == "full" or args.mode == "build_kg":
        # 步骤1: 构建知识图谱
        logger.info("步骤1: 从TCL工业文本构建知识图谱...")
        kg_builder = KnowledgeGraphBuilder(config)
        knowledge_graph = kg_builder.build_from_texts(args.input_dir)
        
        # 保存知识图谱
        kg_path = output_path / "knowledge_graphs.json"
        kg_builder.save_graph(knowledge_graph, kg_path)
        logger.info(f"知识图谱已保存到: {kg_path}")
        
        if args.mode == "build_kg":
            return
    
    if args.mode == "full" or args.mode == "sample":
        # 步骤2: 子图采样
        logger.info("步骤2: 执行子图采样...")
        sampler = SubgraphSampler(config)
        
        # 加载知识图谱
        kg_path = output_path / "knowledge_graphs.json"
        if not kg_path.exists() and args.mode == "sample":
            logger.error("知识图谱文件不存在，请先运行build_kg模式")
            return
            
        knowledge_graph = kg_builder.load_graph(kg_path) if args.mode == "full" else None
        
        # 采样子图
        subgraphs = sampler.sample_diverse_subgraphs(
            knowledge_graph,
            num_samples=config.get("num_subgraph_samples", 1000)
        )
        
        # 保存子图
        subgraphs_path = output_path / "sampled_subgraphs.json"
        sampler.save_subgraphs(subgraphs, subgraphs_path)
        logger.info(f"采样了{len(subgraphs)}个子图，保存到: {subgraphs_path}")
        
        if args.mode == "sample":
            return
    
    if args.mode == "full" or args.mode == "generate":
        # 步骤3: 问题生成
        logger.info("步骤3: 基于子图生成问题...")
        generator = QuestionGenerator(config)
        
        # 加载子图
        subgraphs_path = output_path / "sampled_subgraphs.json"
        if not subgraphs_path.exists() and args.mode == "generate":
            logger.error("子图文件不存在，请先运行sample模式")
            return
            
        subgraphs = sampler.load_subgraphs(subgraphs_path) if args.mode == "full" else None
        
        # 生成问题
        qa_pairs = generator.generate_questions(subgraphs)
        
        # 步骤4: 模糊化处理
        logger.info("步骤4: 应用模糊化处理...")
        obfuscator = ObfuscationProcessor(config)
        obfuscated_qa_pairs = obfuscator.process_qa_pairs(qa_pairs)
        
        # 步骤5: 生成推理轨迹
        logger.info("步骤5: 生成推理轨迹...")
        trajectory_gen = TrajectoryGenerator(config)
        qa_with_trajectories = trajectory_gen.generate_trajectories(obfuscated_qa_pairs)
        
        # 保存QA对
        qa_path = output_path / "qa_pairs.json"
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_with_trajectories, f, ensure_ascii=False, indent=2)
        logger.info(f"生成了{len(qa_with_trajectories)}个QA对，保存到: {qa_path}")
        
        if args.mode == "generate":
            return
    
    if args.mode == "full" or args.mode == "synthesize":
        # 步骤6: 数据综合
        logger.info("步骤6: 综合数据集...")
        synthesizer = DataSynthesizer(config)
        
        # 加载所有数据
        kg_path = output_path / "knowledge_graphs.json"
        qa_path = output_path / "qa_pairs.json"
        
        if not qa_path.exists() and args.mode == "synthesize":
            logger.error("QA对文件不存在，请先运行generate模式")
            return
        
        # 综合数据集
        final_dataset = synthesizer.synthesize_dataset(
            kg_path=kg_path,
            qa_path=qa_path,
            output_path=output_path
        )
        
        # 生成统计信息
        stats_path = output_path / "statistics.json"
        stats = synthesizer.generate_statistics(final_dataset)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集构建完成！统计信息保存到: {stats_path}")
        logger.info(f"数据集统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
    
    logger.info("=== 系统运行完成 ===")


if __name__ == "__main__":
    main()