#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSailor Domain Dataset Generator - Main Entry Point
基于WebSailor核心思想构造TCL工业垂域数据集

WebSailor核心思想：
1. 子图采样：从整个知识图中抽取不同拓扑的子图作为问题候选基础
2. 问题生成：基于子图中节点与关系，设计QA问题，覆盖多种问题类型
3. 模糊化处理：模糊描述中间实体或关系，添加冗余或干扰信息

Author: WebSailor Domain Dataset Team
Date: 2025
"""

import json
import os
import logging
from typing import Dict, List, Any
from pathlib import Path

from core.knowledge_graph_builder import KnowledgeGraphBuilder
from core.subgraph_sampler import SubgraphSampler
from core.question_generator import QuestionGenerator
from core.obfuscation_processor import ObfuscationProcessor
from core.trajectory_generator import TrajectoryGenerator
from core.data_synthesizer import DataSynthesizer


class WebSailorDatasetGenerator:
    """
    WebSailor数据集生成器主类
    实现从文本到高质量QA数据集的完整流程
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化数据集生成器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        
        # 初始化核心组件
        self.kg_builder = KnowledgeGraphBuilder(self.config)
        self.subgraph_sampler = SubgraphSampler(self.config)
        self.question_generator = QuestionGenerator(self.config)
        self.obfuscation_processor = ObfuscationProcessor(self.config)
        self.trajectory_generator = TrajectoryGenerator(self.config)
        self.data_synthesizer = DataSynthesizer(self.config)
        
        logging.info("WebSailor数据集生成器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logging.error(f"配置文件 {config_path} 不存在")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"配置文件格式错误: {e}")
            raise
    
    def _setup_logging(self):
        """设置日志"""
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('websailor_dataset.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self):
        """创建必要的目录"""
        directories = [
            "input_texts",
            "output_dataset", 
            "templates",
            "examples/sample_input",
            "examples/sample_output"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logging.info("目录结构创建完成")
    
    def generate_dataset(self) -> Dict[str, Any]:
        """
        生成完整的WebSailor数据集
        
        Returns:
            生成的数据集统计信息
        """
        logging.info("开始生成WebSailor数据集...")
        
        try:
            # 步骤1: 构建知识图谱
            logging.info("步骤1: 构建知识图谱...")
            knowledge_graph = self.kg_builder.build_from_texts(
                self.config["input_texts_dir"]
            )
            
            # 步骤2: 子图采样 (WebSailor核心思想1)
            logging.info("步骤2: 执行子图采样...")
            subgraphs = self.subgraph_sampler.sample_subgraphs(
                knowledge_graph,
                num_subgraphs=self.config["num_subgraphs"],
                sampling_strategies=self.config["sampling_strategies"]
            )
            
            # 步骤3: 问题生成 (WebSailor核心思想2)
            logging.info("步骤3: 生成问题...")
            qa_pairs = []
            for subgraph in subgraphs:
                questions = self.question_generator.generate_questions(
                    subgraph,
                    question_types=self.config["question_types"]
                )
                qa_pairs.extend(questions)
            
            # 步骤4: 模糊化处理 (WebSailor核心思想3)
            logging.info("步骤4: 执行模糊化处理...")
            obfuscated_qa_pairs = self.obfuscation_processor.process_qa_pairs(
                qa_pairs,
                obfuscation_strategies=self.config["obfuscation_strategies"]
            )
            
            # 步骤5: 生成推理轨迹
            logging.info("步骤5: 生成推理轨迹...")
            trajectories = self.trajectory_generator.generate_trajectories(
                obfuscated_qa_pairs,
                knowledge_graph
            )
            
            # 步骤6: 数据综合
            logging.info("步骤6: 数据综合...")
            final_dataset = self.data_synthesizer.synthesize_dataset(
                qa_pairs=obfuscated_qa_pairs,
                trajectories=trajectories,
                knowledge_graph=knowledge_graph,
                subgraphs=subgraphs
            )
            
            # 保存数据集
            self._save_dataset(final_dataset)
            
            # 生成统计信息
            stats = self._generate_statistics(final_dataset)
            
            logging.info("WebSailor数据集生成完成！")
            return stats
            
        except Exception as e:
            logging.error(f"数据集生成失败: {e}")
            raise
    
    def _save_dataset(self, dataset: Dict[str, Any]):
        """保存数据集到文件"""
        output_dir = Path("output_dataset")
        
        # 保存QA对
        with open(output_dir / "qa_pairs.json", 'w', encoding='utf-8') as f:
            json.dump(dataset["qa_pairs"], f, ensure_ascii=False, indent=2)
        
        # 保存推理轨迹
        with open(output_dir / "trajectories.json", 'w', encoding='utf-8') as f:
            json.dump(dataset["trajectories"], f, ensure_ascii=False, indent=2)
        
        # 保存知识图谱
        with open(output_dir / "knowledge_graphs.json", 'w', encoding='utf-8') as f:
            json.dump(dataset["knowledge_graph"], f, ensure_ascii=False, indent=2)
        
        # 保存统计信息
        with open(output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(dataset["statistics"], f, ensure_ascii=False, indent=2)
        
        logging.info("数据集保存完成")
    
    def _generate_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据集统计信息"""
        stats = {
            "total_qa_pairs": len(dataset["qa_pairs"]),
            "total_trajectories": len(dataset["trajectories"]),
            "total_entities": len(dataset["knowledge_graph"]["entities"]),
            "total_relations": len(dataset["knowledge_graph"]["relations"]),
            "question_types": {},
            "obfuscation_types": {},
            "subgraph_types": {}
        }
        
        # 统计问题类型
        for qa in dataset["qa_pairs"]:
            q_type = qa.get("question_type", "unknown")
            stats["question_types"][q_type] = stats["question_types"].get(q_type, 0) + 1
        
        # 统计模糊化类型
        for qa in dataset["qa_pairs"]:
            obf_type = qa.get("obfuscation_type", "none")
            stats["obfuscation_types"][obf_type] = stats["obfuscation_types"].get(obf_type, 0) + 1
        
        return stats


def main():
    """主函数"""
    try:
        # 创建数据集生成器
        generator = WebSailorDatasetGenerator("config.json")
        
        # 生成数据集
        stats = generator.generate_dataset()
        
        # 打印统计信息
        print("\n" + "="*50)
        print("WebSailor数据集生成完成！")
        print("="*50)
        print(f"总QA对数量: {stats['total_qa_pairs']}")
        print(f"总推理轨迹数量: {stats['total_trajectories']}")
        print(f"总实体数量: {stats['total_entities']}")
        print(f"总关系数量: {stats['total_relations']}")
        print("\n问题类型分布:")
        for q_type, count in stats["question_types"].items():
            print(f"  {q_type}: {count}")
        print("\n模糊化类型分布:")
        for obf_type, count in stats["obfuscation_types"].items():
            print(f"  {obf_type}: {count}")
        print("="*50)
        
    except Exception as e:
        logging.error(f"程序执行失败: {e}")
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())