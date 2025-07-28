#!/usr/bin/env python3
"""
WebSailor Domain-Specific Dataset Construction System - Parallel Version
使用多模型并行加载和处理的增强版本

主要优化：
1. 并行模型加载 - 同时加载多个模型，减少启动时间
2. 批处理优化 - 批量处理文本，提高GPU利用率
3. 流水线处理 - 不同阶段并行执行
4. 异步IO - 文件读写异步化
"""

import argparse
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, Any, List
import torch
import multiprocessing as mp

from core.model_manager import ModelManager, AsyncModelManager, ModelConfig
from core.enhanced_knowledge_graph_builder import EnhancedKnowledgeGraphBuilder
from core.subgraph_sampler import SubgraphSampler
from core.question_generator import QuestionGenerator
from core.obfuscation_processor import ObfuscationProcessor
from core.trajectory_generator import TrajectoryGenerator
from core.data_synthesizer import DataSynthesizer


def setup_logging(level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_model_configs(config: Dict[str, Any], use_gpu: bool = True) -> List[ModelConfig]:
    """创建模型配置"""
    model_configs = []
    
    # 检测可用GPU
    gpu_count = torch.cuda.device_count() if use_gpu else 0
    
    # NER模型配置
    model_configs.append(ModelConfig(
        name="ner_bert",
        model_type="ner",
        model_path=config.get('models', {}).get('ner_model', 'bert-base-chinese'),
        device=f"cuda:{0 % gpu_count}" if gpu_count > 0 else "cpu",
        batch_size=config.get('batch_processing', {}).get('ner_batch_size', 32),
        max_length=512
    ))
    
    # 关系抽取模型配置
    model_configs.append(ModelConfig(
        name="re_bert",
        model_type="relation_extraction",
        model_path=config.get('models', {}).get('re_model', 'bert-base-chinese'),
        device=f"cuda:{1 % gpu_count}" if gpu_count > 1 else f"cuda:{0 % gpu_count}" if gpu_count > 0 else "cpu",
        batch_size=config.get('batch_processing', {}).get('re_batch_size', 16),
        max_length=512
    ))
    
    # 问题生成模型配置
    if config.get('use_llm_for_generation', False):
        model_configs.append(ModelConfig(
            name="qg_t5",
            model_type="question_generation",
            model_path=config.get('models', {}).get('qg_model', 't5-base'),
            device=f"cuda:{2 % gpu_count}" if gpu_count > 2 else f"cuda:{0 % gpu_count}" if gpu_count > 0 else "cpu",
            batch_size=config.get('batch_processing', {}).get('qg_batch_size', 8),
            max_length=512
        ))
    
    return model_configs


class ParallelDatasetBuilder:
    """并行数据集构建器"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model_manager = None
        self.use_async = config.get('use_async_processing', True)
        
    async def build_async(self, input_dir: Path, output_dir: Path, domain: str):
        """异步构建数据集"""
        start_time = time.time()
        
        try:
            # 1. 并行加载模型
            self.logger.info("Step 1: Loading models in parallel...")
            model_configs = create_model_configs(self.config)
            
            if self.use_async:
                self.model_manager = AsyncModelManager(
                    model_configs, 
                    max_workers=self.config.get('parallel', {}).get('max_workers', 4)
                )
            else:
                self.model_manager = ModelManager(
                    model_configs,
                    max_workers=self.config.get('parallel', {}).get('max_workers', 4)
                )
                
            # 并行加载所有模型
            await asyncio.get_event_loop().run_in_executor(
                None, self.model_manager.load_models_parallel
            )
            
            if self.use_async and isinstance(self.model_manager, AsyncModelManager):
                self.model_manager.start_async_workers()
            
            # 2. 并行构建知识图谱
            self.logger.info("Step 2: Building knowledge graph with parallel processing...")
            kg_builder = EnhancedKnowledgeGraphBuilder(
                self.config['knowledge_graph'],
                model_manager=self.model_manager
            )
            
            knowledge_graph = await asyncio.get_event_loop().run_in_executor(
                None, kg_builder.build_from_texts_parallel, input_dir
            )
            
            # 保存知识图谱
            kg_path = output_dir / 'knowledge_graph.json'
            self._save_graph(knowledge_graph, kg_path)
            
            # 3-6. 使用流水线并行处理剩余步骤
            self.logger.info("Step 3-6: Running pipeline processing...")
            
            # 创建处理任务
            tasks = []
            
            # 子图采样任务
            sampler = SubgraphSampler(self.config['subgraph_sampling'])
            tasks.append(asyncio.create_task(
                self._async_sample_subgraphs(sampler, knowledge_graph)
            ))
            
            # 等待子图采样完成
            subgraphs = await tasks[0]
            
            # 并行处理问题生成和模糊化
            question_gen = QuestionGenerator(self.config['question_generation'])
            obfuscator = ObfuscationProcessor(self.config['obfuscation'])
            
            # 分批处理子图
            batch_size = self.config.get('parallel', {}).get('subgraph_batch_size', 100)
            all_qa_pairs = []
            
            for i in range(0, len(subgraphs), batch_size):
                batch_subgraphs = subgraphs[i:i + batch_size]
                
                # 并行生成问题
                qa_batch = await self._async_generate_questions(
                    question_gen, batch_subgraphs
                )
                
                # 并行模糊化处理
                obfuscated_batch = await self._async_obfuscate(
                    obfuscator, qa_batch, knowledge_graph
                )
                
                all_qa_pairs.extend(obfuscated_batch)
            
            # 5. 生成推理轨迹
            self.logger.info("Step 5: Generating reasoning trajectories...")
            trajectory_gen = TrajectoryGenerator(self.config['trajectory_generation'])
            trajectories = await self._async_generate_trajectories(
                trajectory_gen, all_qa_pairs, knowledge_graph
            )
            
            # 6. 数据综合
            self.logger.info("Step 6: Synthesizing final dataset...")
            synthesizer = DataSynthesizer(self.config['data_synthesis'])
            dataset = synthesizer.synthesize(
                knowledge_graph=knowledge_graph,
                qa_pairs=all_qa_pairs,
                trajectories=trajectories,
                domain=domain
            )
            
            # 保存数据集
            self._save_dataset(dataset, output_dir)
            
            # 打印统计信息
            elapsed_time = time.time() - start_time
            self._print_statistics(dataset, kg_builder.get_statistics(), elapsed_time)
            
        finally:
            # 清理资源
            if self.model_manager:
                self.model_manager.shutdown()
                
    async def _async_sample_subgraphs(self, sampler, knowledge_graph):
        """异步采样子图"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, sampler.sample_diverse_subgraphs, knowledge_graph
        )
        
    async def _async_generate_questions(self, generator, subgraphs):
        """异步生成问题"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, generator.generate_from_subgraphs, subgraphs
        )
        
    async def _async_obfuscate(self, obfuscator, qa_pairs, knowledge_graph):
        """异步模糊化处理"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, obfuscator.process_qa_pairs, qa_pairs, knowledge_graph
        )
        
    async def _async_generate_trajectories(self, generator, qa_pairs, knowledge_graph):
        """异步生成轨迹"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, generator.generate_trajectories, qa_pairs, knowledge_graph
        )
        
    def _save_graph(self, graph, path):
        """保存知识图谱"""
        import networkx as nx
        
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    **graph.nodes[node]
                }
                for node in graph.nodes()
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    **graph[u][v]
                }
                for u, v in graph.edges()
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
    def _save_dataset(self, dataset, output_dir):
        """保存数据集"""
        # 保存完整数据集
        dataset_path = output_dir / 'dataset.json'
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
        # 保存分割后的数据集
        for split_name, split_data in dataset.get('splits', {}).items():
            split_path = output_dir / f'{split_name}.json'
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
                
    def _print_statistics(self, dataset, kg_stats, elapsed_time):
        """打印统计信息"""
        stats = dataset.get('statistics', {})
        
        print("\n" + "="*50)
        print("Dataset Construction Statistics")
        print("="*50)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"\nKnowledge Graph:")
        print(f"  - Processed batches: {kg_stats.get('processed_batches', 0)}")
        print(f"  - Total entities: {kg_stats.get('total_entities', 0)}")
        print(f"  - Total relations: {kg_stats.get('total_relations', 0)}")
        print(f"  - Inferred relations: {kg_stats.get('inferred_relations', 0)}")
        
        print(f"\nDataset:")
        print(f"  - Total samples: {stats.get('total_samples', 0)}")
        print(f"  - Question types: {stats.get('question_types', {})}")
        print(f"  - Average difficulty: {stats.get('difficulty', {}).get('mean', 0):.2f}")
        print(f"  - Average ambiguity: {stats.get('ambiguity', {}).get('mean', 0):.2f}")
        print("="*50 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='WebSailor Domain Dataset Construction - Parallel Version'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('input_texts'),
        help='Input directory containing domain texts'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output_dataset'),
        help='Output directory for generated dataset'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.json'),
        help='Configuration file path'
    )
    parser.add_argument(
        '--domain',
        type=str,
        default='TCL_Industry',
        help='Domain name for the dataset'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for model inference'
    )
    parser.add_argument(
        '--async-mode',
        action='store_true',
        help='Use async processing mode'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(logging.DEBUG if args.debug else logging.INFO)
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置
    config['use_async_processing'] = args.async_mode
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建构建器
    builder = ParallelDatasetBuilder(config, logger)
    
    # 运行构建过程
    try:
        if args.async_mode:
            # 异步模式
            asyncio.run(builder.build_async(args.input_dir, args.output_dir, args.domain))
        else:
            # 同步模式（使用线程池）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                builder.build_async(args.input_dir, args.output_dir, args.domain)
            )
    except KeyboardInterrupt:
        logger.info("Construction interrupted by user")
    except Exception as e:
        logger.error(f"Construction failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()