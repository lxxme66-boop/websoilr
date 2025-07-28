"""
数据综合器
整合所有组件，生成完整的数据集
"""

import json
import logging
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random
from tqdm import tqdm
import networkx as nx

from .knowledge_graph_builder import KnowledgeGraphBuilder
from .subgraph_sampler import SubgraphSampler
from .question_generator import QuestionGenerator
from .obfuscation_processor import ObfuscationProcessor
from .trajectory_generator import TrajectoryGenerator

logger = logging.getLogger(__name__)


class DataSynthesizer:
    """
    数据综合器
    WebSailor核心：整合所有组件，生成高质量的垂域数据集
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化各个组件
        logger.info("初始化数据综合器...")
        
        self.kg_builder = KnowledgeGraphBuilder(config)
        self.subgraph_sampler = SubgraphSampler(config)
        self.question_generator = QuestionGenerator(config)
        self.obfuscation_processor = ObfuscationProcessor(config)
        self.trajectory_generator = TrajectoryGenerator(config)
        
        # 数据集配置
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        
        logger.info("数据综合器初始化完成")
        
    def synthesize_dataset(self, input_dir: str, output_dir: str, 
                          num_subgraphs: int = 1000,
                          questions_per_subgraph: int = 5) -> Dict:
        """
        合成完整的数据集
        
        Args:
            input_dir: 输入文本目录
            output_dir: 输出目录
            num_subgraphs: 要采样的子图数量
            questions_per_subgraph: 每个子图生成的问题数
            
        Returns:
            Dict: 数据集统计信息
        """
        logger.info("开始数据集合成流程...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 读取输入文本
        texts = self._load_input_texts(input_dir)
        logger.info(f"加载了{len(texts)}个文本文件")
        
        # 2. 构建知识图谱
        logger.info("构建知识图谱...")
        kg = self.kg_builder.build_from_texts(texts)
        
        # 保存知识图谱
        kg_path = os.path.join(output_dir, "knowledge_graph.json")
        self.kg_builder.save_graph(kg, kg_path)
        
        # 3. 子图采样
        logger.info(f"采样{num_subgraphs}个子图...")
        subgraphs = self.subgraph_sampler.sample_subgraphs(kg, num_subgraphs)
        
        # 保存子图
        subgraphs_path = os.path.join(output_dir, "subgraphs.json")
        self.subgraph_sampler.save_subgraphs(subgraphs, subgraphs_path)
        
        # 4. 问题生成
        logger.info("生成问题...")
        questions = self.question_generator.generate_questions(
            subgraphs, 
            questions_per_subgraph
        )
        
        # 5. 模糊化处理
        logger.info("应用模糊化...")
        obfuscated_questions = self.obfuscation_processor.obfuscate_questions(questions)
        
        # 6. 轨迹生成
        logger.info("生成推理轨迹...")
        qa_with_trajectories = self.trajectory_generator.generate_trajectories(
            obfuscated_questions
        )
        
        # 7. 数据集分割
        logger.info("分割数据集...")
        train_set, val_set, test_set = self._split_dataset(qa_with_trajectories)
        
        # 8. 保存数据集
        self._save_datasets(output_dir, train_set, val_set, test_set)
        
        # 9. 生成统计信息
        stats = self._generate_statistics(
            kg, subgraphs, qa_with_trajectories, 
            train_set, val_set, test_set
        )
        
        # 保存统计信息
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        logger.info("数据集合成完成！")
        logger.info(f"输出目录: {output_dir}")
        
        return stats
        
    def _load_input_texts(self, input_dir: str) -> List[str]:
        """
        加载输入文本
        支持中英文txt文件
        """
        texts = []
        input_path = Path(input_dir)
        
        # 支持的文件扩展名
        extensions = ['.txt', '.text']
        
        for ext in extensions:
            for file_path in input_path.glob(f"*{ext}"):
                try:
                    # 尝试不同的编码
                    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
                    text = None
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                text = f.read().strip()
                                break
                        except UnicodeDecodeError:
                            continue
                            
                    if text:
                        texts.append(text)
                        logger.info(f"成功加载: {file_path.name}")
                    else:
                        logger.warning(f"无法解码文件: {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path.name}: {e}")
                    
        return texts
        
    def _split_dataset(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        将数据集分割为训练集、验证集和测试集
        """
        # 打乱数据
        random.shuffle(qa_pairs)
        
        total = len(qa_pairs)
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        
        train_set = qa_pairs[:train_size]
        val_set = qa_pairs[train_size:train_size + val_size]
        test_set = qa_pairs[train_size + val_size:]
        
        return train_set, val_set, test_set
        
    def _save_datasets(self, output_dir: str, 
                      train_set: List[Dict], 
                      val_set: List[Dict], 
                      test_set: List[Dict]):
        """
        保存数据集到文件
        """
        # 保存为JSON格式
        datasets = {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
        
        for split_name, data in datasets.items():
            # JSON格式
            json_path = os.path.join(output_dir, f"{split_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            # JSONL格式（便于流式处理）
            jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
            logger.info(f"保存{split_name}集: {len(data)}条数据")
            
    def _generate_statistics(self, 
                           kg: nx.DiGraph,
                           subgraphs: List[nx.DiGraph],
                           qa_pairs: List[Dict],
                           train_set: List[Dict],
                           val_set: List[Dict],
                           test_set: List[Dict]) -> Dict:
        """
        生成数据集统计信息
        """
        # 知识图谱统计
        kg_stats = {
            'num_nodes': kg.number_of_nodes(),
            'num_edges': kg.number_of_edges(),
            'density': nx.density(kg),
            'node_types': self._count_node_types(kg),
            'edge_types': self._count_edge_types(kg)
        }
        
        # 子图统计
        subgraph_stats = {
            'total_subgraphs': len(subgraphs),
            'avg_nodes': sum(g.number_of_nodes() for g in subgraphs) / len(subgraphs),
            'avg_edges': sum(g.number_of_edges() for g in subgraphs) / len(subgraphs),
            'topology_distribution': self._count_topologies(subgraphs),
            'complexity_distribution': self._analyze_complexity(subgraphs)
        }
        
        # 问题统计
        question_stats = {
            'total_questions': len(qa_pairs),
            'type_distribution': self._count_question_types(qa_pairs),
            'language_distribution': self._count_languages(qa_pairs),
            'difficulty_distribution': self._analyze_difficulty(qa_pairs),
            'obfuscation_stats': self._analyze_obfuscation(qa_pairs)
        }
        
        # 轨迹统计
        trajectory_stats = {
            'reasoning_type_distribution': self._count_reasoning_types(qa_pairs),
            'avg_trajectory_steps': self._calculate_avg_steps(qa_pairs),
            'coherence_distribution': self._analyze_coherence(qa_pairs)
        }
        
        # 数据集分割统计
        split_stats = {
            'train_size': len(train_set),
            'val_size': len(val_set),
            'test_size': len(test_set),
            'total_size': len(train_set) + len(val_set) + len(test_set)
        }
        
        # 汇总统计
        stats = {
            'dataset_name': 'WebSailor TCL Industrial Domain Dataset',
            'version': self.config.get('version', '1.0.0'),
            'knowledge_graph': kg_stats,
            'subgraphs': subgraph_stats,
            'questions': question_stats,
            'trajectories': trajectory_stats,
            'splits': split_stats,
            'config': {
                'num_subgraphs': len(subgraphs),
                'questions_per_subgraph': len(qa_pairs) // len(subgraphs) if subgraphs else 0,
                'models_used': list(self.config['models'].keys())
            }
        }
        
        return stats
        
    def _count_node_types(self, kg: nx.DiGraph) -> Dict[str, int]:
        """统计节点类型"""
        type_counts = {}
        for node, data in kg.nodes(data=True):
            node_type = data.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
        
    def _count_edge_types(self, kg: nx.DiGraph) -> Dict[str, int]:
        """统计边类型"""
        type_counts = {}
        for u, v, data in kg.edges(data=True):
            edge_type = data.get('type', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
        
    def _count_topologies(self, subgraphs: List[nx.DiGraph]) -> Dict[str, int]:
        """统计拓扑类型分布"""
        topology_counts = {}
        for g in subgraphs:
            topology = g.graph.get('topology', 'unknown')
            topology_counts[topology] = topology_counts.get(topology, 0) + 1
        return topology_counts
        
    def _analyze_complexity(self, subgraphs: List[nx.DiGraph]) -> Dict[str, int]:
        """分析复杂度分布"""
        complexity_bins = {
            'low': 0,
            'medium': 0,
            'high': 0
        }
        
        for g in subgraphs:
            complexity = g.graph.get('complexity', 0.5)
            if complexity < 0.33:
                complexity_bins['low'] += 1
            elif complexity < 0.67:
                complexity_bins['medium'] += 1
            else:
                complexity_bins['high'] += 1
                
        return complexity_bins
        
    def _count_question_types(self, qa_pairs: List[Dict]) -> Dict[str, int]:
        """统计问题类型"""
        type_counts = {}
        for qa in qa_pairs:
            q_type = qa.get('type', 'unknown')
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        return type_counts
        
    def _count_languages(self, qa_pairs: List[Dict]) -> Dict[str, int]:
        """统计语言分布"""
        lang_counts = {}
        for qa in qa_pairs:
            lang = qa.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        return lang_counts
        
    def _analyze_difficulty(self, qa_pairs: List[Dict]) -> Dict[str, int]:
        """分析难度分布"""
        difficulty_bins = {
            'easy': 0,
            'medium': 0,
            'hard': 0
        }
        
        for qa in qa_pairs:
            difficulty = qa.get('difficulty', 0.5)
            if difficulty < 0.33:
                difficulty_bins['easy'] += 1
            elif difficulty < 0.67:
                difficulty_bins['medium'] += 1
            else:
                difficulty_bins['hard'] += 1
                
        return difficulty_bins
        
    def _analyze_obfuscation(self, qa_pairs: List[Dict]) -> Dict:
        """分析模糊化统计"""
        obf_stats = {
            'obfuscated_count': 0,
            'avg_obfuscation_level': 0,
            'obfuscation_types': {}
        }
        
        total_level = 0
        
        for qa in qa_pairs:
            if qa.get('obfuscation_level', 0) > 0:
                obf_stats['obfuscated_count'] += 1
                total_level += qa.get('obfuscation_level', 0)
                
                for obf_type in qa.get('obfuscation_types', []):
                    obf_stats['obfuscation_types'][obf_type] = \
                        obf_stats['obfuscation_types'].get(obf_type, 0) + 1
                        
        if obf_stats['obfuscated_count'] > 0:
            obf_stats['avg_obfuscation_level'] = total_level / obf_stats['obfuscated_count']
            
        return obf_stats
        
    def _count_reasoning_types(self, qa_pairs: List[Dict]) -> Dict[str, int]:
        """统计推理类型"""
        reasoning_counts = {}
        for qa in qa_pairs:
            reasoning_type = qa.get('reasoning_type', 'unknown')
            reasoning_counts[reasoning_type] = reasoning_counts.get(reasoning_type, 0) + 1
        return reasoning_counts
        
    def _calculate_avg_steps(self, qa_pairs: List[Dict]) -> float:
        """计算平均轨迹步数"""
        total_steps = 0
        count = 0
        
        for qa in qa_pairs:
            if 'trajectory' in qa and 'num_steps' in qa['trajectory']:
                total_steps += qa['trajectory']['num_steps']
                count += 1
                
        return total_steps / count if count > 0 else 0
        
    def _analyze_coherence(self, qa_pairs: List[Dict]) -> Dict[str, int]:
        """分析轨迹连贯性"""
        coherence_bins = {
            'low': 0,
            'medium': 0,
            'high': 0
        }
        
        for qa in qa_pairs:
            if 'trajectory' in qa:
                coherence = qa['trajectory'].get('coherence_score', 0.5)
                if coherence < 0.33:
                    coherence_bins['low'] += 1
                elif coherence < 0.67:
                    coherence_bins['medium'] += 1
                else:
                    coherence_bins['high'] += 1
                    
        return coherence_bins