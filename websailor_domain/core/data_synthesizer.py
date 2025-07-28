"""
数据综合器
将所有组件生成的数据综合成最终的数据集
"""

import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import jsonlines
import csv
import random
from datetime import datetime
from collections import defaultdict
import networkx as nx

from .trajectory_generator import ReasoningTrajectory


class DataSynthesizer:
    """
    数据综合器
    负责将各个组件生成的数据整合成最终数据集
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_formats = config.get('output_formats', ['json'])
        self.split_ratios = config.get('split_ratios', {
            'train': 0.7,
            'validation': 0.15,
            'test': 0.15
        })
        self.quality_checks = config.get('quality_checks', {})
        self.metadata_config = config.get('metadata', {})
        
    def synthesize(
        self,
        knowledge_graph: nx.Graph,
        qa_pairs: List[Any],  # ObfuscatedQAPair
        trajectories: List[ReasoningTrajectory],
        domain: str
    ) -> Dict[str, Any]:
        """
        综合所有数据生成最终数据集
        """
        self.logger.info("Starting data synthesis...")
        
        # 1. 组织数据结构
        dataset = self._organize_data_structure(
            knowledge_graph, qa_pairs, trajectories, domain
        )
        
        # 2. 质量检查
        if self.quality_checks:
            dataset = self._perform_quality_checks(dataset)
            
        # 3. 添加元数据
        dataset = self._add_metadata(dataset, domain)
        
        # 4. 数据集划分
        dataset = self._split_dataset(dataset)
        
        # 5. 生成统计信息
        dataset['statistics'] = self._generate_statistics(dataset)
        
        self.logger.info(f"Data synthesis completed. Total samples: {len(dataset['samples'])}")
        
        return dataset
        
    def _organize_data_structure(
        self,
        knowledge_graph: nx.Graph,
        qa_pairs: List[Any],
        trajectories: List[ReasoningTrajectory],
        domain: str
    ) -> Dict[str, Any]:
        """组织数据结构"""
        # 创建轨迹索引
        trajectory_index = {}
        for traj in trajectories:
            key = (traj.qa_pair.question, traj.qa_pair.answer)
            if key not in trajectory_index:
                trajectory_index[key] = []
            trajectory_index[key].append(traj)
            
        # 构建数据样本
        samples = []
        for qa_pair in qa_pairs:
            key = (qa_pair.question, qa_pair.answer)
            trajs = trajectory_index.get(key, [])
            
            sample = {
                'id': f"{domain}_{len(samples):06d}",
                'question': qa_pair.question,
                'answer': qa_pair.answer,
                'question_type': qa_pair.question_type,
                'difficulty': qa_pair.difficulty,
                'ambiguity_score': getattr(qa_pair, 'ambiguity_score', 0.0),
                'original_question': getattr(qa_pair, 'original_question', qa_pair.question),
                'evidence_path': [
                    {
                        'source': source,
                        'relation': relation,
                        'target': target
                    }
                    for source, relation, target in qa_pair.evidence_path
                ],
                'subgraph': {
                    'topology_type': qa_pair.subgraph.topology_type,
                    'num_nodes': qa_pair.subgraph.graph.number_of_nodes(),
                    'num_edges': qa_pair.subgraph.graph.number_of_edges(),
                    'nodes': list(qa_pair.subgraph.graph.nodes()),
                    'edges': [
                        {
                            'source': s,
                            'target': t,
                            'relation': qa_pair.subgraph.graph[s][t].get('relation', 'related_to')
                        }
                        for s, t in qa_pair.subgraph.graph.edges()
                    ]
                },
                'trajectories': [
                    self._serialize_trajectory(traj) for traj in trajs
                ],
                'metadata': qa_pair.metadata
            }
            
            # 添加模糊化元数据
            if hasattr(qa_pair, 'obfuscation_metadata'):
                sample['obfuscation_metadata'] = qa_pair.obfuscation_metadata
                
            samples.append(sample)
            
        # 构建数据集结构
        dataset = {
            'domain': domain,
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'samples': samples,
            'knowledge_graph': self._serialize_knowledge_graph(knowledge_graph)
        }
        
        return dataset
        
    def _serialize_trajectory(self, trajectory: ReasoningTrajectory) -> Dict[str, Any]:
        """序列化推理轨迹"""
        return {
            'reasoning_pattern': trajectory.reasoning_pattern,
            'is_successful': trajectory.is_successful,
            'final_answer': trajectory.final_answer,
            'steps': [
                {
                    'step_type': step.step_type,
                    'description': step.description,
                    'entities_involved': step.entities_involved,
                    'relations_involved': step.relations_involved,
                    'is_correct': step.is_correct
                }
                for step in trajectory.steps
            ],
            'metadata': trajectory.metadata
        }
        
    def _serialize_knowledge_graph(self, kg: nx.Graph) -> Dict[str, Any]:
        """序列化知识图谱"""
        return {
            'num_nodes': kg.number_of_nodes(),
            'num_edges': kg.number_of_edges(),
            'nodes': [
                {
                    'id': node,
                    'type': kg.nodes[node].get('type', 'unknown'),
                    'attributes': {
                        k: v for k, v in kg.nodes[node].items()
                        if k not in ['sentences']  # 排除大文本字段
                    }
                }
                for node in kg.nodes()
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    'relation': kg[source][target].get('relation', 'related_to'),
                    'attributes': {
                        k: v for k, v in kg[source][target].items()
                        if k not in ['sentences', 'relation']
                    }
                }
                for source, target in kg.edges()
            ]
        }
        
    def _perform_quality_checks(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """执行质量检查"""
        samples = dataset['samples']
        filtered_samples = []
        
        for sample in samples:
            # 检查重复
            if self.quality_checks.get('remove_duplicates', True):
                if self._is_duplicate(sample, filtered_samples):
                    continue
                    
            # 验证答案
            if self.quality_checks.get('validate_answers', True):
                if not self._validate_answer(sample):
                    continue
                    
            # 检查一致性
            if self.quality_checks.get('check_consistency', True):
                if not self._check_consistency(sample):
                    continue
                    
            # 检查问题长度
            min_len = self.quality_checks.get('min_question_length', 10)
            max_len = self.quality_checks.get('max_question_length', 200)
            if not (min_len <= len(sample['question']) <= max_len):
                continue
                
            filtered_samples.append(sample)
            
        self.logger.info(f"Quality check: {len(filtered_samples)}/{len(samples)} samples passed")
        dataset['samples'] = filtered_samples
        
        return dataset
        
    def _is_duplicate(self, sample: Dict, existing_samples: List[Dict]) -> bool:
        """检查是否重复"""
        for existing in existing_samples:
            if (sample['question'] == existing['question'] and 
                sample['answer'] == existing['answer']):
                return True
        return False
        
    def _validate_answer(self, sample: Dict) -> bool:
        """验证答案有效性"""
        answer = sample['answer']
        
        # 答案不能为空
        if not answer or answer.strip() == '':
            return False
            
        # 答案应该在子图中
        subgraph_nodes = set(sample['subgraph']['nodes'])
        
        # 对于某些问题类型，答案可能是数字或列表
        if sample['question_type'] in ['aggregation', 'count']:
            return True  # 聚合类答案不需要在子图中
            
        # 检查答案是否在子图节点中
        answer_entities = answer.split(', ')
        for entity in answer_entities:
            if entity in subgraph_nodes:
                return True
                
        return False
        
    def _check_consistency(self, sample: Dict) -> bool:
        """检查数据一致性"""
        # 检查证据路径是否与答案一致
        if sample['evidence_path']:
            last_hop = sample['evidence_path'][-1]
            
            # 对于单跳和多跳问题，最后一跳的目标应该是答案
            if sample['question_type'] in ['single_hop', 'multi_hop']:
                if last_hop['target'] != sample['answer']:
                    return False
                    
        # 检查轨迹一致性
        for traj in sample.get('trajectories', []):
            if traj['is_successful'] and traj['final_answer'] != sample['answer']:
                return False
                
        return True
        
    def _add_metadata(self, dataset: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """添加元数据"""
        # 为每个样本添加元数据
        for i, sample in enumerate(dataset['samples']):
            if self.metadata_config.get('include_difficulty_score', True):
                # 已经有难度分数
                pass
                
            if self.metadata_config.get('include_reasoning_type', True):
                # 从轨迹中提取推理类型
                if sample['trajectories']:
                    sample['reasoning_types'] = list(set(
                        traj['reasoning_pattern'] 
                        for traj in sample['trajectories']
                    ))
                    
            if self.metadata_config.get('include_timestamp', True):
                sample['created_at'] = datetime.now().isoformat()
                
            # 添加领域标签
            sample['domain'] = domain
            
        return dataset
        
    def _split_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """划分数据集"""
        samples = dataset['samples']
        random.shuffle(samples)  # 随机打乱
        
        total = len(samples)
        train_size = int(total * self.split_ratios['train'])
        val_size = int(total * self.split_ratios['validation'])
        
        dataset['splits'] = {
            'train': samples[:train_size],
            'validation': samples[train_size:train_size + val_size],
            'test': samples[train_size + val_size:]
        }
        
        # 为每个样本添加split标签
        for split_name, split_samples in dataset['splits'].items():
            for sample in split_samples:
                sample['split'] = split_name
                
        return dataset
        
    def generate_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """生成数据集统计信息"""
        stats = {
            'total_samples': len(dataset['samples']),
            'domain': dataset['domain'],
            'created_at': dataset['created_at'],
            'knowledge_graph': {
                'num_nodes': dataset['knowledge_graph']['num_nodes'],
                'num_edges': dataset['knowledge_graph']['num_edges']
            }
        }
        
        # 按问题类型统计
        question_type_counts = defaultdict(int)
        for sample in dataset['samples']:
            question_type_counts[sample['question_type']] += 1
        stats['question_types'] = dict(question_type_counts)
        
        # 按拓扑类型统计
        topology_counts = defaultdict(int)
        for sample in dataset['samples']:
            topology_counts[sample['subgraph']['topology_type']] += 1
        stats['topology_types'] = dict(topology_counts)
        
        # 难度分布
        difficulties = [sample['difficulty'] for sample in dataset['samples']]
        stats['difficulty'] = {
            'min': min(difficulties),
            'max': max(difficulties),
            'mean': sum(difficulties) / len(difficulties)
        }
        
        # 模糊度分布
        ambiguities = [sample['ambiguity_score'] for sample in dataset['samples']]
        stats['ambiguity'] = {
            'min': min(ambiguities),
            'max': max(ambiguities),
            'mean': sum(ambiguities) / len(ambiguities)
        }
        
        # 数据集划分统计
        if 'splits' in dataset:
            stats['splits'] = {
                split: len(samples) 
                for split, samples in dataset['splits'].items()
            }
            
        # 推理轨迹统计
        total_trajectories = sum(
            len(sample.get('trajectories', [])) 
            for sample in dataset['samples']
        )
        successful_trajectories = sum(
            sum(1 for traj in sample.get('trajectories', []) if traj['is_successful'])
            for sample in dataset['samples']
        )
        
        stats['trajectories'] = {
            'total': total_trajectories,
            'successful': successful_trajectories,
            'error_rate': 1 - (successful_trajectories / total_trajectories) if total_trajectories > 0 else 0
        }
        
        return stats
        
    def save_dataset(self, dataset: Dict[str, Any], output_path: Path):
        """保存数据集到文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据配置的格式保存
        if 'json' in self.output_formats:
            self._save_json(dataset, output_path)
            
        if 'jsonl' in self.output_formats:
            self._save_jsonl(dataset, output_path.with_suffix('.jsonl'))
            
        if 'csv' in self.output_formats:
            self._save_csv(dataset, output_path.with_suffix('.csv'))
            
        self.logger.info(f"Dataset saved to {output_path}")
        
    def _save_json(self, dataset: Dict[str, Any], path: Path):
        """保存为JSON格式"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
    def _save_jsonl(self, dataset: Dict[str, Any], path: Path):
        """保存为JSONL格式"""
        with jsonlines.open(path, 'w') as writer:
            # 写入元数据
            writer.write({
                'type': 'metadata',
                'domain': dataset['domain'],
                'version': dataset['version'],
                'created_at': dataset['created_at']
            })
            
            # 写入样本
            for sample in dataset['samples']:
                writer.write({
                    'type': 'sample',
                    **sample
                })
                
    def _save_csv(self, dataset: Dict[str, Any], path: Path):
        """保存为CSV格式（简化版）"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id', 'question', 'answer', 'question_type',
                'difficulty', 'ambiguity_score', 'split'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for sample in dataset['samples']:
                row = {
                    field: sample.get(field, '') 
                    for field in fieldnames
                }
                writer.writerow(row)