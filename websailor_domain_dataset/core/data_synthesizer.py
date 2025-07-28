"""
数据综合器
整合所有组件的输出，生成最终的TCL工业垂域数据集
"""

import logging
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class DataSynthesizer:
    """
    数据综合器 - 整合所有WebSailor组件的输出
    
    功能：
    1. 整合子图采样、问题生成、模糊化处理、推理轨迹的结果
    2. 生成统一格式的数据集
    3. 进行数据质量控制和统计分析
    4. 输出多种格式的数据文件
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_formats = config['output_formats']
        self.quality_thresholds = config['quality_thresholds']
        self.include_statistics = config['include_statistics']
        
        # 数据集元数据
        self.dataset_metadata = {
            'name': 'TCL工业垂域知识推理数据集',
            'version': '1.0',
            'description': '基于WebSailor思想构造的TCL工业领域知识推理数据集',
            'creation_date': datetime.now().isoformat(),
            'methodology': 'WebSailor: 子图采样 + 问题生成 + 模糊化处理',
            'domain': 'TCL工业制造',
            'language': 'Chinese'
        }
    
    def synthesize_dataset(self, subgraphs: List[Dict[str, Any]], 
                          all_questions: List[Dict[str, Any]],
                          all_trajectories: List[Dict[str, Any]],
                          knowledge_graph: nx.Graph) -> Dict[str, Any]:
        """
        综合生成最终数据集
        
        Args:
            subgraphs: 子图数据列表
            all_questions: 所有问题数据
            all_trajectories: 所有推理轨迹
            knowledge_graph: 完整知识图谱
            
        Returns:
            Dict: 综合数据集
        """
        logger.info("开始综合生成数据集")
        
        # 1. 数据质量过滤
        filtered_questions = self._filter_by_quality(all_questions)
        filtered_trajectories = self._filter_trajectories(all_trajectories)
        
        # 2. 数据配对和整合
        integrated_data = self._integrate_data(
            subgraphs, filtered_questions, filtered_trajectories
        )
        
        # 3. 生成数据集结构
        dataset = {
            'metadata': self.dataset_metadata,
            'statistics': self._generate_statistics(
                integrated_data, subgraphs, knowledge_graph
            ),
            'data': integrated_data,
            'knowledge_graph_info': self._extract_kg_info(knowledge_graph),
            'quality_metrics': self._calculate_quality_metrics(integrated_data)
        }
        
        # 4. 数据验证
        validation_results = self._validate_dataset(dataset)
        dataset['validation'] = validation_results
        
        logger.info(f"数据集综合完成，包含 {len(integrated_data)} 条数据")
        return dataset
    
    def _filter_by_quality(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于质量阈值过滤问题"""
        filtered = []
        
        for question in questions:
            # 检查基本质量指标
            if self._meets_quality_threshold(question):
                filtered.append(question)
        
        logger.info(f"质量过滤：{len(questions)} -> {len(filtered)} 个问题")
        return filtered
    
    def _meets_quality_threshold(self, question: Dict[str, Any]) -> bool:
        """检查问题是否满足质量阈值"""
        # 检查问题长度
        question_text = question.get('question', '')
        if len(question_text) < self.quality_thresholds.get('min_question_length', 5):
            return False
        
        # 检查答案存在性
        answer = question.get('answer', '')
        if not answer or len(answer) < self.quality_thresholds.get('min_answer_length', 1):
            return False
        
        # 检查难度等级
        difficulty = question.get('difficulty', 1)
        if difficulty < self.quality_thresholds.get('min_difficulty', 1):
            return False
        
        # 检查涉及实体数量
        involved_entities = question.get('involved_entities', [])
        if len(involved_entities) < self.quality_thresholds.get('min_entities', 1):
            return False
        
        return True
    
    def _filter_trajectories(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤推理轨迹"""
        filtered = []
        
        for trajectory in trajectories:
            # 检查轨迹完整性
            steps = trajectory.get('steps', [])
            if len(steps) >= self.quality_thresholds.get('min_trajectory_steps', 2):
                # 检查置信度
                confidence = trajectory.get('confidence_score', 0)
                if confidence >= self.quality_thresholds.get('min_confidence', 0.3):
                    filtered.append(trajectory)
        
        logger.info(f"轨迹过滤：{len(trajectories)} -> {len(filtered)} 条轨迹")
        return filtered
    
    def _integrate_data(self, subgraphs: List[Dict[str, Any]], 
                       questions: List[Dict[str, Any]],
                       trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """整合所有数据组件"""
        integrated_data = []
        
        # 创建轨迹索引
        trajectory_index = {
            traj.get('question_id', ''): traj for traj in trajectories
        }
        
        # 按子图分组问题
        subgraph_questions = defaultdict(list)
        for question in questions:
            # 根据问题ID推断所属子图（简化处理）
            subgraph_id = question.get('question_id', '').split('_')[0]
            subgraph_questions[subgraph_id].append(question)
        
        # 整合每个数据点
        data_id = 0
        for subgraph_data in subgraphs:
            subgraph_strategy = subgraph_data.get('strategy', 'unknown')
            related_questions = subgraph_questions.get(subgraph_strategy, [])
            
            for question in related_questions:
                question_id = question.get('question_id', '')
                trajectory = trajectory_index.get(question_id)
                
                # 创建综合数据点
                data_point = {
                    'id': data_id,
                    'subgraph_info': {
                        'strategy': subgraph_data.get('strategy'),
                        'size': subgraph_data.get('size'),
                        'quality_score': subgraph_data.get('quality_score'),
                        'scenario_features': subgraph_data.get('scenario_features'),
                        'metadata': subgraph_data.get('metadata')
                    },
                    'question': question.get('question'),
                    'answer': question.get('answer'),
                    'question_metadata': {
                        'type': question.get('type'),
                        'subtype': question.get('subtype'),
                        'difficulty': question.get('difficulty'),
                        'required_hops': question.get('required_hops'),
                        'involved_entities': question.get('involved_entities'),
                        'source_nodes': question.get('source_nodes'),
                        'source_edges': question.get('source_edges'),
                        'reasoning_steps': question.get('reasoning_steps')
                    },
                    'obfuscation_info': question.get('obfuscation_info', {}),
                    'is_obfuscated': question.get('is_obfuscated', False),
                    'original_question': question.get('original_question'),
                    'trajectory': trajectory if trajectory else None,
                    'websailor_features': {
                        'subgraph_sampling': True,
                        'question_generation': True,
                        'obfuscation_applied': question.get('is_obfuscated', False),
                        'trajectory_included': trajectory is not None,
                        'uncertainty_level': question.get('obfuscation_info', {}).get('obfuscation_level', 0)
                    }
                }
                
                integrated_data.append(data_point)
                data_id += 1
        
        return integrated_data
    
    def _generate_statistics(self, integrated_data: List[Dict[str, Any]], 
                           subgraphs: List[Dict[str, Any]],
                           knowledge_graph: nx.Graph) -> Dict[str, Any]:
        """生成数据集统计信息"""
        stats = {
            'basic_statistics': {
                'total_samples': len(integrated_data),
                'total_subgraphs': len(subgraphs),
                'knowledge_graph_nodes': len(knowledge_graph.nodes()),
                'knowledge_graph_edges': len(knowledge_graph.edges())
            },
            'question_type_distribution': {},
            'difficulty_distribution': {},
            'obfuscation_statistics': {},
            'subgraph_strategy_distribution': {},
            'trajectory_statistics': {},
            'websailor_coverage': {}
        }
        
        # 问题类型分布
        question_types = [item['question_metadata']['type'] for item in integrated_data]
        stats['question_type_distribution'] = dict(Counter(question_types))
        
        # 难度分布
        difficulties = [item['question_metadata']['difficulty'] for item in integrated_data]
        stats['difficulty_distribution'] = dict(Counter(difficulties))
        
        # 模糊化统计
        obfuscated_count = sum(1 for item in integrated_data if item['is_obfuscated'])
        stats['obfuscation_statistics'] = {
            'total_obfuscated': obfuscated_count,
            'obfuscation_ratio': obfuscated_count / len(integrated_data) if integrated_data else 0,
            'avg_obfuscation_level': sum(
                item.get('obfuscation_info', {}).get('obfuscation_level', 0) 
                for item in integrated_data if item['is_obfuscated']
            ) / max(obfuscated_count, 1)
        }
        
        # 子图策略分布
        strategies = [item['subgraph_info']['strategy'] for item in integrated_data]
        stats['subgraph_strategy_distribution'] = dict(Counter(strategies))
        
        # 轨迹统计
        trajectory_count = sum(1 for item in integrated_data if item['trajectory'])
        if trajectory_count > 0:
            avg_steps = sum(
                len(item['trajectory'].get('steps', [])) 
                for item in integrated_data if item['trajectory']
            ) / trajectory_count
            
            avg_complexity = sum(
                item['trajectory'].get('reasoning_complexity', 0) 
                for item in integrated_data if item['trajectory']
            ) / trajectory_count
            
            stats['trajectory_statistics'] = {
                'total_trajectories': trajectory_count,
                'coverage_ratio': trajectory_count / len(integrated_data),
                'avg_steps_per_trajectory': avg_steps,
                'avg_complexity': avg_complexity
            }
        
        # WebSailor覆盖率
        stats['websailor_coverage'] = {
            'subgraph_sampling_coverage': 1.0,  # 所有数据都来自子图采样
            'question_generation_coverage': 1.0,  # 所有问题都是生成的
            'obfuscation_coverage': obfuscated_count / len(integrated_data) if integrated_data else 0,
            'trajectory_coverage': trajectory_count / len(integrated_data) if integrated_data else 0
        }
        
        return stats
    
    def _extract_kg_info(self, knowledge_graph: nx.Graph) -> Dict[str, Any]:
        """提取知识图谱信息"""
        # 节点类型统计
        node_types = defaultdict(int)
        for node in knowledge_graph.nodes():
            node_type = knowledge_graph.nodes[node].get('type', 'unknown')
            node_types[node_type] += 1
        
        # 关系类型统计
        relation_types = defaultdict(int)
        for _, _, data in knowledge_graph.edges(data=True):
            relation = data.get('relation', 'unknown')
            relation_types[relation] += 1
        
        # 图结构特征
        density = nx.density(knowledge_graph)
        avg_degree = sum(dict(knowledge_graph.degree()).values()) / len(knowledge_graph.nodes()) if knowledge_graph.nodes() else 0
        
        return {
            'node_type_distribution': dict(node_types),
            'relation_type_distribution': dict(relation_types),
            'graph_density': density,
            'average_degree': avg_degree,
            'is_connected': nx.is_connected(knowledge_graph),
            'number_of_components': nx.number_connected_components(knowledge_graph)
        }
    
    def _calculate_quality_metrics(self, integrated_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算数据集质量指标"""
        if not integrated_data:
            return {}
        
        # 问题质量指标
        avg_question_length = sum(len(item['question']) for item in integrated_data) / len(integrated_data)
        avg_answer_length = sum(len(item['answer']) for item in integrated_data) / len(integrated_data)
        avg_difficulty = sum(item['question_metadata']['difficulty'] for item in integrated_data) / len(integrated_data)
        
        # 复杂度指标
        avg_entities_per_question = sum(
            len(item['question_metadata']['involved_entities']) 
            for item in integrated_data
        ) / len(integrated_data)
        
        avg_hops = sum(
            item['question_metadata']['required_hops'] 
            for item in integrated_data
        ) / len(integrated_data)
        
        # WebSailor特征覆盖
        websailor_completeness = sum(
            1 for item in integrated_data 
            if all([
                item['websailor_features']['subgraph_sampling'],
                item['websailor_features']['question_generation'],
                item['websailor_features']['obfuscation_applied'] or not self.config.get('require_obfuscation', False),
                item['websailor_features']['trajectory_included'] or not self.config.get('require_trajectory', False)
            ])
        ) / len(integrated_data)
        
        return {
            'avg_question_length': avg_question_length,
            'avg_answer_length': avg_answer_length,
            'avg_difficulty': avg_difficulty,
            'avg_entities_per_question': avg_entities_per_question,
            'avg_reasoning_hops': avg_hops,
            'websailor_completeness': websailor_completeness,
            'data_diversity': len(set(item['question_metadata']['type'] for item in integrated_data)) / 6  # 假设6种问题类型
        }
    
    def _validate_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """验证数据集完整性和一致性"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'validation_checks': {}
        }
        
        data = dataset.get('data', [])
        
        # 检查1：数据完整性
        for i, item in enumerate(data):
            if not item.get('question') or not item.get('answer'):
                validation_results['errors'].append(f"数据点 {i} 缺少问题或答案")
                validation_results['is_valid'] = False
        
        # 检查2：WebSailor特征覆盖
        websailor_coverage = dataset.get('statistics', {}).get('websailor_coverage', {})
        if websailor_coverage.get('subgraph_sampling_coverage', 0) < 1.0:
            validation_results['warnings'].append("子图采样覆盖率不足100%")
        
        # 检查3：数据质量分布
        quality_metrics = dataset.get('quality_metrics', {})
        if quality_metrics.get('avg_difficulty', 0) < 2.0:
            validation_results['warnings'].append("平均难度偏低，可能影响数据集挑战性")
        
        # 检查4：类型分布均衡性
        type_dist = dataset.get('statistics', {}).get('question_type_distribution', {})
        if type_dist:
            max_count = max(type_dist.values())
            min_count = min(type_dist.values())
            if max_count / min_count > 3:  # 最大与最小差异超过3倍
                validation_results['warnings'].append("问题类型分布不均衡")
        
        validation_results['validation_checks'] = {
            'data_completeness': len([item for item in data if item.get('question') and item.get('answer')]) / len(data) if data else 0,
            'websailor_coverage': websailor_coverage.get('subgraph_sampling_coverage', 0),
            'type_balance_ratio': min(type_dist.values()) / max(type_dist.values()) if type_dist else 0
        }
        
        return validation_results
    
    def export_dataset(self, dataset: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
        """导出数据集到多种格式"""
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = {}
        
        # 导出完整JSON格式
        if 'json' in self.output_formats:
            json_path = output_dir / 'tcl_dataset_complete.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2, default=str)
            exported_files['complete_json'] = str(json_path)
        
        # 导出QA对格式
        if 'qa_pairs' in self.output_formats:
            qa_pairs = []
            for item in dataset['data']:
                qa_pair = {
                    'id': item['id'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'type': item['question_metadata']['type'],
                    'difficulty': item['question_metadata']['difficulty'],
                    'is_obfuscated': item['is_obfuscated']
                }
                if item['is_obfuscated']:
                    qa_pair['original_question'] = item.get('original_question', '')
                qa_pairs.append(qa_pair)
            
            qa_path = output_dir / 'qa_pairs.json'
            with open(qa_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            exported_files['qa_pairs'] = str(qa_path)
        
        # 导出推理轨迹格式
        if 'trajectories' in self.output_formats:
            trajectories = []
            for item in dataset['data']:
                if item['trajectory']:
                    traj_data = {
                        'id': item['id'],
                        'question': item['question'],
                        'answer': item['answer'],
                        'trajectory': item['trajectory']
                    }
                    trajectories.append(traj_data)
            
            traj_path = output_dir / 'trajectories.json'
            with open(traj_path, 'w', encoding='utf-8') as f:
                json.dump(trajectories, f, ensure_ascii=False, indent=2, default=str)
            exported_files['trajectories'] = str(traj_path)
        
        # 导出统计信息
        if 'statistics' in self.output_formats:
            stats_path = output_dir / 'statistics.json'
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(dataset['statistics'], f, ensure_ascii=False, indent=2, default=str)
            exported_files['statistics'] = str(stats_path)
        
        # 导出CSV格式（简化版）
        if 'csv' in self.output_formats:
            csv_data = []
            for item in dataset['data']:
                csv_row = {
                    'id': item['id'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'type': item['question_metadata']['type'],
                    'difficulty': item['question_metadata']['difficulty'],
                    'is_obfuscated': item['is_obfuscated'],
                    'subgraph_strategy': item['subgraph_info']['strategy'],
                    'required_hops': item['question_metadata']['required_hops']
                }
                csv_data.append(csv_row)
            
            df = pd.DataFrame(csv_data)
            csv_path = output_dir / 'dataset.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            exported_files['csv'] = str(csv_path)
        
        logger.info(f"数据集导出完成，文件保存在: {output_dir}")
        return exported_files