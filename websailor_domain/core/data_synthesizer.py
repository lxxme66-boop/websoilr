"""
数据综合器
整合所有组件生成的数据，创建最终的数据集
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pandas as pd
import jsonlines
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataSynthesizer:
    """数据综合器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.synth_config = config.get('dataset_synthesis', {})
        
        # 数据集分割比例
        self.train_ratio = self.synth_config.get('train_ratio', 0.8)
        self.val_ratio = self.synth_config.get('val_ratio', 0.1)
        self.test_ratio = self.synth_config.get('test_ratio', 0.1)
        
        # 输出格式
        self.output_formats = self.synth_config.get('output_formats', ['json'])
        
        # 质量检查配置
        self.quality_checks = self.synth_config.get('quality_checks', {})
        
    def synthesize_dataset(self, kg_path: Path, qa_path: Path, 
                         output_path: Path) -> Dict:
        """综合数据集"""
        logger.info("开始综合数据集...")
        
        # 加载数据
        knowledge_graph = self._load_json(kg_path)
        qa_pairs = self._load_json(qa_path)
        
        # 数据预处理
        processed_data = self._preprocess_data(qa_pairs, knowledge_graph)
        
        # 数据分割
        train_data, val_data, test_data = self._split_dataset(processed_data)
        
        # 数据增强（可选）
        train_data = self._augment_data(train_data)
        
        # 保存数据集
        dataset = {
            'train': train_data,
            'validation': val_data,
            'test': test_data,
            'metadata': self._create_metadata(train_data, val_data, test_data)
        }
        
        # 导出不同格式
        self._export_dataset(dataset, output_path)
        
        logger.info(f"数据集综合完成: {len(train_data)} 训练, "
                   f"{len(val_data)} 验证, {len(test_data)} 测试")
        
        return dataset
    
    def _load_json(self, path: Path) -> Dict:
        """加载JSON文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _preprocess_data(self, qa_pairs: List[Dict], 
                        knowledge_graph: Dict) -> List[Dict]:
        """预处理数据"""
        processed = []
        
        # 创建节点和边的索引
        node_index = {node['id']: node for node in knowledge_graph['nodes']}
        edge_index = defaultdict(list)
        for edge in knowledge_graph['edges']:
            edge_index[(edge['source'], edge['target'])].append(edge)
        
        for qa in qa_pairs:
            # 验证质量
            if not self._validate_qa_quality(qa):
                continue
            
            # 增强QA对的信息
            enhanced_qa = self._enhance_qa_pair(qa, node_index, edge_index)
            
            # 格式化数据
            formatted_qa = self._format_qa_pair(enhanced_qa)
            
            processed.append(formatted_qa)
        
        logger.info(f"预处理完成: {len(processed)}/{len(qa_pairs)} 个QA对通过质量检查")
        
        return processed
    
    def _validate_qa_quality(self, qa: Dict) -> bool:
        """验证QA对质量"""
        # 问题长度检查
        q_len = len(qa['question'])
        min_len = self.quality_checks.get('min_question_length', 10)
        max_len = self.quality_checks.get('max_question_length', 200)
        
        if not (min_len <= q_len <= max_len):
            return False
        
        # 答案验证
        if self.quality_checks.get('answer_validation', True):
            if not qa.get('answer') or len(qa['answer']) < 5:
                return False
        
        # 轨迹验证
        if self.quality_checks.get('trajectory_validation', True):
            if 'trajectory' in qa and not qa['trajectory'].get('steps'):
                return False
        
        return True
    
    def _enhance_qa_pair(self, qa: Dict, node_index: Dict, 
                        edge_index: Dict) -> Dict:
        """增强QA对信息"""
        enhanced = qa.copy()
        
        # 添加实体信息
        if 'evidence' in qa:
            entities = []
            for node in qa['evidence'].get('nodes', []):
                if node['id'] in node_index:
                    entity_info = node_index[node['id']].copy()
                    entities.append(entity_info)
            enhanced['entities'] = entities
        
        # 添加关系路径
        if 'evidence' in qa and 'path' in qa['evidence']:
            path = qa['evidence']['path']
            path_info = []
            for i in range(len(path) - 1):
                edge_key = (path[i], path[i+1])
                if edge_key in edge_index:
                    path_info.append({
                        'from': path[i],
                        'to': path[i+1],
                        'relations': [e['relation'] for e in edge_index[edge_key]]
                    })
            enhanced['path_info'] = path_info
        
        # 添加难度评分
        enhanced['difficulty'] = self._calculate_difficulty(qa)
        
        return enhanced
    
    def _calculate_difficulty(self, qa: Dict) -> float:
        """计算问题难度"""
        difficulty = 0.0
        
        # 基于问题类型
        type_scores = {
            'factual': 0.2,
            'comparison': 0.4,
            'reasoning': 0.6,
            'multi_hop': 0.7,
            'counterfactual': 0.8,
            'temporal': 0.5,
            'causal': 0.6
        }
        difficulty += type_scores.get(qa.get('type', 'factual'), 0.3)
        
        # 基于子图复杂度
        subgraph = qa.get('subgraph', {})
        complexity = (subgraph.get('num_nodes', 0) / 20.0 + 
                     subgraph.get('num_edges', 0) / 30.0) / 2
        difficulty += min(complexity, 0.3)
        
        # 基于模糊化
        if qa.get('is_obfuscated', False):
            difficulty += 0.2
        
        # 基于推理步骤数
        if 'trajectory' in qa:
            steps = len(qa['trajectory'].get('steps', []))
            difficulty += min(steps / 20.0, 0.2)
        
        return min(difficulty, 1.0)
    
    def _format_qa_pair(self, qa: Dict) -> Dict:
        """格式化QA对"""
        formatted = {
            'id': qa.get('id', self._generate_id()),
            'question': qa['question'],
            'answer': qa['answer'],
            'type': qa.get('type', 'unknown'),
            'language': qa.get('language', 'zh_cn'),
            'difficulty': qa.get('difficulty', 0.5),
            'metadata': {
                'is_obfuscated': qa.get('is_obfuscated', False),
                'obfuscation_strategies': qa.get('obfuscation_strategies', []),
                'reasoning_type': qa.get('reasoning_type', 'deductive'),
                'trajectory_format': qa.get('trajectory_format', 'chain_of_thought')
            }
        }
        
        # 添加子图信息（压缩版）
        if 'subgraph' in qa:
            formatted['subgraph_info'] = {
                'num_nodes': qa['subgraph']['num_nodes'],
                'num_edges': qa['subgraph']['num_edges'],
                'topology': qa['subgraph'].get('topology', 'unknown')
            }
        
        # 添加轨迹（如果存在）
        if 'trajectory' in qa:
            formatted['trajectory'] = {
                'format': qa['trajectory']['format'],
                'reasoning_type': qa['trajectory']['reasoning_type'],
                'num_steps': len(qa['trajectory']['steps']),
                'steps': qa['trajectory']['steps'][:5]  # 只保留前5步
            }
        
        # 添加实体和路径信息
        if 'entities' in qa:
            formatted['entities'] = [e['id'] for e in qa['entities'][:5]]
        
        if 'path_info' in qa:
            formatted['has_path'] = True
            formatted['path_length'] = len(qa['path_info'])
        
        return formatted
    
    def _generate_id(self) -> str:
        """生成唯一ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _split_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """分割数据集"""
        # 按难度分层采样
        data_df = pd.DataFrame(data)
        data_df['difficulty_bin'] = pd.cut(data_df['difficulty'], bins=5)
        
        # 分层分割
        train_data, temp_data = train_test_split(
            data, 
            test_size=(1 - self.train_ratio),
            stratify=data_df['difficulty_bin'],
            random_state=42
        )
        
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=42
        )
        
        return train_data, val_data, test_data
    
    def _augment_data(self, data: List[Dict]) -> List[Dict]:
        """数据增强（可选）"""
        augmented = data.copy()
        
        # 简单的数据增强策略
        # 1. 问题改写（使用同义词替换）
        # 2. 答案扩展（添加解释）
        # 这里只是占位符，实际实现需要更复杂的逻辑
        
        return augmented
    
    def _create_metadata(self, train_data: List[Dict], 
                        val_data: List[Dict], test_data: List[Dict]) -> Dict:
        """创建数据集元数据"""
        metadata = {
            'dataset_name': 'WebSailor TCL Industrial Domain Dataset',
            'version': self.config.get('version', '1.0.0'),
            'domain': self.config.get('domain', 'TCL工业垂域'),
            'creation_date': pd.Timestamp.now().isoformat(),
            'statistics': {
                'total_samples': len(train_data) + len(val_data) + len(test_data),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'test_samples': len(test_data),
                'question_types': self._count_types(train_data + val_data + test_data),
                'languages': self._count_languages(train_data + val_data + test_data),
                'difficulty_distribution': self._get_difficulty_distribution(
                    train_data + val_data + test_data
                )
            },
            'models_used': {
                'expert_model': self.config['models']['expert_model']['path'],
                'qa_generator': self.config['models']['qa_generator_model']['path'],
                'reconstructor': self.config['models']['reconstructor_model']['path'],
                'kg_extractor': self.config['models']['kg_extractor_model']['path']
            }
        }
        
        return metadata
    
    def _count_types(self, data: List[Dict]) -> Dict[str, int]:
        """统计问题类型"""
        type_counts = defaultdict(int)
        for item in data:
            type_counts[item['type']] += 1
        return dict(type_counts)
    
    def _count_languages(self, data: List[Dict]) -> Dict[str, int]:
        """统计语言分布"""
        lang_counts = defaultdict(int)
        for item in data:
            lang_counts[item['language']] += 1
        return dict(lang_counts)
    
    def _get_difficulty_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """获取难度分布"""
        difficulties = [item['difficulty'] for item in data]
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['very_easy', 'easy', 'medium', 'hard', 'very_hard']
        
        counts = pd.cut(difficulties, bins=bins, labels=labels).value_counts()
        return counts.to_dict()
    
    def _export_dataset(self, dataset: Dict, output_path: Path):
        """导出数据集为不同格式"""
        for format_type in self.output_formats:
            if format_type == 'json':
                self._export_json(dataset, output_path)
            elif format_type == 'jsonl':
                self._export_jsonl(dataset, output_path)
            elif format_type == 'csv':
                self._export_csv(dataset, output_path)
    
    def _export_json(self, dataset: Dict, output_path: Path):
        """导出为JSON格式"""
        # 完整数据集
        full_path = output_path / 'websailor_tcl_dataset.json'
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # 分别保存训练、验证、测试集
        for split in ['train', 'validation', 'test']:
            split_path = output_path / f'{split}.json'
            with open(split_path, 'w', encoding='utf-8') as f:
                json.dump(dataset[split], f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON数据集已保存到: {output_path}")
    
    def _export_jsonl(self, dataset: Dict, output_path: Path):
        """导出为JSONL格式"""
        for split in ['train', 'validation', 'test']:
            split_path = output_path / f'{split}.jsonl'
            with jsonlines.open(split_path, mode='w') as writer:
                for item in dataset[split]:
                    writer.write(item)
        
        logger.info(f"JSONL数据集已保存到: {output_path}")
    
    def _export_csv(self, dataset: Dict, output_path: Path):
        """导出为CSV格式（简化版）"""
        for split in ['train', 'validation', 'test']:
            # 提取基本字段
            simplified_data = []
            for item in dataset[split]:
                simplified_data.append({
                    'id': item['id'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'type': item['type'],
                    'language': item['language'],
                    'difficulty': item['difficulty']
                })
            
            df = pd.DataFrame(simplified_data)
            csv_path = output_path / f'{split}.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"CSV数据集已保存到: {output_path}")
    
    def generate_statistics(self, dataset: Dict) -> Dict:
        """生成详细的统计信息"""
        all_data = dataset['train'] + dataset['validation'] + dataset['test']
        
        stats = {
            'overview': {
                'total_qa_pairs': len(all_data),
                'train_size': len(dataset['train']),
                'val_size': len(dataset['validation']),
                'test_size': len(dataset['test']),
                'unique_questions': len(set(item['question'] for item in all_data))
            },
            'question_analysis': {
                'avg_question_length': sum(len(item['question']) for item in all_data) / len(all_data),
                'question_type_distribution': self._count_types(all_data),
                'language_distribution': self._count_languages(all_data)
            },
            'answer_analysis': {
                'avg_answer_length': sum(len(item['answer']) for item in all_data) / len(all_data),
                'answers_with_trajectory': sum(1 for item in all_data if 'trajectory' in item)
            },
            'complexity_analysis': {
                'difficulty_distribution': self._get_difficulty_distribution(all_data),
                'obfuscated_questions': sum(1 for item in all_data 
                                          if item['metadata']['is_obfuscated']),
                'avg_subgraph_nodes': sum(item.get('subgraph_info', {}).get('num_nodes', 0) 
                                        for item in all_data) / len(all_data),
                'avg_subgraph_edges': sum(item.get('subgraph_info', {}).get('num_edges', 0) 
                                        for item in all_data) / len(all_data)
            },
            'trajectory_analysis': {
                'questions_with_trajectory': sum(1 for item in all_data if 'trajectory' in item),
                'trajectory_formats': defaultdict(int),
                'reasoning_types': defaultdict(int)
            }
        }
        
        # 统计轨迹格式和推理类型
        for item in all_data:
            if 'trajectory' in item:
                stats['trajectory_analysis']['trajectory_formats'][item['trajectory']['format']] += 1
                stats['trajectory_analysis']['reasoning_types'][item['trajectory']['reasoning_type']] += 1
        
        stats['trajectory_analysis']['trajectory_formats'] = dict(
            stats['trajectory_analysis']['trajectory_formats']
        )
        stats['trajectory_analysis']['reasoning_types'] = dict(
            stats['trajectory_analysis']['reasoning_types']
        )
        
        return stats