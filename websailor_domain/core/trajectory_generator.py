"""
推理轨迹生成器
为QA对生成推理轨迹，展示从问题到答案的推理过程
"""

import json
import logging
import random
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """推理轨迹生成器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.traj_config = config.get('trajectory_generation', {})
        
        # 轨迹参数
        self.max_steps = self.traj_config.get('max_steps', 10)
        self.reasoning_types = self.traj_config.get('reasoning_types', [])
        self.trajectory_formats = self.traj_config.get('trajectory_formats', [])
        
        # 加载专家模型
        self._load_expert_model()
        
    def _load_expert_model(self):
        """加载专家模型用于轨迹生成"""
        model_config = self.config['models']['expert_model']
        model_path = model_config['path']
        
        logger.info(f"加载专家模型: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            logger.warning(f"无法加载指定模型，使用默认模型: {e}")
            # 使用较小的默认模型
            self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b")
            self.model = AutoModelForCausalLM.from_pretrained(
                "THUDM/chatglm-6b",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model.eval()
        
        # 设置生成参数
        self.max_length = model_config.get('max_length', 8192)
        self.temperature = model_config.get('temperature', 0.7)
        
    def generate_trajectories(self, qa_pairs: List[Dict]) -> List[Dict]:
        """为QA对生成推理轨迹"""
        qa_with_trajectories = []
        
        for qa in tqdm(qa_pairs, desc="生成推理轨迹"):
            # 选择推理类型和格式
            reasoning_type = self._select_reasoning_type(qa)
            trajectory_format = self._select_trajectory_format(qa)
            
            # 生成轨迹
            trajectory = self._generate_trajectory(
                qa, reasoning_type, trajectory_format
            )
            
            # 添加轨迹到QA对
            qa['trajectory'] = trajectory
            qa['reasoning_type'] = reasoning_type
            qa['trajectory_format'] = trajectory_format
            
            qa_with_trajectories.append(qa)
        
        logger.info(f"轨迹生成完成: {len(qa_with_trajectories)} 个QA对")
        
        return qa_with_trajectories
    
    def _select_reasoning_type(self, qa: Dict) -> str:
        """根据问题类型选择推理类型"""
        q_type = qa.get('type', 'factual')
        
        # 基于问题类型的推理类型映射
        type_mapping = {
            'factual': ['deductive'],
            'comparison': ['analogical', 'deductive'],
            'reasoning': ['deductive', 'inductive'],
            'multi_hop': ['deductive'],
            'counterfactual': ['abductive', 'analogical'],
            'temporal': ['deductive', 'inductive'],
            'causal': ['abductive', 'deductive']
        }
        
        suitable_types = type_mapping.get(q_type, ['deductive'])
        return random.choice(suitable_types)
    
    def _select_trajectory_format(self, qa: Dict) -> str:
        """选择轨迹格式"""
        # 基于问题复杂度选择
        subgraph = qa.get('subgraph', {})
        num_nodes = subgraph.get('num_nodes', 0)
        
        if num_nodes > 10:
            # 复杂问题使用图思维
            return 'graph_of_thought'
        elif num_nodes > 5:
            # 中等复杂度使用树思维
            return 'tree_of_thought'
        else:
            # 简单问题使用链式思维
            return 'chain_of_thought'
    
    def _generate_trajectory(self, qa: Dict, reasoning_type: str, 
                           trajectory_format: str) -> Dict:
        """生成推理轨迹"""
        # 构造提示
        prompt = self._create_trajectory_prompt(qa, reasoning_type, trajectory_format)
        
        # 使用LLM生成轨迹
        trajectory_text = self._generate_with_llm(prompt)
        
        # 解析轨迹
        trajectory_steps = self._parse_trajectory(
            trajectory_text, reasoning_type, trajectory_format
        )
        
        # 构建轨迹对象
        trajectory = {
            'reasoning_type': reasoning_type,
            'format': trajectory_format,
            'steps': trajectory_steps,
            'raw_text': trajectory_text
        }
        
        return trajectory
    
    def _create_trajectory_prompt(self, qa: Dict, reasoning_type: str, 
                                trajectory_format: str) -> str:
        """创建轨迹生成提示"""
        # 格式化子图信息
        subgraph_desc = self._format_subgraph_for_trajectory(qa['subgraph'])
        
        # 格式化问题和答案
        question = qa['question']
        answer = qa['answer']
        
        # 根据轨迹格式构造提示
        if trajectory_format == 'chain_of_thought':
            prompt = f"""请为以下问题生成链式思维（Chain of Thought）推理轨迹。

知识图谱信息：
{subgraph_desc}

问题：{question}
答案：{answer}

推理类型：{reasoning_type}

请生成详细的推理步骤，展示如何从问题和知识图谱得出答案。每个步骤应该清晰、逻辑连贯。

推理轨迹：
步骤1："""
        
        elif trajectory_format == 'tree_of_thought':
            prompt = f"""请为以下问题生成树形思维（Tree of Thought）推理轨迹。

知识图谱信息：
{subgraph_desc}

问题：{question}
答案：{answer}

推理类型：{reasoning_type}

请生成分支式的推理过程，探索多个可能的推理路径，最终收敛到正确答案。

推理轨迹：
主干："""
        
        else:  # graph_of_thought
            prompt = f"""请为以下问题生成图形思维（Graph of Thought）推理轨迹。

知识图谱信息：
{subgraph_desc}

问题：{question}
答案：{answer}

推理类型：{reasoning_type}

请生成网状的推理过程，展示概念之间的多重联系和推理路径。

推理轨迹：
节点1："""
        
        return prompt
    
    def _format_subgraph_for_trajectory(self, subgraph: Dict) -> str:
        """为轨迹生成格式化子图"""
        lines = []
        
        # 节点信息（更详细）
        lines.append("实体信息：")
        for node in subgraph['nodes']:
            lines.append(f"  - {node['id']} (类型: {node['type']}, 置信度: {node.get('confidence', 1.0):.2f})")
        
        # 关系信息（更详细）
        lines.append("\n关系网络：")
        for edge in subgraph['edges']:
            lines.append(f"  - {edge['source']} --[{edge['relation']}]--> {edge['target']}")
        
        # 拓扑信息
        if 'topology' in subgraph:
            lines.append(f"\n拓扑结构：{subgraph['topology']}")
            
        # 关键路径
        if 'path' in subgraph:
            lines.append(f"\n关键路径：{' -> '.join(subgraph['path'])}")
            
        return '\n'.join(lines)
    
    def _generate_with_llm(self, prompt: str) -> str:
        """使用LLM生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                              max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, '').strip()
        
        return generated_text
    
    def _parse_trajectory(self, trajectory_text: str, reasoning_type: str, 
                        trajectory_format: str) -> List[Dict]:
        """解析轨迹文本为结构化步骤"""
        steps = []
        
        if trajectory_format == 'chain_of_thought':
            # 解析链式思维
            step_patterns = [
                r'步骤\d+[：:](.+?)(?=步骤\d+|$)',
                r'Step\s*\d+[：:](.+?)(?=Step\s*\d+|$)',
                r'\d+\.\s*(.+?)(?=\d+\.|$)'
            ]
            
            import re
            for pattern in step_patterns:
                matches = re.findall(pattern, trajectory_text, re.DOTALL)
                if matches:
                    for i, match in enumerate(matches):
                        steps.append({
                            'id': i + 1,
                            'type': 'sequential',
                            'content': match.strip(),
                            'reasoning_type': reasoning_type
                        })
                    break
            
            # 如果没有匹配到模式，按行分割
            if not steps:
                lines = [line.strip() for line in trajectory_text.split('\n') if line.strip()]
                for i, line in enumerate(lines[:self.max_steps]):
                    steps.append({
                        'id': i + 1,
                        'type': 'sequential',
                        'content': line,
                        'reasoning_type': reasoning_type
                    })
        
        elif trajectory_format == 'tree_of_thought':
            # 解析树形思维（简化版）
            # 实际应用中需要更复杂的解析逻辑
            branches = trajectory_text.split('分支')
            for i, branch in enumerate(branches):
                if branch.strip():
                    steps.append({
                        'id': i,
                        'type': 'branch',
                        'content': branch.strip(),
                        'reasoning_type': reasoning_type,
                        'parent': 0 if i == 0 else (i - 1) // 2
                    })
        
        else:  # graph_of_thought
            # 解析图形思维（简化版）
            nodes = trajectory_text.split('节点')
            for i, node in enumerate(nodes):
                if node.strip():
                    steps.append({
                        'id': i,
                        'type': 'node',
                        'content': node.strip(),
                        'reasoning_type': reasoning_type,
                        'connections': []  # 实际应用中需要解析连接关系
                    })
        
        # 确保至少有一些步骤
        if not steps:
            steps.append({
                'id': 1,
                'type': 'summary',
                'content': trajectory_text,
                'reasoning_type': reasoning_type
            })
        
        return steps