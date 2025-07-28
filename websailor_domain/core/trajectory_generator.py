"""
推理轨迹生成器
生成从问题到答案的推理轨迹，用于训练模型的推理能力
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrajectoryGenerator:
    """
    推理轨迹生成器
    WebSailor思想：生成高质量的推理轨迹，展示如何从问题推导到答案
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 轨迹步数配置
        self.trajectory_steps = config['data_settings'].get(
            'trajectory_steps',
            [3, 5, 7, 10]
        )
        
        # 初始化专家模型
        self._initialize_expert_model()
        
        # 推理模式
        self.reasoning_patterns = {
            'deductive': self._generate_deductive_trajectory,
            'inductive': self._generate_inductive_trajectory,
            'abductive': self._generate_abductive_trajectory,
            'analogical': self._generate_analogical_trajectory
        }
        
    def _initialize_expert_model(self):
        """初始化专家模型（用于生成轨迹）"""
        logger.info("初始化专家模型...")
        
        model_config = self.config['models']['expert_model']
        model_path = model_config['path']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # 生成配置
        self.generation_config = {
            'max_length': model_config.get('max_length', 8192),
            'temperature': model_config.get('temperature', 0.7),
            'top_p': model_config.get('top_p', 0.95),
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        
    def generate_trajectories(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        为QA对生成推理轨迹
        
        Args:
            qa_pairs: QA对列表
            
        Returns:
            List[Dict]: 带有推理轨迹的QA对
        """
        logger.info(f"开始为{len(qa_pairs)}个QA对生成推理轨迹...")
        
        qa_with_trajectories = []
        
        for qa in tqdm(qa_pairs, desc="生成推理轨迹"):
            # 选择合适的推理模式
            reasoning_type = self._select_reasoning_type(qa)
            
            # 生成轨迹
            trajectory = self._generate_trajectory(qa, reasoning_type)
            
            # 添加轨迹到QA对
            qa_with_trajectory = qa.copy()
            qa_with_trajectory['trajectory'] = trajectory
            qa_with_trajectory['reasoning_type'] = reasoning_type
            
            qa_with_trajectories.append(qa_with_trajectory)
            
        logger.info(f"推理轨迹生成完成，共生成{len(qa_with_trajectories)}个轨迹")
        return qa_with_trajectories
        
    def _select_reasoning_type(self, qa: Dict) -> str:
        """
        根据问题类型选择推理模式
        """
        q_type = qa.get('type', 'factual')
        
        # 问题类型到推理模式的映射
        type_to_reasoning = {
            'factual': 'deductive',      # 事实类使用演绎推理
            'reasoning': 'deductive',    # 推理类使用演绎推理
            'multi_hop': 'deductive',    # 多跳使用演绎推理
            'comparative': 'analogical', # 比较类使用类比推理
            'causal': 'abductive'        # 因果类使用溯因推理
        }
        
        return type_to_reasoning.get(q_type, 'deductive')
        
    def _generate_trajectory(self, qa: Dict, reasoning_type: str) -> Dict:
        """
        生成推理轨迹
        """
        # 获取推理函数
        reasoning_func = self.reasoning_patterns.get(
            reasoning_type, 
            self._generate_deductive_trajectory
        )
        
        # 生成轨迹
        trajectory = reasoning_func(qa)
        
        # 后处理
        trajectory = self._postprocess_trajectory(trajectory, qa)
        
        return trajectory
        
    def _generate_deductive_trajectory(self, qa: Dict) -> Dict:
        """
        生成演绎推理轨迹
        从一般原理推导到具体结论
        """
        question = qa['question']
        answer = qa['answer']
        entities = qa.get('entities', [])
        
        # 确定步数
        num_steps = self._determine_trajectory_steps(qa)
        
        # 构建提示
        prompt = f"""请为以下问题生成一个{num_steps}步的演绎推理轨迹。

问题：{question}
答案：{answer}

要求：
1. 从一般原理或已知事实开始
2. 逐步推导，每一步都基于前一步
3. 最终得出答案
4. 每步包含：思考过程、使用的知识、得出的结论

请按以下格式生成推理轨迹：

步骤1：[初始观察]
思考：...
知识：...
结论：...

步骤2：[推理过程]
思考：...
知识：...
结论：...

...

步骤{num_steps}：[最终结论]
思考：...
知识：...
结论：{answer}
"""

        # 生成轨迹
        trajectory_text = self._generate_with_model(prompt)
        
        # 解析轨迹
        trajectory = self._parse_trajectory(trajectory_text, num_steps)
        
        return {
            'type': 'deductive',
            'steps': trajectory,
            'num_steps': num_steps,
            'entities_used': entities
        }
        
    def _generate_inductive_trajectory(self, qa: Dict) -> Dict:
        """
        生成归纳推理轨迹
        从具体观察推导到一般规律
        """
        question = qa['question']
        answer = qa['answer']
        
        num_steps = self._determine_trajectory_steps(qa)
        
        prompt = f"""请为以下问题生成一个{num_steps}步的归纳推理轨迹。

问题：{question}
答案：{answer}

要求：
1. 从具体的观察或例子开始
2. 寻找模式或规律
3. 归纳出一般性结论
4. 应用到具体问题

请按格式生成推理轨迹，每步包含：观察、模式识别、归纳。
"""

        trajectory_text = self._generate_with_model(prompt)
        trajectory = self._parse_trajectory(trajectory_text, num_steps)
        
        return {
            'type': 'inductive',
            'steps': trajectory,
            'num_steps': num_steps
        }
        
    def _generate_abductive_trajectory(self, qa: Dict) -> Dict:
        """
        生成溯因推理轨迹
        从结果推导最可能的原因
        """
        question = qa['question']
        answer = qa['answer']
        
        num_steps = self._determine_trajectory_steps(qa)
        
        prompt = f"""请为以下问题生成一个{num_steps}步的溯因推理轨迹。

问题：{question}
答案：{answer}

要求：
1. 识别需要解释的现象或结果
2. 列举可能的原因
3. 评估每个原因的可能性
4. 选择最佳解释

请按格式生成推理轨迹，每步包含：现象分析、假设生成、假设评估。
"""

        trajectory_text = self._generate_with_model(prompt)
        trajectory = self._parse_trajectory(trajectory_text, num_steps)
        
        return {
            'type': 'abductive',
            'steps': trajectory,
            'num_steps': num_steps
        }
        
    def _generate_analogical_trajectory(self, qa: Dict) -> Dict:
        """
        生成类比推理轨迹
        通过相似性进行推理
        """
        question = qa['question']
        answer = qa['answer']
        
        num_steps = self._determine_trajectory_steps(qa)
        
        prompt = f"""请为以下问题生成一个{num_steps}步的类比推理轨迹。

问题：{question}
答案：{answer}

要求：
1. 识别源领域（熟悉的）
2. 识别目标领域（需要理解的）
3. 建立映射关系
4. 基于映射进行推理

请按格式生成推理轨迹，每步包含：相似性识别、映射建立、推理转移。
"""

        trajectory_text = self._generate_with_model(prompt)
        trajectory = self._parse_trajectory(trajectory_text, num_steps)
        
        return {
            'type': 'analogical',
            'steps': trajectory,
            'num_steps': num_steps
        }
        
    def _determine_trajectory_steps(self, qa: Dict) -> int:
        """
        根据问题复杂度确定轨迹步数
        """
        difficulty = qa.get('difficulty', 0.5)
        q_type = qa.get('type', 'factual')
        
        # 基础步数
        if difficulty < 0.3:
            base_steps = 3
        elif difficulty < 0.6:
            base_steps = 5
        elif difficulty < 0.8:
            base_steps = 7
        else:
            base_steps = 10
            
        # 根据问题类型调整
        type_adjustments = {
            'factual': 0,
            'reasoning': 1,
            'multi_hop': 2,
            'comparative': 1,
            'causal': 2
        }
        
        adjustment = type_adjustments.get(q_type, 0)
        
        # 确保在配置的范围内
        final_steps = base_steps + adjustment
        available_steps = self.trajectory_steps
        
        # 选择最接近的配置值
        return min(available_steps, key=lambda x: abs(x - final_steps))
        
    def _generate_with_model(self, prompt: str) -> str:
        """
        使用专家模型生成文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分
        if "请按" in response and "格式生成推理轨迹" in response:
            # 找到实际生成的轨迹部分
            parts = response.split("格式生成推理轨迹")
            if len(parts) > 1:
                response = parts[-1].strip()
                
        return response
        
    def _parse_trajectory(self, trajectory_text: str, expected_steps: int) -> List[Dict]:
        """
        解析轨迹文本为结构化格式
        """
        steps = []
        
        # 尝试按步骤分割
        step_markers = [f"步骤{i}" for i in range(1, expected_steps + 1)]
        
        current_step = None
        current_content = []
        
        lines = trajectory_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 检查是否是新步骤
            is_new_step = False
            for i, marker in enumerate(step_markers):
                if marker in line:
                    # 保存前一步
                    if current_step is not None:
                        steps.append(self._parse_step_content(
                            current_step, 
                            '\n'.join(current_content)
                        ))
                    
                    current_step = i + 1
                    current_content = [line]
                    is_new_step = True
                    break
                    
            if not is_new_step and current_step is not None:
                current_content.append(line)
                
        # 保存最后一步
        if current_step is not None:
            steps.append(self._parse_step_content(
                current_step, 
                '\n'.join(current_content)
            ))
            
        # 如果解析失败，创建默认轨迹
        if len(steps) < expected_steps:
            steps = self._create_default_trajectory(trajectory_text, expected_steps)
            
        return steps
        
    def _parse_step_content(self, step_num: int, content: str) -> Dict:
        """
        解析单个步骤的内容
        """
        step_data = {
            'step': step_num,
            'thought': '',
            'knowledge': '',
            'conclusion': '',
            'raw_content': content
        }
        
        # 尝试提取结构化信息
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('思考：') or line.startswith('Thought:'):
                step_data['thought'] = line.split('：', 1)[-1].strip()
            elif line.startswith('知识：') or line.startswith('Knowledge:'):
                step_data['knowledge'] = line.split('：', 1)[-1].strip()
            elif line.startswith('结论：') or line.startswith('Conclusion:'):
                step_data['conclusion'] = line.split('：', 1)[-1].strip()
                
        # 如果没有提取到结构化信息，使用原始内容
        if not step_data['thought']:
            step_data['thought'] = content
            
        return step_data
        
    def _create_default_trajectory(self, text: str, num_steps: int) -> List[Dict]:
        """
        创建默认轨迹（当解析失败时）
        """
        # 将文本分成大致相等的部分
        sentences = text.split('。')
        step_size = max(1, len(sentences) // num_steps)
        
        steps = []
        for i in range(num_steps):
            start_idx = i * step_size
            end_idx = min((i + 1) * step_size, len(sentences))
            
            step_content = '。'.join(sentences[start_idx:end_idx])
            
            steps.append({
                'step': i + 1,
                'thought': step_content,
                'knowledge': '',
                'conclusion': '',
                'raw_content': step_content
            })
            
        return steps
        
    def _postprocess_trajectory(self, trajectory: Dict, qa: Dict) -> Dict:
        """
        后处理轨迹，确保质量和一致性
        """
        # 确保最后一步包含答案
        if trajectory['steps']:
            last_step = trajectory['steps'][-1]
            if qa['answer'] not in last_step.get('conclusion', ''):
                last_step['conclusion'] = qa['answer']
                
        # 添加元信息
        trajectory['question'] = qa['question']
        trajectory['answer'] = qa['answer']
        trajectory['question_type'] = qa.get('type', 'unknown')
        trajectory['difficulty'] = qa.get('difficulty', 0.5)
        
        # 验证轨迹连贯性
        trajectory['coherence_score'] = self._evaluate_coherence(trajectory)
        
        return trajectory
        
    def _evaluate_coherence(self, trajectory: Dict) -> float:
        """
        评估轨迹的连贯性
        """
        steps = trajectory.get('steps', [])
        
        if len(steps) < 2:
            return 1.0
            
        # 简单的连贯性评分
        # 检查步骤之间是否有逻辑联系
        coherence_score = 1.0
        
        for i in range(1, len(steps)):
            prev_step = steps[i-1]
            curr_step = steps[i]
            
            # 检查是否有内容
            if not curr_step.get('thought'):
                coherence_score -= 0.1
                
            # 检查是否与前一步有关联
            # 这里可以使用更复杂的方法，如语义相似度
            prev_content = str(prev_step.get('conclusion', '')) + str(prev_step.get('thought', ''))
            curr_content = str(curr_step.get('thought', ''))
            
            # 简单检查：是否有共同的实体
            if not any(word in curr_content for word in prev_content.split() if len(word) > 2):
                coherence_score -= 0.05
                
        return max(0.0, coherence_score)
        
    def save_trajectories(self, qa_with_trajectories: List[Dict], output_path: str):
        """
        保存带轨迹的QA对
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_with_trajectories, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已保存{len(qa_with_trajectories)}个带轨迹的QA对到: {output_path}")