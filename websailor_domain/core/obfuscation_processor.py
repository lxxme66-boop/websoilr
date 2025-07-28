"""
模糊化处理器
实现WebSailor的核心思想：模糊描述中间实体或关系
添加冗余或干扰信息,使问题信息密度高但精确信息少
"""

import logging
import random
from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class ObfuscationProcessor:
    """
    模糊化处理器
    WebSailor核心组件：通过模糊化增加问题的不确定性和推理难度
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 模糊化级别
        self.obfuscation_levels = config['data_settings'].get(
            'obfuscation_levels', 
            [0.2, 0.4, 0.6]
        )
        
        # 初始化模糊化模板
        self._init_obfuscation_patterns()
        
    def _init_obfuscation_patterns(self):
        """初始化模糊化模式"""
        # 实体模糊化模板
        self.entity_patterns = {
            'zh': {
                '产品': ['这种产品', '某个产品', '相关产品', '该产品', '一种产品'],
                '技术': ['这项技术', '某种技术', '相关技术', '该技术', '一种技术'],
                '材料': ['这种材料', '某种材料', '相关材料', '该材料', '一种材料'],
                '设备': ['这台设备', '某个设备', '相关设备', '该设备', '一种设备'],
                '公司': ['这家公司', '某公司', '相关企业', '该企业', '一家公司'],
                '工艺': ['这种工艺', '某个工艺', '相关工艺', '该工艺', '一种工艺'],
                'default': ['这个{TYPE}', '某个{TYPE}', '相关的{TYPE}', '该{TYPE}', '一种{TYPE}']
            },
            'en': {
                '产品': ['this product', 'a certain product', 'related product', 'the product', 'a product'],
                '技术': ['this technology', 'a certain technology', 'related technology', 'the technology', 'a technology'],
                '材料': ['this material', 'a certain material', 'related material', 'the material', 'a material'],
                '设备': ['this equipment', 'certain equipment', 'related equipment', 'the equipment', 'an equipment'],
                '公司': ['this company', 'a certain company', 'related enterprise', 'the enterprise', 'a company'],
                '工艺': ['this process', 'a certain process', 'related process', 'the process', 'a process'],
                'default': ['this {TYPE}', 'a certain {TYPE}', 'related {TYPE}', 'the {TYPE}', 'a {TYPE}']
            }
        }
        
        # 关系模糊化模板
        self.relation_patterns = {
            'zh': {
                '使用': ['与...有关', '涉及到', '关联于', '相关的', '存在某种联系'],
                '包含': ['涉及', '相关', '包括某些', '有关于', '存在关联'],
                '生产': ['相关于', '涉及到', '有关', '存在联系', '相关的'],
                '研发': ['开展相关工作', '进行某些活动', '有所涉及', '存在关联', '相关研究'],
                '依赖': ['需要', '相关', '有所关联', '存在某种关系', '涉及到'],
                'default': ['与...有关', '涉及到', '关联于', '相关的', '存在某种联系']
            },
            'en': {
                '使用': ['related to', 'involves', 'associated with', 'connected to', 'has some connection'],
                '包含': ['involves', 'related', 'includes some', 'concerning', 'has association'],
                '生产': ['related to', 'involves', 'concerning', 'has connection', 'associated'],
                '研发': ['conducts related work', 'performs activities', 'has involvement', 'has association', 'related research'],
                '依赖': ['requires', 'related', 'has association', 'has some relationship', 'involves'],
                'default': ['related to', 'involves', 'associated with', 'connected to', 'has some connection']
            }
        }
        
        # 干扰信息模板
        self.noise_templates = {
            'zh': [
                "值得注意的是，{noise_fact}。",
                "另外，{noise_fact}也是需要考虑的因素。",
                "同时，{noise_fact}可能会产生影响。",
                "此外，还存在{noise_fact}的情况。",
                "需要说明的是，{noise_fact}。"
            ],
            'en': [
                "It's worth noting that {noise_fact}.",
                "Additionally, {noise_fact} is also a factor to consider.",
                "Meanwhile, {noise_fact} may have an impact.",
                "Furthermore, there is {noise_fact}.",
                "It should be noted that {noise_fact}."
            ]
        }
        
        # TCL垂域干扰事实
        self.noise_facts = {
            'zh': [
                "行业标准在不断更新",
                "技术发展具有不确定性",
                "市场需求在持续变化",
                "存在多种技术路线",
                "不同应用场景有不同要求",
                "成本因素需要综合考虑",
                "环保要求日益严格",
                "国际竞争日趋激烈"
            ],
            'en': [
                "industry standards are constantly updating",
                "technological development has uncertainties",
                "market demand is continuously changing",
                "multiple technical routes exist",
                "different application scenarios have different requirements",
                "cost factors need comprehensive consideration",
                "environmental requirements are increasingly strict",
                "international competition is intensifying"
            ]
        }
        
    def obfuscate_questions(self, questions: List[Dict], 
                           target_level: Optional[float] = None) -> List[Dict]:
        """
        对问题进行模糊化处理
        
        Args:
            questions: 原始问题列表
            target_level: 目标模糊化级别（0-1）
            
        Returns:
            List[Dict]: 模糊化后的问题列表
        """
        logger.info(f"开始对{len(questions)}个问题进行模糊化处理...")
        
        obfuscated_questions = []
        
        for question in questions:
            # 确定模糊化级别
            if target_level is None:
                level = random.choice(self.obfuscation_levels)
            else:
                level = target_level
                
            # 根据问题类型和难度调整级别
            adjusted_level = self._adjust_obfuscation_level(question, level)
            
            # 执行模糊化
            obfuscated = self._obfuscate_single_question(question, adjusted_level)
            obfuscated_questions.append(obfuscated)
            
        logger.info(f"模糊化处理完成，共处理{len(obfuscated_questions)}个问题")
        return obfuscated_questions
        
    def _adjust_obfuscation_level(self, question: Dict, base_level: float) -> float:
        """
        根据问题特征调整模糊化级别
        WebSailor思想：不同类型的问题需要不同程度的模糊化
        """
        q_type = question.get('type', 'factual')
        difficulty = question.get('difficulty', 0.5)
        
        # 根据问题类型调整
        type_adjustments = {
            'factual': -0.1,      # 事实类问题减少模糊化
            'reasoning': 0.1,     # 推理类问题增加模糊化
            'multi_hop': 0.2,     # 多跳问题大幅增加模糊化
            'comparative': 0.0,   # 比较类问题保持不变
            'causal': 0.15        # 因果类问题增加模糊化
        }
        
        adjustment = type_adjustments.get(q_type, 0.0)
        
        # 根据难度调整
        if difficulty > 0.7:
            adjustment += 0.1
        elif difficulty < 0.3:
            adjustment -= 0.1
            
        # 确保在合理范围内
        return max(0.1, min(0.9, base_level + adjustment))
        
    def _obfuscate_single_question(self, question: Dict, level: float) -> Dict:
        """
        对单个问题进行模糊化
        """
        obfuscated = question.copy()
        
        # 获取语言
        lang = question.get('language', 'zh')
        
        # 1. 实体模糊化
        obfuscated['question'] = self._obfuscate_entities(
            question['question'], 
            question.get('entities', []),
            level,
            lang
        )
        
        # 2. 关系模糊化（如果有）
        if 'relations' in question:
            obfuscated['question'] = self._obfuscate_relations(
                obfuscated['question'],
                question['relations'],
                level,
                lang
            )
            
        # 3. 添加干扰信息
        if random.random() < level:
            obfuscated['question'] = self._add_noise(
                obfuscated['question'],
                lang
            )
            
        # 4. 答案模糊化（部分模糊）
        if 'answer' in question and random.random() < level * 0.5:
            obfuscated['answer'] = self._obfuscate_answer(
                question['answer'],
                level,
                lang
            )
            
        # 记录模糊化信息
        obfuscated['obfuscation_level'] = level
        obfuscated['obfuscation_types'] = self._get_applied_obfuscations(
            question, obfuscated
        )
        
        return obfuscated
        
    def _obfuscate_entities(self, text: str, entities: List[str], 
                           level: float, lang: str) -> str:
        """
        实体模糊化
        WebSailor核心：用模糊描述替换具体实体
        """
        obfuscated_text = text
        
        # 统计实体类型
        entity_types = self._identify_entity_types(entities)
        
        # 根据级别决定模糊化的实体数量
        num_to_obfuscate = int(len(entities) * level)
        entities_to_obfuscate = random.sample(entities, num_to_obfuscate) if entities else []
        
        for entity in entities_to_obfuscate:
            # 获取实体类型
            entity_type = entity_types.get(entity, 'default')
            
            # 选择模糊化模式
            patterns = self.entity_patterns[lang].get(
                entity_type, 
                self.entity_patterns[lang]['default']
            )
            pattern = random.choice(patterns)
            
            # 如果是默认模式，需要填充类型
            if '{TYPE}' in pattern:
                pattern = pattern.format(TYPE=entity_type)
                
            # 执行替换
            # 使用正则确保完整匹配
            obfuscated_text = re.sub(
                r'\b' + re.escape(entity) + r'\b',
                pattern,
                obfuscated_text
            )
            
        return obfuscated_text
        
    def _obfuscate_relations(self, text: str, relations: List[str], 
                            level: float, lang: str) -> str:
        """
        关系模糊化
        """
        obfuscated_text = text
        
        # 根据级别决定模糊化的关系数量
        num_to_obfuscate = int(len(relations) * level)
        relations_to_obfuscate = random.sample(relations, num_to_obfuscate) if relations else []
        
        for relation in relations_to_obfuscate:
            # 选择模糊化模式
            patterns = self.relation_patterns[lang].get(
                relation,
                self.relation_patterns[lang]['default']
            )
            pattern = random.choice(patterns)
            
            # 执行替换
            if relation in obfuscated_text:
                obfuscated_text = obfuscated_text.replace(relation, pattern)
                
        return obfuscated_text
        
    def _add_noise(self, text: str, lang: str) -> str:
        """
        添加干扰信息
        WebSailor核心：增加信息密度但减少精确信息
        """
        # 选择噪声模板
        template = random.choice(self.noise_templates[lang])
        
        # 选择噪声事实
        noise_fact = random.choice(self.noise_facts[lang])
        
        # 生成噪声句子
        noise_sentence = template.format(noise_fact=noise_fact)
        
        # 决定插入位置
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            # 在中间插入
            insert_pos = random.randint(1, len(sentences) - 1)
            sentences.insert(insert_pos, noise_sentence)
        else:
            # 在末尾添加
            sentences.append(noise_sentence)
            
        # 重新组合
        if lang == 'zh':
            return ''.join(sentences)
        else:
            return ' '.join(sentences)
            
    def _obfuscate_answer(self, answer: str, level: float, lang: str) -> str:
        """
        答案模糊化（轻度）
        保持答案的正确性，但增加一些不确定性表达
        """
        # 不确定性前缀
        uncertainty_prefixes = {
            'zh': [
                "根据现有信息，",
                "一般来说，",
                "通常情况下，",
                "可能是",
                "据了解，"
            ],
            'en': [
                "Based on available information, ",
                "Generally speaking, ",
                "Typically, ",
                "It might be ",
                "As far as we know, "
            ]
        }
        
        # 不确定性后缀
        uncertainty_suffixes = {
            'zh': [
                "，但具体情况可能有所不同。",
                "，需要根据实际情况判断。",
                "，这是一般性的理解。",
                "，但可能存在其他因素。",
                "，仅供参考。"
            ],
            'en': [
                ", but specific situations may vary.",
                ", needs to be judged based on actual circumstances.",
                ", this is a general understanding.",
                ", but other factors may exist.",
                ", for reference only."
            ]
        }
        
        # 根据级别决定是否添加不确定性
        if random.random() < level * 0.7:
            # 添加前缀
            if random.random() < 0.5:
                prefix = random.choice(uncertainty_prefixes[lang])
                answer = prefix + answer
                
            # 添加后缀
            if random.random() < 0.5:
                suffix = random.choice(uncertainty_suffixes[lang])
                answer = answer + suffix
                
        return answer
        
    def _identify_entity_types(self, entities: List[str]) -> Dict[str, str]:
        """
        识别实体类型
        简化版本，实际应该从知识图谱中获取
        """
        entity_types = {}
        
        # TCL垂域关键词映射
        type_keywords = {
            '产品': ['产品', '系列', '型号', 'TCL', 'Q10G', 'X11G'],
            '技术': ['技术', '算法', 'AI', '智能', 'Mini-LED', 'QLED'],
            '材料': ['材料', '基板', '涂层', '薄膜', '玻璃'],
            '设备': ['设备', '生产线', '机器', '系统', '平台'],
            '公司': ['公司', '企业', '集团', 'TCL', '华星'],
            '工艺': ['工艺', '流程', '制程', '方法', '步骤']
        }
        
        for entity in entities:
            entity_type = 'default'
            
            # 基于关键词匹配
            for type_name, keywords in type_keywords.items():
                if any(keyword in entity for keyword in keywords):
                    entity_type = type_name
                    break
                    
            entity_types[entity] = entity_type
            
        return entity_types
        
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        # 中文分句
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            sentences = re.split(r'[。！？]', text)
            # 保留标点
            result = []
            for i, sent in enumerate(sentences[:-1]):
                if sent:
                    result.append(sent + text[text.find(sent) + len(sent)])
            if sentences[-1]:
                result.append(sentences[-1])
            return result
        else:
            # 英文分句
            sentences = re.split(r'[.!?]', text)
            # 保留标点
            result = []
            for i, sent in enumerate(sentences[:-1]):
                if sent.strip():
                    result.append(sent.strip() + text[text.find(sent) + len(sent)])
            if sentences[-1].strip():
                result.append(sentences[-1].strip())
            return result
            
    def _get_applied_obfuscations(self, original: Dict, obfuscated: Dict) -> List[str]:
        """
        记录应用的模糊化类型
        """
        applied = []
        
        # 检查实体模糊化
        if original['question'] != obfuscated['question']:
            if any(entity not in obfuscated['question'] 
                   for entity in original.get('entities', [])):
                applied.append('entity_obfuscation')
                
        # 检查关系模糊化
        if 'relations' in original:
            if any(relation not in obfuscated['question'] 
                   for relation in original['relations']):
                applied.append('relation_obfuscation')
                
        # 检查噪声添加
        if len(obfuscated['question']) > len(original['question']) * 1.2:
            applied.append('noise_injection')
            
        # 检查答案模糊化
        if original.get('answer', '') != obfuscated.get('answer', ''):
            applied.append('answer_uncertainty')
            
        return applied
        
    def analyze_obfuscation_impact(self, 
                                  original_questions: List[Dict],
                                  obfuscated_questions: List[Dict]) -> Dict:
        """
        分析模糊化的影响
        """
        analysis = {
            'total_questions': len(original_questions),
            'obfuscation_stats': defaultdict(int),
            'average_text_expansion': 0,
            'entity_obfuscation_rate': 0,
            'relation_obfuscation_rate': 0
        }
        
        total_expansion = 0
        total_entities = 0
        obfuscated_entities = 0
        
        for orig, obfusc in zip(original_questions, obfuscated_questions):
            # 统计模糊化类型
            for obf_type in obfusc.get('obfuscation_types', []):
                analysis['obfuscation_stats'][obf_type] += 1
                
            # 计算文本膨胀率
            orig_len = len(orig['question'])
            obfusc_len = len(obfusc['question'])
            total_expansion += (obfusc_len - orig_len) / orig_len
            
            # 统计实体模糊化率
            orig_entities = orig.get('entities', [])
            total_entities += len(orig_entities)
            for entity in orig_entities:
                if entity not in obfusc['question']:
                    obfuscated_entities += 1
                    
        # 计算平均值
        analysis['average_text_expansion'] = total_expansion / len(original_questions)
        analysis['entity_obfuscation_rate'] = (
            obfuscated_entities / total_entities if total_entities > 0 else 0
        )
        
        return analysis