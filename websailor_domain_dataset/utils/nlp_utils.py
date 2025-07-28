"""
NLP工具函数
提供文本处理、实体识别、关系抽取等功能
"""

import re
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict

class NLPUtils:
    """NLP工具类"""
    
    def __init__(self):
        # 初始化jieba
        jieba.initialize()
        
        # TCL工业领域词典
        self.domain_dict = {
            '产品': ['电视', '空调', '冰箱', '洗衣机', '显示器', '音响'],
            '技术': ['AI技术', '物联网', '智能制造', '自动化', '机器学习'],
            '材料': ['塑料', '金属', '玻璃', '橡胶', '陶瓷', '复合材料'],
            '工艺': ['注塑', '冲压', '焊接', '装配', '测试', '包装'],
            '设备': ['生产线', '机器人', '传感器', '控制器', '检测仪'],
            '质量指标': ['良品率', '故障率', '精度', '稳定性', '可靠性'],
            '性能参数': ['功率', '效率', '速度', '温度', '压力', '电流']
        }
        
        # 添加领域词汇到jieba
        for category, words in self.domain_dict.items():
            for word in words:
                jieba.add_word(word)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取文本中的实体"""
        entities = []
        words = pseg.cut(text)
        
        for word, flag in words:
            # 基于词性标注识别实体
            if flag in ['n', 'nr', 'ns', 'nt', 'nz']:  # 名词类
                entity_type = self._classify_entity(word)
                entities.append({
                    'text': word,
                    'type': entity_type,
                    'pos': flag
                })
        
        return entities
    
    def _classify_entity(self, word: str) -> str:
        """分类实体类型"""
        for category, words in self.domain_dict.items():
            if word in words:
                return category
        
        # 基于关键词匹配
        if any(keyword in word for keyword in ['率', '度', '量']):
            return '质量指标'
        elif any(keyword in word for keyword in ['机', '器', '设备']):
            return '设备'
        elif any(keyword in word for keyword in ['工艺', '流程', '方法']):
            return '工艺'
        else:
            return '实体'
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """提取文本中的关系"""
        relations = []
        
        # 基于模式匹配的关系抽取
        patterns = [
            (r'(\w+)影响(\w+)', '影响'),
            (r'(\w+)导致(\w+)', '导致'),
            (r'(\w+)包含(\w+)', '包含'),
            (r'(\w+)依赖(\w+)', '依赖'),
            (r'(\w+)应用于(\w+)', '应用于'),
            (r'(\w+)改进(\w+)', '改进'),
            (r'(\w+)优化(\w+)', '优化'),
            (r'(\w+)测试(\w+)', '测试'),
            (r'(\w+)生产(\w+)', '生产'),
            (r'(\w+)解决(\w+)', '解决')
        ]
        
        for pattern, relation_type in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                relations.append({
                    'subject': match.group(1),
                    'relation': relation_type,
                    'object': match.group(2),
                    'confidence': 0.8
                })
        
        return relations
    
    def segment_text(self, text: str) -> List[str]:
        """文本分词"""
        return list(jieba.cut(text))
    
    def extract_keywords(self, text: str, topk: int = 10) -> List[str]:
        """提取关键词"""
        import jieba.analyse
        return jieba.analyse.extract_tags(text, topK=topk)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        words1 = set(self.segment_text(text1))
        words2 = set(self.segment_text(text2))
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 去除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def is_question(self, text: str) -> bool:
        """判断是否为问句"""
        question_markers = ['？', '?', '什么', '哪个', '如何', '为什么', '怎么']
        return any(marker in text for marker in question_markers)
    
    def extract_question_type(self, question: str) -> str:
        """识别问题类型"""
        if any(word in question for word in ['什么', '哪个', '哪些']):
            return 'factual'
        elif any(word in question for word in ['关系', '影响', '作用']):
            return 'relational'
        elif any(word in question for word in ['如何', '怎么', '通过什么']):
            return 'multi_hop'
        elif any(word in question for word in ['比较', '不同', '优势']):
            return 'comparative'
        elif any(word in question for word in ['为什么', '原因', '如果']):
            return 'reasoning'
        elif any(word in question for word in ['路径', '步骤', '过程']):
            return 'path_finding'
        else:
            return 'general'