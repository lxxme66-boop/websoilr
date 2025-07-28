"""
知识图谱构建器
从TCL工业领域文本构建知识图谱，为后续子图采样提供基础
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """知识图谱构建器 - 从TCL工业文本构建领域知识图谱"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_types = config['entity_types']
        self.relation_types = config['relation_types']
        self.embedding_model = SentenceTransformer(config['embedding_model'])
        
        # 初始化jieba分词
        self._init_jieba()
        
        # TCL工业领域专业词汇
        self.domain_keywords = {
            '产品': ['电视', '空调', '冰箱', '洗衣机', '手机', '显示器', '音响', 'TCL', '雷鸟'],
            '技术': ['量子点', 'QLED', 'Mini LED', 'AI', '人工智能', '8K', '4K', 'HDR', '杜比'],  
            '工艺': ['制造', '组装', '测试', '质检', '封装', '贴片', '焊接', '注塑'],
            '材料': ['液晶', '背光', '面板', '芯片', '电路板', '塑料', '金属', '玻璃'],
            '设备': ['生产线', '测试仪', '贴片机', '焊接机', '注塑机', '检测设备'],
            '质量指标': ['亮度', '对比度', '色域', '响应时间', '刷新率', '分辨率', '功耗'],
            '性能参数': ['尺寸', '重量', '厚度', '功率', '电压', '频率', '容量'],
            '应用场景': ['家庭', '办公', '教育', '医疗', '商业', '工业', '户外'],
            '问题': ['故障', '缺陷', '异常', '错误', '失效', '损坏', '不良'],
            '解决方案': ['优化', '改进', '修复', '升级', '调整', '更换', '维护']
        }
        
    def _init_jieba(self):
        """初始化jieba分词器，添加TCL工业领域词汇"""
        # 添加TCL工业专业词汇到jieba词典
        for entity_type, keywords in self.domain_keywords.items():
            for keyword in keywords:
                jieba.add_word(keyword)
                
    def build_from_texts(self, input_dir: str) -> nx.Graph:
        """从输入文本目录构建知识图谱"""
        logger.info(f"开始从 {input_dir} 构建知识图谱")
        
        # 读取所有文本文件
        texts = self._read_text_files(input_dir)
        logger.info(f"读取了 {len(texts)} 个文本文件")
        
        # 提取实体和关系
        entities = self._extract_entities(texts)
        relations = self._extract_relations(texts, entities)
        
        # 构建图谱
        graph = self._build_graph(entities, relations)
        
        logger.info(f"知识图谱构建完成: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
        return graph
        
    def _read_text_files(self, input_dir: str) -> List[str]:
        """读取输入目录中的所有文本文件"""
        texts = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            # 如果输入目录不存在，创建示例文本
            logger.warning(f"输入目录 {input_dir} 不存在，创建示例文本")
            input_path.mkdir(parents=True, exist_ok=True)
            self._create_sample_texts(input_path)
        
        # 读取所有.txt文件
        for file_path in input_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
            except Exception as e:
                logger.error(f"读取文件 {file_path} 失败: {e}")
                
        return texts
        
    def _create_sample_texts(self, input_path: Path):
        """创建TCL工业领域的示例文本"""
        sample_texts = [
            """TCL作为全球领先的消费电子品牌，在电视、空调、冰箱等产品领域具有强大的技术实力。
            公司采用量子点QLED技术，提升了显示效果的亮度和色域表现。
            在制造工艺方面，TCL拥有完整的生产线，包括面板制造、组装测试等环节。""",
            
            """Mini LED背光技术是TCL电视的核心技术之一，通过精密的背光控制提升对比度。
            该技术需要高精度的贴片工艺和严格的质量检测流程。
            在实际应用中，Mini LED技术显著改善了HDR内容的显示效果。""",
            
            """TCL智能电视搭载AI人工智能系统，支持语音控制和内容推荐功能。
            系统集成了杜比音效处理技术，提供沉浸式的音频体验。
            产品支持8K分辨率显示，满足高端用户的需求。""",
            
            """在质量控制方面，TCL建立了完善的测试体系，包括亮度测试、色域检测等多项指标。
            生产过程中采用自动化检测设备，确保产品质量的一致性。
            当出现显示异常问题时，通过系统诊断可以快速定位故障原因。""",
            
            """TCL空调产品采用变频技术，在节能和舒适性方面表现优异。
            压缩机是空调的核心部件，直接影响制冷效果和能耗表现。
            通过优化制冷剂循环系统，可以提升空调的整体性能。"""
        ]
        
        for i, text in enumerate(sample_texts):
            file_path = input_path / f"domain_text_{i+1}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
        logger.info(f"创建了 {len(sample_texts)} 个示例文本文件")
        
    def _extract_entities(self, texts: List[str]) -> Dict[str, Dict[str, Any]]:
        """从文本中提取实体"""
        entities = {}
        
        for text in texts:
            # 分词和词性标注
            words = pseg.cut(text)
            
            for word, flag in words:
                word = word.strip()
                if len(word) < 2:  # 过滤过短的词
                    continue
                    
                # 根据词性和领域词汇判断实体类型
                entity_type = self._classify_entity(word, flag)
                if entity_type:
                    if word not in entities:
                        entities[word] = {
                            'type': entity_type,
                            'frequency': 0,
                            'contexts': [],
                            'embedding': None
                        }
                    entities[word]['frequency'] += 1
                    
                    # 保存上下文（前后各5个字符）
                    start_idx = max(0, text.find(word) - 10)
                    end_idx = min(len(text), text.find(word) + len(word) + 10)
                    context = text[start_idx:end_idx]
                    entities[word]['contexts'].append(context)
        
        # 计算实体嵌入
        for entity, info in entities.items():
            contexts = info['contexts'][:5]  # 最多使用5个上下文
            context_text = ' '.join(contexts)
            info['embedding'] = self.embedding_model.encode(context_text)
            
        logger.info(f"提取了 {len(entities)} 个实体")
        return entities
        
    def _classify_entity(self, word: str, pos_flag: str) -> str:
        """根据词汇和词性分类实体类型"""
        # 首先检查是否在领域关键词中
        for entity_type, keywords in self.domain_keywords.items():
            if word in keywords:
                return entity_type
                
        # 根据词性判断
        if pos_flag in ['n', 'nr', 'ns', 'nt', 'nz']:  # 名词类
            # 进一步根据词汇特征判断
            if any(keyword in word for keyword in self.domain_keywords['技术']):
                return '技术'
            elif any(keyword in word for keyword in self.domain_keywords['产品']):
                return '产品'
            elif any(keyword in word for keyword in self.domain_keywords['材料']):
                return '材料'
            elif re.search(r'\d+', word):  # 包含数字的可能是参数
                return '性能参数'
            else:
                return '产品'  # 默认分类为产品
                
        elif pos_flag in ['v', 'vn']:  # 动词类，可能是工艺或解决方案
            return '工艺'
            
        return None
        
    def _extract_relations(self, texts: List[str], entities: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """从文本中提取实体间关系"""
        relations = []
        entity_list = list(entities.keys())
        
        for text in texts:
            # 在每个句子中寻找实体对
            sentences = re.split(r'[。！？]', text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 5:
                    continue
                    
                # 找到句子中的实体
                sentence_entities = []
                for entity in entity_list:
                    if entity in sentence:
                        sentence_entities.append(entity)
                
                # 为实体对推断关系
                for i in range(len(sentence_entities)):
                    for j in range(i+1, len(sentence_entities)):
                        entity1, entity2 = sentence_entities[i], sentence_entities[j]
                        relation = self._infer_relation(entity1, entity2, sentence, entities)
                        if relation:
                            relations.append((entity1, relation, entity2))
        
        logger.info(f"提取了 {len(relations)} 个关系")
        return relations
        
    def _infer_relation(self, entity1: str, entity2: str, sentence: str, entities: Dict) -> str:
        """推断两个实体之间的关系"""
        type1 = entities[entity1]['type']
        type2 = entities[entity2]['type']
        
        # 基于实体类型和句子内容推断关系
        if '包含' in sentence or '组成' in sentence:
            return '包含'
        elif '依赖' in sentence or '需要' in sentence:
            return '依赖'
        elif '影响' in sentence or '导致' in sentence:
            return '影响'
        elif '改进' in sentence or '提升' in sentence:
            return '改进'
        elif '应用' in sentence or '用于' in sentence:
            return '应用于'
        elif '解决' in sentence or '处理' in sentence:
            return '解决'
        elif '优化' in sentence:
            return '优化'
        elif '测试' in sentence or '检测' in sentence:
            return '测试'
        elif '生产' in sentence or '制造' in sentence:
            return '生产'
        else:
            # 基于实体类型的默认关系
            if type1 == '技术' and type2 == '产品':
                return '应用于'
            elif type1 == '工艺' and type2 == '产品':
                return '生产'
            elif type1 == '材料' and type2 == '产品':
                return '包含'
            elif type1 == '问题' and type2 == '解决方案':
                return '解决'
            else:
                return '相关'
                
    def _build_graph(self, entities: Dict[str, Dict[str, Any]], relations: List[Tuple[str, str, str]]) -> nx.Graph:
        """构建知识图谱"""
        graph = nx.Graph()
        
        # 添加节点
        for entity, info in entities.items():
            graph.add_node(entity, **info)
            
        # 添加边
        for entity1, relation, entity2 in relations:
            if entity1 in graph.nodes and entity2 in graph.nodes:
                graph.add_edge(entity1, entity2, relation=relation)
                
        return graph