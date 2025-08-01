"""
文档相关性检查器
验证生成的问答对是否基于原始文档内容
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import jieba
from collections import Counter
import re


class DocumentRelevanceChecker:
    """文档相关性检查器"""
    
    def __init__(self, documents: List[str] = None):
        """
        初始化检查器
        
        Args:
            documents: 原始文档列表
        """
        self.logger = logging.getLogger('DocumentRelevanceChecker')
        self.documents = documents or []
        
        # 初始化句子嵌入模型
        try:
            self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}")
            self.model = None
        
        # 预处理文档
        self._preprocess_documents()
    
    def _preprocess_documents(self):
        """预处理文档"""
        if not self.documents:
            return
        
        # 构建文档索引
        self.doc_sentences = []
        self.doc_index = []  # 记录每个句子属于哪个文档
        
        for doc_idx, doc in enumerate(self.documents):
            # 分句
            sentences = self._split_sentences(doc)
            self.doc_sentences.extend(sentences)
            self.doc_index.extend([doc_idx] * len(sentences))
        
        # 计算文档嵌入
        if self.model and self.doc_sentences:
            self.doc_embeddings = self.model.encode(self.doc_sentences)
        
        # 构建关键词索引
        self._build_keyword_index()
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        # 中文分句
        sentences = re.split(r'[。！？；\n]+', text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _build_keyword_index(self):
        """构建关键词索引"""
        self.keyword_index = {}
        
        for doc_idx, doc in enumerate(self.documents):
            # 分词
            words = jieba.lcut(doc)
            # 过滤停用词和短词
            keywords = [w for w in words if len(w) >= 2]
            
            # 统计词频
            word_freq = Counter(keywords)
            
            # 记录每个关键词出现的文档
            for word, freq in word_freq.items():
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                self.keyword_index[word].append({
                    'doc_idx': doc_idx,
                    'frequency': freq
                })
    
    def check_question_relevance(self, question: str) -> Dict[str, float]:
        """
        检查问题与文档的相关性
        
        Args:
            question: 问题文本
            
        Returns:
            相关性得分字典
        """
        scores = {
            'semantic_relevance': 0.0,
            'keyword_relevance': 0.0,
            'factual_grounding': 0.0,
            'overall_relevance': 0.0
        }
        
        if not self.documents:
            return scores
        
        # 1. 语义相关性
        if self.model and hasattr(self, 'doc_embeddings'):
            question_embedding = self.model.encode([question])
            similarities = np.dot(self.doc_embeddings, question_embedding.T).flatten()
            scores['semantic_relevance'] = float(np.max(similarities))
        
        # 2. 关键词相关性
        scores['keyword_relevance'] = self._compute_keyword_relevance(question)
        
        # 3. 事实基础性（检查问题中的实体是否在文档中出现）
        scores['factual_grounding'] = self._check_factual_grounding(question)
        
        # 4. 综合相关性
        weights = {
            'semantic_relevance': 0.4,
            'keyword_relevance': 0.3,
            'factual_grounding': 0.3
        }
        
        scores['overall_relevance'] = sum(
            scores[metric] * weight 
            for metric, weight in weights.items()
        )
        
        return scores
    
    def _compute_keyword_relevance(self, text: str) -> float:
        """计算关键词相关性"""
        if not self.keyword_index:
            return 0.0
        
        # 分词
        words = jieba.lcut(text)
        keywords = [w for w in words if len(w) >= 2]
        
        if not keywords:
            return 0.0
        
        # 计算匹配的关键词比例
        matched_keywords = 0
        for keyword in keywords:
            if keyword in self.keyword_index:
                matched_keywords += 1
        
        return matched_keywords / len(keywords)
    
    def _check_factual_grounding(self, question: str) -> float:
        """检查事实基础性"""
        # 提取可能的实体（技术术语、专有名词等）
        entities = self._extract_entities(question)
        
        if not entities:
            return 0.5  # 没有明确实体，给中等分数
        
        # 检查实体在文档中的出现情况
        found_entities = 0
        for entity in entities:
            for doc in self.documents:
                if entity in doc:
                    found_entities += 1
                    break
        
        return found_entities / len(entities)
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体（简化版）"""
        entities = []
        
        # 提取英文缩写和技术术语
        english_terms = re.findall(r'\b[A-Z][A-Za-z]*\b|\b[A-Z]+\b', text)
        entities.extend(english_terms)
        
        # 提取可能的专有名词（连续的名词）
        words = jieba.lcut(text)
        pos_tags = jieba.posseg.lcut(text)
        
        for word, pos in pos_tags:
            if pos in ['n', 'nr', 'ns', 'nt', 'nz'] and len(word) >= 2:
                entities.append(word)
        
        return list(set(entities))
    
    def check_answer_consistency(self, question: str, answer: str) -> Dict[str, float]:
        """
        检查答案与文档的一致性
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            一致性得分字典
        """
        scores = {
            'factual_accuracy': 0.0,
            'information_coverage': 0.0,
            'hallucination_risk': 0.0,
            'overall_consistency': 0.0
        }
        
        if not self.documents:
            return scores
        
        # 1. 事实准确性（答案中的信息是否可以在文档中找到）
        scores['factual_accuracy'] = self._check_factual_accuracy(answer)
        
        # 2. 信息覆盖度（答案是否充分利用了文档信息）
        scores['information_coverage'] = self._check_information_coverage(question, answer)
        
        # 3. 幻觉风险（答案中是否包含文档中没有的信息）
        scores['hallucination_risk'] = self._assess_hallucination_risk(answer)
        
        # 4. 综合一致性
        weights = {
            'factual_accuracy': 0.4,
            'information_coverage': 0.3,
            'hallucination_risk': 0.3
        }
        
        # 注意：hallucination_risk 是负面指标，需要反转
        consistency_score = (
            scores['factual_accuracy'] * weights['factual_accuracy'] +
            scores['information_coverage'] * weights['information_coverage'] +
            (1 - scores['hallucination_risk']) * weights['hallucination_risk']
        )
        
        scores['overall_consistency'] = consistency_score
        
        return scores
    
    def _check_factual_accuracy(self, answer: str) -> float:
        """检查事实准确性"""
        # 提取答案中的关键信息片段
        answer_sentences = self._split_sentences(answer)
        
        if not answer_sentences:
            return 0.0
        
        # 检查每个句子是否有文档支持
        supported_sentences = 0
        for sentence in answer_sentences:
            if self._is_sentence_supported(sentence):
                supported_sentences += 1
        
        return supported_sentences / len(answer_sentences)
    
    def _is_sentence_supported(self, sentence: str) -> bool:
        """检查句子是否有文档支持"""
        # 简化版：检查句子中的关键词是否在文档中出现
        keywords = jieba.lcut(sentence)
        keywords = [w for w in keywords if len(w) >= 2]
        
        if not keywords:
            return True  # 没有实质内容的句子不扣分
        
        # 检查关键词覆盖率
        found_keywords = 0
        for keyword in keywords:
            for doc in self.documents:
                if keyword in doc:
                    found_keywords += 1
                    break
        
        # 如果超过50%的关键词在文档中找到，认为有支持
        return found_keywords / len(keywords) > 0.5
    
    def _check_information_coverage(self, question: str, answer: str) -> float:
        """检查信息覆盖度"""
        # 找出与问题最相关的文档片段
        relevant_info = self._find_relevant_information(question)
        
        if not relevant_info:
            return 0.5
        
        # 检查答案是否包含了相关信息
        coverage_score = 0.0
        for info in relevant_info:
            if self._is_info_included(info, answer):
                coverage_score += 1
        
        return min(1.0, coverage_score / max(len(relevant_info), 1))
    
    def _find_relevant_information(self, question: str) -> List[str]:
        """找出与问题相关的信息"""
        relevant_info = []
        
        # 使用语义相似度找出相关句子
        if self.model and hasattr(self, 'doc_embeddings'):
            question_embedding = self.model.encode([question])
            similarities = np.dot(self.doc_embeddings, question_embedding.T).flatten()
            
            # 找出相似度最高的前5个句子
            top_indices = np.argsort(similarities)[-5:]
            for idx in top_indices:
                if similarities[idx] > 0.5:  # 相似度阈值
                    relevant_info.append(self.doc_sentences[idx])
        
        return relevant_info
    
    def _is_info_included(self, info: str, answer: str) -> bool:
        """检查信息是否包含在答案中"""
        # 简化版：检查关键词重叠
        info_keywords = set(w for w in jieba.lcut(info) if len(w) >= 2)
        answer_keywords = set(w for w in jieba.lcut(answer) if len(w) >= 2)
        
        if not info_keywords:
            return True
        
        overlap = len(info_keywords & answer_keywords)
        return overlap / len(info_keywords) > 0.3
    
    def _assess_hallucination_risk(self, answer: str) -> float:
        """评估幻觉风险"""
        # 提取答案中的具体信息（数字、技术术语等）
        specific_info = self._extract_specific_information(answer)
        
        if not specific_info:
            return 0.0  # 没有具体信息，幻觉风险低
        
        # 检查这些信息是否在文档中出现
        unsupported_info = 0
        for info in specific_info:
            found = False
            for doc in self.documents:
                if info in doc:
                    found = True
                    break
            if not found:
                unsupported_info += 1
        
        return unsupported_info / len(specific_info)
    
    def _extract_specific_information(self, text: str) -> List[str]:
        """提取具体信息"""
        specific_info = []
        
        # 提取数字和单位
        numbers = re.findall(r'\d+[\.\d]*\s*[%℃度倍个]', text)
        specific_info.extend(numbers)
        
        # 提取技术术语
        tech_terms = re.findall(r'[A-Z][A-Za-z]*[-_]?[A-Za-z]*', text)
        specific_info.extend(tech_terms)
        
        # 提取引号中的内容
        quoted = re.findall(r'["""](.*?)["""]', text)
        specific_info.extend(quoted)
        
        return list(set(specific_info))
    
    def compute_qa_validity_score(self, question: str, answer: str) -> Dict[str, float]:
        """
        计算问答对的有效性综合得分
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            有效性得分字典
        """
        # 检查问题相关性
        question_scores = self.check_question_relevance(question)
        
        # 检查答案一致性
        answer_scores = self.check_answer_consistency(question, answer)
        
        # 计算综合有效性得分
        validity_score = (
            question_scores['overall_relevance'] * 0.4 +
            answer_scores['overall_consistency'] * 0.6
        )
        
        return {
            'question_relevance': question_scores['overall_relevance'],
            'answer_consistency': answer_scores['overall_consistency'],
            'validity_score': validity_score,
            'question_details': question_scores,
            'answer_details': answer_scores
        }
    
    def batch_check(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        批量检查问答对
        
        Args:
            qa_pairs: 问答对列表
            
        Returns:
            包含有效性得分的问答对列表
        """
        results = []
        
        for qa_pair in qa_pairs:
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            
            # 计算有效性得分
            validity_scores = self.compute_qa_validity_score(question, answer)
            
            # 添加到结果
            result = qa_pair.copy()
            result['validity_scores'] = validity_scores
            results.append(result)
        
        return results