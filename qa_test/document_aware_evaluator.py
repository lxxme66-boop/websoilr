"""
文档感知评测器
评估问题从文档中提取的合理性
"""

import json
import logging
import os
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from llm_evaluator import LLMEvaluator
from nlp_metrics import NLPMetrics


@dataclass
class DocumentContext:
    """文档上下文"""
    document_id: str
    content: str
    metadata: Optional[Dict] = None


@dataclass
class DocumentAwareQAPair:
    """带文档上下文的问答对"""
    question: str
    answer: str
    source_document: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class ExtractionEvaluationResult:
    """提取合理性评估结果"""
    qa_pair: DocumentAwareQAPair
    document_context: Optional[DocumentContext]
    extraction_scores: Dict[str, float]
    reasonableness_score: float
    details: Dict[str, any]


class DocumentAwareEvaluator:
    """文档感知的问答评测器"""
    
    def __init__(self, config: Dict):
        """
        初始化文档感知评测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 初始化组件
        self.llm_evaluator = LLMEvaluator(config.get('llm', {}))
        self.nlp_metrics = NLPMetrics(config.get('nlp', {}))
        
        # 文档存储
        self.documents = {}
        
        # TF-IDF向量化器
        self.tfidf_vectorizer = None
        self.document_vectors = None
        
        # 初始化jieba
        jieba.initialize()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger('DocumentAwareEvaluator')
        logger.setLevel(logging.INFO)
        return logger
        
    def load_documents(self, document_path: Union[str, List[str]]) -> None:
        """
        加载源文档
        
        Args:
            document_path: 文档路径或路径列表
        """
        if isinstance(document_path, str):
            document_paths = [document_path]
        else:
            document_paths = document_path
            
        for path in document_paths:
            if path.endswith('.json'):
                self._load_json_documents(path)
            elif path.endswith('.txt'):
                self._load_text_documents(path)
            else:
                self.logger.warning(f"不支持的文档格式: {path}")
                
        # 构建TF-IDF向量
        self._build_tfidf_vectors()
        
        self.logger.info(f"加载了 {len(self.documents)} 个文档")
        
    def _load_json_documents(self, path: str) -> None:
        """加载JSON格式的文档"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for idx, doc in enumerate(data):
                    if isinstance(doc, dict):
                        doc_id = doc.get('id', f"doc_{idx}")
                        content = doc.get('content', '') or doc.get('text', '')
                        self.documents[doc_id] = DocumentContext(
                            document_id=doc_id,
                            content=content,
                            metadata=doc.get('metadata', {})
                        )
                    else:
                        doc_id = f"doc_{idx}"
                        self.documents[doc_id] = DocumentContext(
                            document_id=doc_id,
                            content=str(doc)
                        )
            elif isinstance(data, dict):
                for doc_id, content in data.items():
                    self.documents[doc_id] = DocumentContext(
                        document_id=doc_id,
                        content=str(content)
                    )
                    
        except Exception as e:
            self.logger.error(f"加载JSON文档失败: {path}, 错误: {e}")
            
    def _load_text_documents(self, path: str) -> None:
        """加载文本文档"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            doc_id = os.path.basename(path).split('.')[0]
            self.documents[doc_id] = DocumentContext(
                document_id=doc_id,
                content=content
            )
        except Exception as e:
            self.logger.error(f"加载文本文档失败: {path}, 错误: {e}")
            
    def _build_tfidf_vectors(self) -> None:
        """构建文档的TF-IDF向量"""
        if not self.documents:
            return
            
        # 准备文档内容
        doc_ids = list(self.documents.keys())
        doc_contents = [self.documents[doc_id].content for doc_id in doc_ids]
        
        # 创建TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=jieba.cut,
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # 计算TF-IDF向量
        self.document_vectors = self.tfidf_vectorizer.fit_transform(doc_contents)
        self.doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
    def find_source_document(self, qa_pair: DocumentAwareQAPair) -> Optional[DocumentContext]:
        """
        查找问答对的源文档
        
        Args:
            qa_pair: 问答对
            
        Returns:
            源文档上下文
        """
        # 如果已指定文档ID
        if qa_pair.document_id and qa_pair.document_id in self.documents:
            return self.documents[qa_pair.document_id]
            
        # 如果有源文档内容
        if qa_pair.source_document:
            # 使用TF-IDF相似度查找最相似的文档
            query_vector = self.tfidf_vectorizer.transform([qa_pair.source_document])
            similarities = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # 找到最相似的文档
            max_idx = np.argmax(similarities)
            if similarities[max_idx] > 0.5:  # 相似度阈值
                doc_id = list(self.doc_id_to_index.keys())[max_idx]
                return self.documents[doc_id]
                
        # 使用问题和答案查找相关文档
        combined_text = f"{qa_pair.question} {qa_pair.answer}"
        query_vector = self.tfidf_vectorizer.transform([combined_text])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # 找到最相似的文档
        max_idx = np.argmax(similarities)
        if similarities[max_idx] > 0.3:  # 较低的阈值
            doc_id = list(self.doc_id_to_index.keys())[max_idx]
            return self.documents[doc_id]
            
        return None
        
    def evaluate_extraction_reasonableness(
        self, 
        qa_pair: DocumentAwareQAPair,
        document_context: Optional[DocumentContext] = None
    ) -> ExtractionEvaluationResult:
        """
        评估问题从文档中提取的合理性
        
        Args:
            qa_pair: 问答对
            document_context: 文档上下文（如果未提供，将自动查找）
            
        Returns:
            评估结果
        """
        # 查找源文档
        if document_context is None:
            document_context = self.find_source_document(qa_pair)
            
        extraction_scores = {}
        details = {}
        
        if document_context is None:
            # 没有找到源文档
            extraction_scores['document_found'] = 0.0
            extraction_scores['content_relevance'] = 0.0
            extraction_scores['answer_grounding'] = 0.0
            extraction_scores['question_specificity'] = 0.0
            reasonableness_score = 0.0
            
            details['error'] = '未找到源文档'
            
        else:
            # 评估各个维度
            extraction_scores['document_found'] = 1.0
            
            # 1. 内容相关性 - 问题和答案与文档的相关性
            content_relevance = self._evaluate_content_relevance(
                qa_pair, document_context
            )
            extraction_scores['content_relevance'] = content_relevance
            
            # 2. 答案依据性 - 答案是否有文档支撑
            answer_grounding = self._evaluate_answer_grounding(
                qa_pair.answer, document_context.content
            )
            extraction_scores['answer_grounding'] = answer_grounding
            
            # 3. 问题具体性 - 问题是否针对文档内容
            question_specificity = self._evaluate_question_specificity(
                qa_pair.question, document_context.content
            )
            extraction_scores['question_specificity'] = question_specificity
            
            # 4. 信息完整性 - 答案是否完整地回答了问题
            information_completeness = self._evaluate_information_completeness(
                qa_pair, document_context
            )
            extraction_scores['information_completeness'] = information_completeness
            
            # 5. 使用LLM评估提取合理性
            if self.config.get('use_llm_evaluation', True):
                llm_score = self._llm_evaluate_extraction(qa_pair, document_context)
                extraction_scores['llm_evaluation'] = llm_score
                
            # 计算总体合理性分数
            weights = self.config.get('extraction_weights', {
                'content_relevance': 0.25,
                'answer_grounding': 0.30,
                'question_specificity': 0.20,
                'information_completeness': 0.15,
                'llm_evaluation': 0.10
            })
            
            reasonableness_score = sum(
                extraction_scores.get(key, 0) * weights.get(key, 0)
                for key in weights
            )
            
            # 添加详细信息
            details['document_id'] = document_context.document_id
            details['document_length'] = len(document_context.content)
            details['extraction_analysis'] = self._analyze_extraction(
                qa_pair, document_context
            )
            
        return ExtractionEvaluationResult(
            qa_pair=qa_pair,
            document_context=document_context,
            extraction_scores=extraction_scores,
            reasonableness_score=reasonableness_score,
            details=details
        )
        
    def _evaluate_content_relevance(
        self, 
        qa_pair: DocumentAwareQAPair,
        document_context: DocumentContext
    ) -> float:
        """评估内容相关性"""
        # 提取关键词
        qa_keywords = set(jieba.analyse.extract_tags(
            f"{qa_pair.question} {qa_pair.answer}", 
            topK=20
        ))
        doc_keywords = set(jieba.analyse.extract_tags(
            document_context.content, 
            topK=50
        ))
        
        # 计算关键词重叠
        overlap = len(qa_keywords & doc_keywords)
        if len(qa_keywords) == 0:
            return 0.0
            
        keyword_score = overlap / len(qa_keywords)
        
        # 计算TF-IDF相似度
        qa_text = f"{qa_pair.question} {qa_pair.answer}"
        texts = [qa_text, document_context.content]
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            similarity = 0.0
            
        # 综合评分
        return 0.6 * keyword_score + 0.4 * similarity
        
    def _evaluate_answer_grounding(self, answer: str, document_content: str) -> float:
        """评估答案在文档中的依据性"""
        # 将答案分句
        answer_sentences = re.split(r'[。！？；]', answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
        
        if not answer_sentences:
            return 0.0
            
        # 检查每个句子在文档中的支撑度
        grounded_sentences = 0
        
        for sentence in answer_sentences:
            # 提取句子关键词
            keywords = jieba.analyse.extract_tags(sentence, topK=5)
            
            # 检查关键词在文档中的出现
            keyword_found = sum(1 for kw in keywords if kw in document_content)
            
            if keyword_found >= len(keywords) * 0.6:  # 60%的关键词出现
                grounded_sentences += 1
                
        return grounded_sentences / len(answer_sentences)
        
    def _evaluate_question_specificity(self, question: str, document_content: str) -> float:
        """评估问题的具体性"""
        # 检查问题是否包含文档特定的实体或概念
        question_keywords = set(jieba.analyse.extract_tags(question, topK=10))
        doc_keywords = set(jieba.analyse.extract_tags(document_content, topK=30))
        
        # 特定关键词（在文档中出现但不是通用词汇）
        specific_keywords = question_keywords & doc_keywords
        
        if not question_keywords:
            return 0.0
            
        specificity_score = len(specific_keywords) / len(question_keywords)
        
        # 检查问题类型
        general_patterns = [
            r'^什么是',
            r'^如何',
            r'^为什么',
            r'^请[解释说明]',
            r'^[介绍描述]一下'
        ]
        
        is_general = any(re.match(pattern, question) for pattern in general_patterns)
        
        # 如果是通用问题模式，降低分数
        if is_general and specificity_score < 0.5:
            specificity_score *= 0.7
            
        return specificity_score
        
    def _evaluate_information_completeness(
        self,
        qa_pair: DocumentAwareQAPair,
        document_context: DocumentContext
    ) -> float:
        """评估信息完整性"""
        # 使用NLP指标评估答案的完整性
        completeness_score = self.nlp_metrics.calculate_answer_completeness(
            qa_pair.question,
            qa_pair.answer
        )
        
        # 检查答案长度是否合理
        answer_length = len(qa_pair.answer)
        if answer_length < 20:  # 答案太短
            completeness_score *= 0.7
        elif answer_length > 500:  # 答案可能过于冗长
            completeness_score *= 0.9
            
        return completeness_score
        
    def _llm_evaluate_extraction(
        self,
        qa_pair: DocumentAwareQAPair,
        document_context: DocumentContext
    ) -> float:
        """使用LLM评估提取合理性"""
        prompt = f"""请评估以下问答对从文档中提取的合理性。

文档内容（前500字）：
{document_context.content[:500]}...

问题：{qa_pair.question}
答案：{qa_pair.answer}

请从以下方面评估：
1. 问题是否与文档内容相关
2. 答案是否基于文档内容
3. 问题和答案是否合理地从文档中提取

请给出0-1的评分，其中1表示完全合理，0表示完全不合理。
只返回数字评分。"""
        
        try:
            score = self.llm_evaluator.evaluate_with_prompt(prompt)
            return float(score)
        except:
            return 0.5  # 默认中等分数
            
    def _analyze_extraction(
        self,
        qa_pair: DocumentAwareQAPair,
        document_context: DocumentContext
    ) -> Dict:
        """分析提取过程"""
        analysis = {}
        
        # 找到答案在文档中的位置
        answer_position = document_context.content.find(qa_pair.answer[:50])
        if answer_position != -1:
            analysis['answer_found_in_document'] = True
            analysis['answer_position'] = answer_position
        else:
            analysis['answer_found_in_document'] = False
            
        # 分析问题类型
        question_type = self._classify_question_type(qa_pair.question)
        analysis['question_type'] = question_type
        
        # 计算覆盖率
        qa_text = f"{qa_pair.question} {qa_pair.answer}"
        qa_keywords = set(jieba.analyse.extract_tags(qa_text, topK=20))
        doc_keywords = set(jieba.analyse.extract_tags(document_context.content, topK=100))
        
        coverage = len(qa_keywords & doc_keywords) / len(qa_keywords) if qa_keywords else 0
        analysis['keyword_coverage'] = coverage
        
        return analysis
        
    def _classify_question_type(self, question: str) -> str:
        """分类问题类型"""
        patterns = {
            '定义型': [r'什么是', r'.*是什么', r'.*的定义'],
            '原因型': [r'为什么', r'.*的原因', r'.*导致'],
            '方法型': [r'如何', r'怎么', r'.*的方法', r'.*步骤'],
            '比较型': [r'.*和.*的区别', r'.*与.*相比', r'.*不同'],
            '列举型': [r'有哪些', r'.*包括', r'列举'],
            '描述型': [r'描述', r'介绍', r'说明', r'解释']
        }
        
        for q_type, patterns_list in patterns.items():
            for pattern in patterns_list:
                if re.search(pattern, question):
                    return q_type
                    
        return '其他'
        
    def batch_evaluate(
        self,
        qa_pairs: List[DocumentAwareQAPair],
        progress_callback=None
    ) -> List[ExtractionEvaluationResult]:
        """
        批量评估问答对
        
        Args:
            qa_pairs: 问答对列表
            progress_callback: 进度回调函数
            
        Returns:
            评估结果列表
        """
        results = []
        
        for idx, qa_pair in enumerate(qa_pairs):
            try:
                result = self.evaluate_extraction_reasonableness(qa_pair)
                results.append(result)
                
                if progress_callback:
                    progress_callback(idx + 1, len(qa_pairs))
                    
            except Exception as e:
                self.logger.error(f"评估问答对失败: {e}")
                # 创建失败结果
                results.append(ExtractionEvaluationResult(
                    qa_pair=qa_pair,
                    document_context=None,
                    extraction_scores={'error': 0.0},
                    reasonableness_score=0.0,
                    details={'error': str(e)}
                ))
                
        return results
        
    def generate_extraction_report(
        self,
        results: List[ExtractionEvaluationResult]
    ) -> Dict:
        """生成提取合理性报告"""
        report = {
            'total_qa_pairs': len(results),
            'average_reasonableness_score': np.mean([r.reasonableness_score for r in results]),
            'score_distribution': {},
            'dimension_analysis': defaultdict(list),
            'document_coverage': {},
            'issues': []
        }
        
        # 分数分布
        score_ranges = [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for min_score, max_score in score_ranges:
            count = sum(1 for r in results if min_score <= r.reasonableness_score < max_score)
            report['score_distribution'][f'{min_score}-{max_score}'] = count
            
        # 维度分析
        for result in results:
            for dimension, score in result.extraction_scores.items():
                report['dimension_analysis'][dimension].append(score)
                
        # 计算各维度平均分
        for dimension, scores in report['dimension_analysis'].items():
            report['dimension_analysis'][dimension] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores)
            }
            
        # 文档覆盖率
        doc_qa_count = defaultdict(int)
        for result in results:
            if result.document_context:
                doc_qa_count[result.document_context.document_id] += 1
                
        report['document_coverage'] = dict(doc_qa_count)
        
        # 识别问题
        for result in results:
            if result.reasonableness_score < 0.5:
                issue = {
                    'qa_id': result.qa_pair.metadata.get('id', 'unknown') if result.qa_pair.metadata else 'unknown',
                    'question': result.qa_pair.question[:50] + '...',
                    'score': result.reasonableness_score,
                    'main_issues': []
                }
                
                # 找出主要问题
                for dimension, score in result.extraction_scores.items():
                    if score < 0.5:
                        issue['main_issues'].append(dimension)
                        
                report['issues'].append(issue)
                
        return report