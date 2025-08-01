"""
基于文档的问答对评测器
加载文档作为知识库，评测问答对是否与文档内容一致
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from collections import Counter
import re


@dataclass
class Document:
    """文档数据结构"""
    filename: str
    content: str
    sentences: List[str]
    embeddings: Optional[np.ndarray] = None


@dataclass
class QAPair:
    """问答对数据结构"""
    question: str
    answer: str
    id: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class DocumentEvaluationResult:
    """基于文档的评测结果"""
    qa_pair: QAPair
    relevance_score: float  # 答案与文档的相关性
    accuracy_score: float   # 答案的准确性
    coverage_score: float   # 答案覆盖的信息完整度
    consistency_score: float  # 答案与文档的一致性
    total_score: float
    supporting_sentences: List[Tuple[str, float]]  # 支持句子及其相似度
    details: Dict[str, any]


class DocumentBasedEvaluator:
    """基于文档的问答对评测器"""
    
    def __init__(self, document_dir: str, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        """
        初始化评测器
        
        Args:
            document_dir: 文档目录路径
            model_name: 句子嵌入模型名称
        """
        self.logger = self._setup_logger()
        self.document_dir = document_dir
        self.documents = []
        self.sentence_model = SentenceTransformer(model_name)
        
        # 加载文档
        self._load_documents()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DocumentBasedEvaluator')
        logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def _load_documents(self):
        """加载所有文档"""
        self.logger.info(f"Loading documents from {self.document_dir}")
        
        # 支持多种文档格式
        supported_extensions = ['.txt', '.md']
        
        for root, dirs, files in os.walk(self.document_dir):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    filepath = os.path.join(root, file)
                    self._load_single_document(filepath)
        
        self.logger.info(f"Loaded {len(self.documents)} documents")
        
        # 为所有文档生成句子嵌入
        self._generate_document_embeddings()
    
    def _load_single_document(self, filepath: str):
        """加载单个文档"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 将文档分割成句子
            sentences = self._split_into_sentences(content)
            
            doc = Document(
                filename=os.path.basename(filepath),
                content=content,
                sentences=sentences
            )
            
            self.documents.append(doc)
            self.logger.info(f"Loaded document: {doc.filename} ({len(sentences)} sentences)")
            
        except Exception as e:
            self.logger.error(f"Error loading document {filepath}: {e}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 使用中英文句子分割
        sentences = re.split(r'[。！？!?\n]+', text)
        # 过滤空句子并去除首尾空白
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _generate_document_embeddings(self):
        """为所有文档生成句子嵌入"""
        self.logger.info("Generating sentence embeddings for all documents")
        
        for doc in self.documents:
            if doc.sentences:
                doc.embeddings = self.sentence_model.encode(doc.sentences)
                self.logger.debug(f"Generated embeddings for {doc.filename}: shape {doc.embeddings.shape}")
    
    def evaluate_qa_pair(self, qa_pair: QAPair) -> DocumentEvaluationResult:
        """
        评测单个问答对
        
        Args:
            qa_pair: 问答对
            
        Returns:
            评测结果
        """
        self.logger.debug(f"Evaluating QA pair: {qa_pair.question}")
        
        # 1. 找到与问题最相关的文档句子
        question_embedding = self.sentence_model.encode([qa_pair.question])
        relevant_sentences = self._find_relevant_sentences(question_embedding, top_k=5)
        
        # 2. 评估答案与相关句子的一致性
        answer_embedding = self.sentence_model.encode([qa_pair.answer])
        
        # 计算各项得分
        relevance_score = self._calculate_relevance_score(answer_embedding, relevant_sentences)
        accuracy_score = self._calculate_accuracy_score(qa_pair.answer, relevant_sentences)
        coverage_score = self._calculate_coverage_score(qa_pair.answer, relevant_sentences)
        consistency_score = self._calculate_consistency_score(qa_pair.answer, relevant_sentences)
        
        # 计算总分
        total_score = (
            relevance_score * 0.3 +
            accuracy_score * 0.3 +
            coverage_score * 0.2 +
            consistency_score * 0.2
        )
        
        # 构建详细信息
        details = {
            'question_relevant_sentences': relevant_sentences[:3],
            'answer_length': len(qa_pair.answer),
            'key_terms_matched': self._extract_key_terms_match(qa_pair.answer, relevant_sentences)
        }
        
        return DocumentEvaluationResult(
            qa_pair=qa_pair,
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            coverage_score=coverage_score,
            consistency_score=consistency_score,
            total_score=total_score,
            supporting_sentences=relevant_sentences[:3],
            details=details
        )
    
    def _find_relevant_sentences(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """找到与查询最相关的句子"""
        all_sentences = []
        all_similarities = []
        
        for doc in self.documents:
            if doc.embeddings is not None:
                similarities = cosine_similarity(query_embedding, doc.embeddings)[0]
                for i, (sentence, sim) in enumerate(zip(doc.sentences, similarities)):
                    all_sentences.append((sentence, sim, doc.filename))
                    all_similarities.append(sim)
        
        # 按相似度排序
        sorted_indices = np.argsort(all_similarities)[::-1][:top_k]
        
        relevant_sentences = []
        for idx in sorted_indices:
            sentence, sim, filename = all_sentences[idx]
            relevant_sentences.append((sentence, float(sim)))
        
        return relevant_sentences
    
    def _calculate_relevance_score(self, answer_embedding: np.ndarray, 
                                   relevant_sentences: List[Tuple[str, float]]) -> float:
        """计算答案与相关句子的相关性得分"""
        if not relevant_sentences:
            return 0.0
        
        # 获取最相关句子的嵌入
        sentences = [sent for sent, _ in relevant_sentences]
        sentence_embeddings = self.sentence_model.encode(sentences)
        
        # 计算答案与相关句子的平均相似度
        similarities = cosine_similarity(answer_embedding, sentence_embeddings)[0]
        
        # 使用加权平均，权重为原始相似度
        weights = [sim for _, sim in relevant_sentences]
        weighted_sim = np.average(similarities, weights=weights)
        
        return float(weighted_sim)
    
    def _calculate_accuracy_score(self, answer: str, 
                                  relevant_sentences: List[Tuple[str, float]]) -> float:
        """计算答案的准确性得分"""
        if not relevant_sentences:
            return 0.0
        
        # 提取答案和相关句子的关键词
        answer_keywords = set(jieba.cut(answer))
        
        total_score = 0.0
        for sentence, weight in relevant_sentences:
            sentence_keywords = set(jieba.cut(sentence))
            
            # 计算关键词重叠度
            overlap = len(answer_keywords & sentence_keywords)
            total_keywords = len(answer_keywords | sentence_keywords)
            
            if total_keywords > 0:
                score = overlap / total_keywords
                total_score += score * weight
        
        # 归一化
        total_weight = sum(w for _, w in relevant_sentences)
        if total_weight > 0:
            total_score /= total_weight
        
        return total_score
    
    def _calculate_coverage_score(self, answer: str, 
                                  relevant_sentences: List[Tuple[str, float]]) -> float:
        """计算答案的覆盖度得分"""
        if not relevant_sentences:
            return 0.0
        
        # 提取所有相关句子的关键信息
        all_keywords = set()
        for sentence, _ in relevant_sentences:
            keywords = set(jieba.cut(sentence))
            all_keywords.update(keywords)
        
        # 计算答案覆盖了多少关键信息
        answer_keywords = set(jieba.cut(answer))
        covered = len(answer_keywords & all_keywords)
        
        if len(all_keywords) > 0:
            coverage = covered / len(all_keywords)
        else:
            coverage = 0.0
        
        # 考虑答案长度的合理性
        answer_length = len(answer)
        if answer_length < 10:  # 答案太短
            coverage *= 0.5
        elif answer_length > 500:  # 答案太长
            coverage *= 0.8
        
        return coverage
    
    def _calculate_consistency_score(self, answer: str, 
                                     relevant_sentences: List[Tuple[str, float]]) -> float:
        """计算答案与文档的一致性得分"""
        if not relevant_sentences:
            return 0.0
        
        # 检查答案中是否包含与文档矛盾的信息
        # 这里使用简单的否定词检测
        negation_words = ['不', '没有', '否', '无', '非', '未']
        
        answer_has_negation = any(word in answer for word in negation_words)
        
        consistency_scores = []
        for sentence, weight in relevant_sentences:
            sentence_has_negation = any(word in sentence for word in negation_words)
            
            # 如果答案和句子的否定性不一致，降低一致性分数
            if answer_has_negation != sentence_has_negation:
                consistency_scores.append(0.5 * weight)
            else:
                consistency_scores.append(1.0 * weight)
        
        # 计算加权平均
        total_weight = sum(w for _, w in relevant_sentences)
        if total_weight > 0:
            avg_consistency = sum(consistency_scores) / total_weight
        else:
            avg_consistency = 1.0
        
        return avg_consistency
    
    def _extract_key_terms_match(self, answer: str, 
                                 relevant_sentences: List[Tuple[str, float]]) -> Dict[str, int]:
        """提取答案与相关句子的关键词匹配情况"""
        # 提取答案的关键词
        answer_words = list(jieba.cut(answer))
        answer_counter = Counter(answer_words)
        
        # 提取相关句子的关键词
        sentence_words = []
        for sentence, _ in relevant_sentences:
            sentence_words.extend(jieba.cut(sentence))
        sentence_counter = Counter(sentence_words)
        
        # 找出共同的关键词
        common_terms = {}
        for term in answer_counter:
            if term in sentence_counter and len(term) > 1:  # 忽略单字词
                common_terms[term] = min(answer_counter[term], sentence_counter[term])
        
        return common_terms
    
    def evaluate_qa_file(self, qa_file_path: str, output_path: str, top_k: int = 100):
        """
        评测整个问答对文件
        
        Args:
            qa_file_path: 问答对JSON文件路径
            output_path: 输出文件路径
            top_k: 选择Top-K个最佳问答对
        """
        self.logger.info(f"Evaluating QA file: {qa_file_path}")
        
        # 加载问答对
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        if isinstance(qa_data, dict) and 'qa_pairs' in qa_data:
            qa_pairs_data = qa_data['qa_pairs']
        else:
            qa_pairs_data = qa_data
        
        # 转换为QAPair对象
        qa_pairs = []
        for i, item in enumerate(qa_pairs_data):
            qa_pair = QAPair(
                question=item['question'],
                answer=item['answer'],
                id=item.get('id', f'qa_{i}'),
                metadata=item.get('metadata', {})
            )
            qa_pairs.append(qa_pair)
        
        self.logger.info(f"Loaded {len(qa_pairs)} QA pairs")
        
        # 评测所有问答对
        results = []
        for qa_pair in qa_pairs:
            result = self.evaluate_qa_pair(qa_pair)
            results.append(result)
            
            self.logger.info(
                f"Evaluated {qa_pair.id}: "
                f"relevance={result.relevance_score:.3f}, "
                f"accuracy={result.accuracy_score:.3f}, "
                f"coverage={result.coverage_score:.3f}, "
                f"consistency={result.consistency_score:.3f}, "
                f"total={result.total_score:.3f}"
            )
        
        # 按总分排序
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        # 选择Top-K
        selected_results = results[:top_k]
        
        # 准备输出数据
        output_data = {
            'total_evaluated': len(results),
            'selected_count': len(selected_results),
            'evaluation_summary': {
                'avg_relevance': np.mean([r.relevance_score for r in selected_results]),
                'avg_accuracy': np.mean([r.accuracy_score for r in selected_results]),
                'avg_coverage': np.mean([r.coverage_score for r in selected_results]),
                'avg_consistency': np.mean([r.consistency_score for r in selected_results]),
                'avg_total': np.mean([r.total_score for r in selected_results])
            },
            'qa_pairs': []
        }
        
        for result in selected_results:
            qa_data = {
                'id': result.qa_pair.id,
                'question': result.qa_pair.question,
                'answer': result.qa_pair.answer,
                'scores': {
                    'relevance': result.relevance_score,
                    'accuracy': result.accuracy_score,
                    'coverage': result.coverage_score,
                    'consistency': result.consistency_score,
                    'total': result.total_score
                },
                'supporting_evidence': [
                    {'sentence': sent, 'similarity': sim} 
                    for sent, sim in result.supporting_sentences
                ],
                'metadata': result.qa_pair.metadata
            }
            output_data['qa_pairs'].append(qa_data)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(selected_results)} best QA pairs to {output_path}")
        
        # 生成评测报告
        self._generate_evaluation_report(results, output_path)
    
    def _generate_evaluation_report(self, results: List[DocumentEvaluationResult], base_output_path: str):
        """生成详细的评测报告"""
        report_path = base_output_path.replace('.json', '_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 问答对评测报告\n\n")
            f.write(f"## 评测概览\n\n")
            f.write(f"- 总评测数量: {len(results)}\n")
            f.write(f"- 文档数量: {len(self.documents)}\n")
            f.write(f"- 文档列表: {', '.join([doc.filename for doc in self.documents])}\n\n")
            
            # 分数分布
            f.write("## 分数分布\n\n")
            
            scores = [r.total_score for r in results]
            f.write(f"- 最高分: {max(scores):.3f}\n")
            f.write(f"- 最低分: {min(scores):.3f}\n")
            f.write(f"- 平均分: {np.mean(scores):.3f}\n")
            f.write(f"- 标准差: {np.std(scores):.3f}\n\n")
            
            # 各维度平均分
            f.write("## 各维度平均分\n\n")
            f.write(f"- 相关性: {np.mean([r.relevance_score for r in results]):.3f}\n")
            f.write(f"- 准确性: {np.mean([r.accuracy_score for r in results]):.3f}\n")
            f.write(f"- 覆盖度: {np.mean([r.coverage_score for r in results]):.3f}\n")
            f.write(f"- 一致性: {np.mean([r.consistency_score for r in results]):.3f}\n\n")
            
            # Top 10 最佳问答对
            f.write("## Top 10 最佳问答对\n\n")
            for i, result in enumerate(results[:10]):
                f.write(f"### {i+1}. {result.qa_pair.question}\n\n")
                f.write(f"**答案**: {result.qa_pair.answer}\n\n")
                f.write(f"**得分**: {result.total_score:.3f} ")
                f.write(f"(相关性: {result.relevance_score:.3f}, ")
                f.write(f"准确性: {result.accuracy_score:.3f}, ")
                f.write(f"覆盖度: {result.coverage_score:.3f}, ")
                f.write(f"一致性: {result.consistency_score:.3f})\n\n")
                
                if result.supporting_sentences:
                    f.write("**支持句子**:\n")
                    for sent, sim in result.supporting_sentences[:2]:
                        f.write(f"- {sent} (相似度: {sim:.3f})\n")
                    f.write("\n")
            
            # Bottom 5 最差问答对
            f.write("## Bottom 5 最差问答对\n\n")
            for i, result in enumerate(results[-5:]):
                f.write(f"### {i+1}. {result.qa_pair.question}\n\n")
                f.write(f"**答案**: {result.qa_pair.answer}\n\n")
                f.write(f"**得分**: {result.total_score:.3f} ")
                f.write(f"(相关性: {result.relevance_score:.3f}, ")
                f.write(f"准确性: {result.accuracy_score:.3f}, ")
                f.write(f"覆盖度: {result.coverage_score:.3f}, ")
                f.write(f"一致性: {result.consistency_score:.3f})\n\n")
        
        self.logger.info(f"Generated evaluation report: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='基于文档的问答对评测')
    parser.add_argument('--documents', '-d', required=True, help='文档目录路径')
    parser.add_argument('--input', '-i', required=True, help='输入问答对JSON文件')
    parser.add_argument('--output', '-o', required=True, help='输出评测结果文件')
    parser.add_argument('--top-k', '-k', type=int, default=100, help='选择Top-K个最佳问答对')
    
    args = parser.parse_args()
    
    # 创建评测器
    evaluator = DocumentBasedEvaluator(args.documents)
    
    # 评测问答对
    evaluator.evaluate_qa_file(args.input, args.output, args.top_k)