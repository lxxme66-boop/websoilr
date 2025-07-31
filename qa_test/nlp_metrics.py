"""
传统NLP指标计算模块
实现BLEU、ROUGE、困惑度等评测指标
"""

import re
import math
import logging
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jieba
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade


class NLPMetrics:
    """NLP指标计算器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化NLP指标计算器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger('NLPMetrics')
        
        # 初始化分词器
        self.language = self.config.get('language', 'chinese')
        if self.language == 'chinese':
            jieba.initialize()
        
        # 初始化ROUGE评分器
        self.rouge_types = self.config.get('rouge_types', ['rouge1', 'rouge2', 'rougeL'])
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def compute_metrics(self, question: str, answer: str) -> Dict[str, float]:
        """
        计算所有NLP指标
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            指标字典
        """
        metrics = {}
        
        # 1. BLEU分数
        try:
            bleu_scores = self._compute_bleu(question, answer)
            metrics.update(bleu_scores)
        except Exception as e:
            self.logger.error(f"BLEU computation failed: {e}")
            metrics['bleu'] = 0.0
        
        # 2. ROUGE分数
        try:
            rouge_scores = self._compute_rouge(question, answer)
            metrics.update(rouge_scores)
        except Exception as e:
            self.logger.error(f"ROUGE computation failed: {e}")
            metrics['rouge1'] = 0.0
        
        # 3. 语言流畅度
        try:
            fluency_score = self._compute_fluency(answer)
            metrics['fluency'] = fluency_score
        except Exception as e:
            self.logger.error(f"Fluency computation failed: {e}")
            metrics['fluency'] = 0.5
        
        # 4. 词汇多样性
        try:
            diversity_score = self._compute_lexical_diversity(answer)
            metrics['lexical_diversity'] = diversity_score
        except Exception as e:
            self.logger.error(f"Diversity computation failed: {e}")
            metrics['lexical_diversity'] = 0.5
        
        # 5. 关键词覆盖率
        try:
            coverage_score = self._compute_keyword_coverage(question, answer)
            metrics['keyword_coverage'] = coverage_score
        except Exception as e:
            self.logger.error(f"Coverage computation failed: {e}")
            metrics['keyword_coverage'] = 0.5
        
        return metrics
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            
        Returns:
            词列表
        """
        if self.language == 'chinese':
            return list(jieba.cut(text))
        else:
            return nltk.word_tokenize(text.lower())
    
    def _compute_bleu(self, question: str, answer: str) -> Dict[str, float]:
        """
        计算BLEU分数
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            BLEU分数字典
        """
        # 分词
        reference = self._tokenize(question)
        candidate = self._tokenize(answer)
        
        # 使用平滑函数避免零分
        smoothing = SmoothingFunction()
        
        # 计算不同n-gram的BLEU分数
        bleu_scores = {}
        
        # BLEU-1到BLEU-4
        for n in range(1, 5):
            weights = [0] * 4
            weights[n-1] = 1
            score = sentence_bleu(
                [reference], 
                candidate, 
                weights=weights,
                smoothing_function=smoothing.method1
            )
            bleu_scores[f'bleu{n}'] = score
        
        # 综合BLEU分数
        bleu_scores['bleu'] = sentence_bleu(
            [reference], 
            candidate,
            smoothing_function=smoothing.method1
        )
        
        return bleu_scores
    
    def _compute_rouge(self, question: str, answer: str) -> Dict[str, float]:
        """
        计算ROUGE分数
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            ROUGE分数字典
        """
        # 对中文需要分词后重新组合
        if self.language == 'chinese':
            question_tokens = self._tokenize(question)
            answer_tokens = self._tokenize(answer)
            question_processed = ' '.join(question_tokens)
            answer_processed = ' '.join(answer_tokens)
        else:
            question_processed = question
            answer_processed = answer
        
        # 计算ROUGE分数
        scores = self.rouge_scorer_obj.score(question_processed, answer_processed)
        
        # 提取F1分数
        rouge_scores = {}
        for rouge_type in self.rouge_types:
            rouge_scores[rouge_type] = scores[rouge_type].fmeasure
        
        return rouge_scores
    
    def _compute_fluency(self, text: str) -> float:
        """
        计算语言流畅度
        
        Args:
            text: 文本
            
        Returns:
            流畅度分数 (0-1)
        """
        if self.language == 'chinese':
            # 中文流畅度评估
            return self._compute_chinese_fluency(text)
        else:
            # 英文使用Flesch Reading Ease
            try:
                # Flesch Reading Ease范围是0-100，转换到0-1
                score = flesch_reading_ease(text)
                # 将分数归一化到0-1范围
                normalized_score = max(0, min(100, score)) / 100
                return normalized_score
            except:
                return 0.5
    
    def _compute_chinese_fluency(self, text: str) -> float:
        """
        计算中文流畅度
        
        Args:
            text: 中文文本
            
        Returns:
            流畅度分数
        """
        fluency_score = 0.5  # 基础分
        
        # 1. 检查标点符号使用
        punctuation_pattern = r'[，。！？；：、]'
        punctuation_count = len(re.findall(punctuation_pattern, text))
        sentence_count = len(re.split(r'[。！？]', text))
        
        if sentence_count > 0:
            # 合理的标点符号密度
            punct_density = punctuation_count / len(text)
            if 0.05 <= punct_density <= 0.15:
                fluency_score += 0.1
        
        # 2. 句子长度分布
        sentences = re.split(r'[。！？]', text)
        sentences = [s for s in sentences if s.strip()]
        
        if sentences:
            sentence_lengths = [len(s) for s in sentences]
            avg_length = np.mean(sentence_lengths)
            
            # 理想的中文句子长度在15-30字之间
            if 15 <= avg_length <= 30:
                fluency_score += 0.2
            elif 10 <= avg_length <= 40:
                fluency_score += 0.1
        
        # 3. 词汇重复度（过度重复降低流畅度）
        words = self._tokenize(text)
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.5:
                fluency_score += 0.2
        
        return min(fluency_score, 1.0)
    
    def _compute_lexical_diversity(self, text: str) -> float:
        """
        计算词汇多样性
        
        Args:
            text: 文本
            
        Returns:
            多样性分数 (0-1)
        """
        words = self._tokenize(text)
        
        if not words:
            return 0.0
        
        # Type-Token Ratio (TTR)
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # 对于较长文本，使用MSTTR (Mean Segmental TTR)
        if len(words) > 50:
            window_size = 50
            msttr_scores = []
            
            for i in range(0, len(words) - window_size + 1):
                window = words[i:i + window_size]
                window_ttr = len(set(window)) / len(window)
                msttr_scores.append(window_ttr)
            
            if msttr_scores:
                diversity_score = np.mean(msttr_scores)
            else:
                diversity_score = ttr
        else:
            diversity_score = ttr
        
        return diversity_score
    
    def _compute_keyword_coverage(self, question: str, answer: str) -> float:
        """
        计算关键词覆盖率
        
        Args:
            question: 问题文本
            answer: 答案文本
            
        Returns:
            覆盖率分数 (0-1)
        """
        # 提取问题关键词
        question_words = self._tokenize(question)
        answer_words = self._tokenize(answer)
        
        # 过滤停用词
        question_keywords = self._filter_keywords(question_words)
        answer_keywords = set(answer_words)
        
        if not question_keywords:
            return 0.5  # 如果没有关键词，返回中等分数
        
        # 计算覆盖率
        covered = question_keywords & answer_keywords
        coverage = len(covered) / len(question_keywords)
        
        return coverage
    
    def _filter_keywords(self, words: List[str]) -> Set[str]:
        """
        过滤关键词（去除停用词）
        
        Args:
            words: 词列表
            
        Returns:
            关键词集合
        """
        # 简单的停用词列表
        if self.language == 'chinese':
            stopwords = {
                '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                '自己', '这', '那', '什么', '吗', '呢', '吧', '啊', '哦', '呀', '么'
            }
        else:
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which'
            }
        
        # 过滤停用词和短词
        keywords = {
            word for word in words 
            if word not in stopwords and len(word) > 1
        }
        
        return keywords
    
    def compute_perplexity(self, text: str, model=None) -> float:
        """
        计算困惑度（需要语言模型）
        
        Args:
            text: 文本
            model: 语言模型（可选）
            
        Returns:
            困惑度分数
        """
        # 这里提供一个简化的实现
        # 实际使用时应该使用预训练的语言模型
        
        if model is None:
            # 使用简单的n-gram模型估算
            words = self._tokenize(text)
            if len(words) < 2:
                return 1.0
            
            # 计算bigram概率
            bigram_counts = Counter(zip(words[:-1], words[1:]))
            unigram_counts = Counter(words)
            
            # 简化的困惑度计算
            log_prob = 0
            for i in range(1, len(words)):
                bigram = (words[i-1], words[i])
                bigram_count = bigram_counts.get(bigram, 0.1)  # 平滑
                unigram_count = unigram_counts.get(words[i-1], 1)
                
                prob = bigram_count / unigram_count
                log_prob += math.log(prob)
            
            perplexity = math.exp(-log_prob / (len(words) - 1))
            
            # 归一化到0-1范围（假设困惑度范围是1-1000）
            normalized = 1.0 / (1.0 + math.log(perplexity))
            return normalized
        else:
            # 使用提供的模型计算
            # 这里需要根据具体模型实现
            pass
    
    def compute_coherence(self, question: str, answer: str) -> float:
        """
        计算语义连贯性
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            连贯性分数
        """
        # 简化的连贯性评估
        question_words = set(self._tokenize(question))
        answer_words = self._tokenize(answer)
        
        # 计算答案中与问题相关词汇的分布
        relevant_positions = []
        for i, word in enumerate(answer_words):
            if word in question_words:
                relevant_positions.append(i)
        
        if not relevant_positions or len(answer_words) == 0:
            return 0.5
        
        # 相关词汇分布越均匀，连贯性越好
        avg_distance = len(answer_words) / (len(relevant_positions) + 1)
        actual_distances = []
        
        prev_pos = -1
        for pos in relevant_positions:
            actual_distances.append(pos - prev_pos)
            prev_pos = pos
        actual_distances.append(len(answer_words) - prev_pos - 1)
        
        # 计算分布均匀度
        if actual_distances:
            variance = np.var(actual_distances)
            # 方差越小，分布越均匀
            coherence = 1.0 / (1.0 + variance / avg_distance)
        else:
            coherence = 0.5
        
        return min(coherence, 1.0)