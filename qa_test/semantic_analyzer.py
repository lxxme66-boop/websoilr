"""
语义分析模块
使用预训练模型计算语义相似度
"""

import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import jieba


class SemanticAnalyzer:
    """语义分析器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化语义分析器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger('SemanticAnalyzer')
        
        # 模型配置
        self.model_name = self.config.get(
            'model_name', 
            'paraphrase-multilingual-MiniLM-L12-v2'  # 支持中文的多语言模型
        )
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self._load_model()
        
        # 缓存
        self.cache = {}
        self.cache_enabled = self.config.get('cache_enabled', True)
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            self.logger.info(f"Loading semantic model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # 降级到CPU
            if self.device == 'cuda':
                self.logger.info("Retrying with CPU...")
                self.device = 'cpu'
                self.model = SentenceTransformer(self.model_name, device='cpu')
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数 (0-1)
        """
        # 检查缓存
        cache_key = f"{text1}::{text2}"
        if self.cache_enabled and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # 编码文本
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            
            # 计算余弦相似度
            similarity = cosine_similarity(
                embeddings[0].cpu().numpy().reshape(1, -1),
                embeddings[1].cpu().numpy().reshape(1, -1)
            )[0][0]
            
            # 确保在0-1范围内
            similarity = float(max(0, min(1, similarity)))
            
            # 缓存结果
            if self.cache_enabled:
                self.cache[cache_key] = similarity
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.5  # 返回中等相似度作为默认值
    
    def compute_batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        批量计算文本相似度
        
        Args:
            texts1: 第一组文本
            texts2: 第二组文本
            
        Returns:
            相似度分数列表
        """
        if len(texts1) != len(texts2):
            raise ValueError("Text lists must have the same length")
        
        try:
            # 批量编码
            embeddings1 = self.model.encode(texts1, convert_to_tensor=True, batch_size=32)
            embeddings2 = self.model.encode(texts2, convert_to_tensor=True, batch_size=32)
            
            # 计算相似度
            similarities = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                sim = cosine_similarity(
                    emb1.cpu().numpy().reshape(1, -1),
                    emb2.cpu().numpy().reshape(1, -1)
                )[0][0]
                similarities.append(float(max(0, min(1, sim))))
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Error in batch similarity computation: {e}")
            return [0.5] * len(texts1)
    
    def analyze_answer_relevance(self, question: str, answer: str) -> Dict[str, float]:
        """
        分析答案相关性的多个方面
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            相关性分析结果
        """
        results = {}
        
        # 1. 整体语义相似度
        results['overall_similarity'] = self.compute_similarity(question, answer)
        
        # 2. 分句相似度（找出答案中最相关的句子）
        answer_sentences = self._split_sentences(answer)
        if answer_sentences:
            sentence_similarities = []
            for sentence in answer_sentences:
                sim = self.compute_similarity(question, sentence)
                sentence_similarities.append(sim)
            
            results['max_sentence_similarity'] = max(sentence_similarities)
            results['avg_sentence_similarity'] = np.mean(sentence_similarities)
        else:
            results['max_sentence_similarity'] = results['overall_similarity']
            results['avg_sentence_similarity'] = results['overall_similarity']
        
        # 3. 关键短语相似度
        question_phrases = self._extract_key_phrases(question)
        answer_phrases = self._extract_key_phrases(answer)
        
        if question_phrases and answer_phrases:
            phrase_similarities = []
            for q_phrase in question_phrases[:5]:  # 限制数量避免计算过多
                max_sim = 0
                for a_phrase in answer_phrases[:10]:
                    sim = self.compute_similarity(q_phrase, a_phrase)
                    max_sim = max(max_sim, sim)
                phrase_similarities.append(max_sim)
            
            results['phrase_similarity'] = np.mean(phrase_similarities) if phrase_similarities else 0.5
        else:
            results['phrase_similarity'] = 0.5
        
        # 4. 主题一致性
        results['topic_consistency'] = self._compute_topic_consistency(question, answer)
        
        return results
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分割句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 中文句子分割
        sentences = []
        
        # 使用多个标点符号分割
        import re
        sentence_endings = r'[。！？；\n]'
        raw_sentences = re.split(sentence_endings, text)
        
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:  # 过滤太短的句子
                sentences.append(sentence)
        
        return sentences
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        提取关键短语
        
        Args:
            text: 输入文本
            
        Returns:
            关键短语列表
        """
        # 简单的关键短语提取
        phrases = []
        
        # 使用jieba进行分词
        words = list(jieba.cut(text))
        
        # 提取2-4词的短语
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                phrase = ''.join(words[i:i+n])
                if len(phrase) > 2:  # 过滤太短的短语
                    phrases.append(phrase)
        
        # 去重并限制数量
        phrases = list(set(phrases))[:20]
        
        return phrases
    
    def _compute_topic_consistency(self, question: str, answer: str) -> float:
        """
        计算主题一致性
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            主题一致性分数
        """
        try:
            # 获取问题和答案的向量表示
            q_embedding = self.model.encode(question, convert_to_tensor=True)
            a_embedding = self.model.encode(answer, convert_to_tensor=True)
            
            # 计算向量的方向一致性（不仅仅是相似度）
            # 使用点积除以向量长度
            dot_product = torch.dot(q_embedding, a_embedding)
            q_norm = torch.norm(q_embedding)
            a_norm = torch.norm(a_embedding)
            
            if q_norm > 0 and a_norm > 0:
                consistency = (dot_product / (q_norm * a_norm)).item()
                # 将结果映射到0-1范围
                consistency = (consistency + 1) / 2
            else:
                consistency = 0.5
            
            return float(consistency)
            
        except Exception as e:
            self.logger.error(f"Error computing topic consistency: {e}")
            return 0.5
    
    def find_similar_questions(self, question: str, question_bank: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        从问题库中找出相似问题
        
        Args:
            question: 查询问题
            question_bank: 问题库
            top_k: 返回前K个相似问题
            
        Returns:
            相似问题及其相似度列表
        """
        if not question_bank:
            return []
        
        try:
            # 编码查询问题
            query_embedding = self.model.encode(question, convert_to_tensor=True)
            
            # 编码问题库
            bank_embeddings = self.model.encode(question_bank, convert_to_tensor=True, batch_size=32)
            
            # 计算相似度
            similarities = cosine_similarity(
                query_embedding.cpu().numpy().reshape(1, -1),
                bank_embeddings.cpu().numpy()
            )[0]
            
            # 获取Top-K
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((question_bank[idx], float(similarities[idx])))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar questions: {e}")
            return []
    
    def compute_answer_coverage(self, question: str, answer: str) -> float:
        """
        计算答案对问题的覆盖程度
        
        Args:
            question: 问题
            answer: 答案
            
        Returns:
            覆盖度分数
        """
        # 将问题分解为多个方面
        question_aspects = self._extract_question_aspects(question)
        
        if not question_aspects:
            # 如果无法分解，使用整体相似度
            return self.compute_similarity(question, answer)
        
        # 计算每个方面的覆盖度
        aspect_scores = []
        for aspect in question_aspects:
            # 找出答案中最相关的部分
            answer_sentences = self._split_sentences(answer)
            if answer_sentences:
                max_sim = max(self.compute_similarity(aspect, sent) for sent in answer_sentences)
            else:
                max_sim = self.compute_similarity(aspect, answer)
            aspect_scores.append(max_sim)
        
        # 返回平均覆盖度
        return np.mean(aspect_scores) if aspect_scores else 0.5
    
    def _extract_question_aspects(self, question: str) -> List[str]:
        """
        提取问题的不同方面
        
        Args:
            question: 问题文本
            
        Returns:
            问题方面列表
        """
        aspects = []
        
        # 简单的规则：通过标点符号分割
        import re
        
        # 分割问题
        sub_questions = re.split(r'[，、]', question)
        for sub_q in sub_questions:
            sub_q = sub_q.strip()
            if len(sub_q) > 3:
                aspects.append(sub_q)
        
        # 如果没有找到子问题，返回原问题
        if not aspects:
            aspects = [question]
        
        return aspects