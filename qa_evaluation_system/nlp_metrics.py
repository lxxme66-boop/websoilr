"""
NLP metrics module for QA evaluation
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import jieba
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from bert_score import score as bert_score_func
import textstat
import nltk
from collections import Counter
import re

logger = logging.getLogger(__name__)


class NLPMetrics:
    """Calculate various NLP metrics for QA evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_config = config.get('nlp_models', {})
        
        # Initialize models
        self._init_models()
        
        # Download required NLTK data
        self._download_nltk_data()
    
    def _init_models(self):
        """Initialize NLP models"""
        device = self.nlp_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Sentence transformer for semantic similarity
        model_name = self.nlp_config.get('sentence_transformer', 
                                         'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        try:
            self.sentence_model = SentenceTransformer(model_name, device=device)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # BERT model for advanced metrics
        bert_model_name = self.nlp_config.get('bert_model', 'bert-base-chinese')
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
            self.bert_model.eval()
            logger.info(f"Loaded BERT model: {bert_model_name}")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                     use_stemmer=False)
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            logger.warning("Failed to download some NLTK data")
    
    def calculate_metrics(self, question: str, answer: str) -> Dict[str, Any]:
        """Calculate all NLP metrics for a QA pair"""
        metrics = {
            'semantic_similarity': self._calculate_semantic_similarity(question, answer),
            'lexical_overlap': self._calculate_lexical_overlap(question, answer),
            'readability': self._calculate_readability(answer),
            'coherence': self._calculate_coherence(answer),
            'informativeness': self._calculate_informativeness(answer),
            'answer_relevance': self._calculate_answer_relevance(question, answer)
        }
        
        # Calculate aggregate NLP score
        metrics['nlp_score'] = self._calculate_nlp_score(metrics)
        
        return metrics
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate semantic similarity between texts"""
        if self.sentence_model is None:
            return {'score': 0.5, 'method': 'fallback'}
        
        try:
            # Encode texts
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return {
                'score': float(similarity),
                'method': 'sentence_transformer'
            }
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return {'score': 0.5, 'method': 'error'}
    
    def _calculate_lexical_overlap(self, question: str, answer: str) -> Dict[str, Any]:
        """Calculate lexical overlap metrics"""
        # Tokenize
        q_tokens = list(jieba.cut(question.lower()))
        a_tokens = list(jieba.cut(answer.lower()))
        
        # Remove stopwords
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那'}
        q_tokens_clean = [t for t in q_tokens if t not in stopwords and len(t) > 1]
        a_tokens_clean = [t for t in a_tokens if t not in stopwords and len(t) > 1]
        
        # Calculate overlap
        q_set = set(q_tokens_clean)
        a_set = set(a_tokens_clean)
        overlap = q_set.intersection(a_set)
        
        # Metrics
        precision = len(overlap) / len(a_set) if len(a_set) > 0 else 0
        recall = len(overlap) / len(q_set) if len(q_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(question, answer)
        
        return {
            'token_overlap': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'overlap_tokens': list(overlap)[:10]
            },
            'rouge': {
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure
            }
        }
    
    def _calculate_readability(self, text: str) -> Dict[str, Any]:
        """Calculate readability metrics"""
        # For Chinese text, we need different metrics
        is_chinese = len(re.findall(r'[\u4e00-\u9fff]', text)) > len(text) * 0.5
        
        if is_chinese:
            # Chinese readability metrics
            chars = len(text)
            words = len(list(jieba.cut(text)))
            sentences = max(1, len(re.split(r'[。！？]', text)))
            
            # Average sentence length
            avg_sentence_length = chars / sentences
            
            # Average word length
            avg_word_length = chars / words if words > 0 else 0
            
            # Complexity score (simplified)
            complexity = min(1.0, avg_sentence_length / 50)  # Normalize to 0-1
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'complexity_score': complexity,
                'is_chinese': True
            }
        else:
            # English readability metrics
            try:
                flesch_ease = textstat.flesch_reading_ease(text)
                gunning_fog = textstat.gunning_fog(text)
                
                # Normalize to 0-1 scale
                flesch_norm = max(0, min(1, flesch_ease / 100))
                fog_norm = max(0, min(1, 1 - (gunning_fog - 6) / 14))  # 6-20 scale
                
                return {
                    'flesch_reading_ease': flesch_ease,
                    'gunning_fog': gunning_fog,
                    'normalized_score': (flesch_norm + fog_norm) / 2,
                    'is_chinese': False
                }
            except:
                return {
                    'normalized_score': 0.5,
                    'is_chinese': False,
                    'error': 'calculation_failed'
                }
    
    def _calculate_coherence(self, text: str) -> Dict[str, float]:
        """Calculate text coherence metrics"""
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {'score': 1.0, 'sentence_count': len(sentences)}
        
        # Calculate sentence-to-sentence similarity
        if self.sentence_model is not None:
            try:
                embeddings = self.sentence_model.encode(sentences)
                
                # Calculate adjacent sentence similarities
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities) if similarities else 0.5
                
                return {
                    'score': float(avg_similarity),
                    'sentence_count': len(sentences),
                    'method': 'embedding_similarity'
                }
            except:
                pass
        
        # Fallback: lexical cohesion
        cohesion_scores = []
        for i in range(len(sentences) - 1):
            tokens1 = set(jieba.cut(sentences[i]))
            tokens2 = set(jieba.cut(sentences[i+1]))
            overlap = len(tokens1.intersection(tokens2))
            total = len(tokens1.union(tokens2))
            score = overlap / total if total > 0 else 0
            cohesion_scores.append(score)
        
        avg_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.5
        
        return {
            'score': float(avg_cohesion),
            'sentence_count': len(sentences),
            'method': 'lexical_cohesion'
        }
    
    def _calculate_informativeness(self, text: str) -> Dict[str, Any]:
        """Calculate informativeness metrics"""
        # Tokenize
        tokens = list(jieba.cut(text))
        
        # Vocabulary richness
        unique_tokens = set(tokens)
        vocabulary_richness = len(unique_tokens) / len(tokens) if tokens else 0
        
        # Information density (non-stopword ratio)
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己'}
        content_tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        information_density = len(content_tokens) / len(tokens) if tokens else 0
        
        # Named entities and numbers (simple detection)
        numbers = len(re.findall(r'\d+', text))
        
        # Technical terms (heuristic: longer words, English terms)
        technical_terms = [t for t in unique_tokens if len(t) > 3 or re.match(r'[a-zA-Z]+', t)]
        
        # Calculate score
        informativeness_score = (
            vocabulary_richness * 0.3 +
            information_density * 0.4 +
            min(1.0, numbers / 10) * 0.15 +
            min(1.0, len(technical_terms) / 20) * 0.15
        )
        
        return {
            'score': informativeness_score,
            'vocabulary_richness': vocabulary_richness,
            'information_density': information_density,
            'number_count': numbers,
            'technical_term_count': len(technical_terms)
        }
    
    def _calculate_answer_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """Calculate answer relevance to question"""
        # Semantic relevance
        semantic_sim = self._calculate_semantic_similarity(question, answer)
        
        # Question type detection
        question_types = {
            'what': ['什么', '何', 'what'],
            'why': ['为什么', '为何', 'why'],
            'how': ['怎么', '如何', 'how'],
            'when': ['什么时候', '何时', 'when'],
            'where': ['哪里', '何处', 'where'],
            'who': ['谁', 'who'],
            'which': ['哪个', '哪些', 'which'],
            'yesno': ['吗', '是否', '是不是', '有没有']
        }
        
        detected_type = 'unknown'
        for qtype, keywords in question_types.items():
            if any(kw in question.lower() for kw in keywords):
                detected_type = qtype
                break
        
        # Check if answer addresses the question type
        type_addressed = 1.0
        if detected_type == 'why' and not any(kw in answer for kw in ['因为', '由于', '原因']):
            type_addressed = 0.7
        elif detected_type == 'how' and not any(kw in answer for kw in ['步骤', '方法', '首先', '然后']):
            type_addressed = 0.7
        elif detected_type == 'yesno' and len(answer) > 100:
            type_addressed = 0.8  # Yes/no questions should have concise answers
        
        # BERT-based relevance (if available)
        bert_relevance = 0.5
        if self.bert_model is not None:
            try:
                bert_relevance = self._calculate_bert_relevance(question, answer)
            except:
                pass
        
        # Combine scores
        relevance_score = (
            semantic_sim['score'] * 0.4 +
            type_addressed * 0.3 +
            bert_relevance * 0.3
        )
        
        return {
            'score': relevance_score,
            'semantic_similarity': semantic_sim['score'],
            'question_type': detected_type,
            'type_addressed': type_addressed,
            'bert_relevance': bert_relevance
        }
    
    def _calculate_bert_relevance(self, question: str, answer: str) -> float:
        """Calculate relevance using BERT"""
        if self.bert_tokenizer is None or self.bert_model is None:
            return 0.5
        
        try:
            # Tokenize and encode
            inputs = self.bert_tokenizer(
                question, answer,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state
                
                # Use CLS token embedding
                cls_embedding = embeddings[:, 0, :].cpu().numpy()
                
                # Simple relevance score based on CLS embedding magnitude
                relevance = np.linalg.norm(cls_embedding) / 100  # Normalize
                relevance = min(1.0, max(0.0, relevance))
                
            return float(relevance)
        except Exception as e:
            logger.error(f"Error in BERT relevance calculation: {e}")
            return 0.5
    
    def _calculate_nlp_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate aggregate NLP score"""
        # Extract scores
        semantic_score = metrics['semantic_similarity']['score']
        lexical_f1 = metrics['lexical_overlap']['token_overlap']['f1']
        rouge_l = metrics['lexical_overlap']['rouge']['rougeL']
        
        readability = metrics['readability'].get('normalized_score', 
                                                metrics['readability'].get('complexity_score', 0.5))
        coherence = metrics['coherence']['score']
        informativeness = metrics['informativeness']['score']
        relevance = metrics['answer_relevance']['score']
        
        # Weighted average
        nlp_score = (
            semantic_score * 0.2 +
            (lexical_f1 + rouge_l) / 2 * 0.15 +
            readability * 0.1 +
            coherence * 0.15 +
            informativeness * 0.15 +
            relevance * 0.25
        )
        
        return float(nlp_score)
    
    def batch_calculate_metrics(self, qa_pairs: List[Tuple[str, str]], 
                               batch_size: int = 32) -> List[Dict[str, Any]]:
        """Calculate metrics for multiple QA pairs"""
        results = []
        
        for i in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[i:i + batch_size]
            
            for question, answer in batch:
                metrics = self.calculate_metrics(question, answer)
                results.append(metrics)
        
        return results