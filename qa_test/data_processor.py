"""
数据处理模块
实现数据预处理、清洗和关键词提取
"""

import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import jieba
import jieba.analyse
from collections import Counter


@dataclass
class ProcessedQAPair:
    """处理后的问答对"""
    question: str
    answer: str
    original_question: str
    original_answer: str
    metadata: Optional[Dict] = None


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化数据处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger('DataProcessor')
        
        # 处理配置
        self.remove_html = self.config.get('remove_html', True)
        self.remove_urls = self.config.get('remove_urls', True)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        self.remove_special_chars = self.config.get('remove_special_chars', False)
        self.min_length = self.config.get('min_length', 5)
        self.max_length = self.config.get('max_length', 5000)
        
        # 初始化jieba
        jieba.initialize()
    
    def process_qa_pair(self, qa_pair) -> ProcessedQAPair:
        """
        处理单个问答对
        
        Args:
            qa_pair: 问答对对象
            
        Returns:
            处理后的问答对
        """
        # 保存原始文本
        original_question = qa_pair.question
        original_answer = qa_pair.answer
        
        # 处理问题和答案
        processed_question = self.clean_text(qa_pair.question)
        processed_answer = self.clean_text(qa_pair.answer)
        
        return ProcessedQAPair(
            question=processed_question,
            answer=processed_answer,
            original_question=original_question,
            original_answer=original_answer,
            metadata=getattr(qa_pair, 'metadata', None)
        )
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 去除HTML标签
        if self.remove_html:
            text = self._remove_html_tags(text)
        
        # 去除URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # 规范化空白字符
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # 去除特殊字符
        if self.remove_special_chars:
            text = self._remove_special_characters(text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def _remove_html_tags(self, text: str) -> str:
        """去除HTML标签"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def _remove_urls(self, text: str) -> str:
        """去除URLs"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符"""
        # 替换多个空白字符为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除中文字符间的空格
        text = re.sub(r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])', r'\1\2', text)
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """去除特殊字符"""
        # 保留中文、英文、数字和基本标点
        pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）《》\s]'
        return re.sub(pattern, '', text)
    
    def extract_keywords(self, text: str, topk: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            topk: 返回前K个关键词
            
        Returns:
            关键词列表
        """
        # 使用jieba的TF-IDF算法提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=topk, withWeight=False)
        return keywords
    
    def extract_key_phrases(self, text: str, max_length: int = 4) -> List[str]:
        """
        提取关键短语
        
        Args:
            text: 输入文本
            max_length: 短语最大长度
            
        Returns:
            关键短语列表
        """
        # 分词
        words = list(jieba.cut(text))
        
        # 提取n-gram短语
        phrases = []
        for n in range(2, min(max_length + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = ''.join(words[i:i+n])
                # 过滤包含停用词的短语
                if self._is_valid_phrase(phrase):
                    phrases.append(phrase)
        
        # 统计频率并返回高频短语
        phrase_counter = Counter(phrases)
        top_phrases = [phrase for phrase, _ in phrase_counter.most_common(20)]
        
        return top_phrases
    
    def _is_valid_phrase(self, phrase: str) -> bool:
        """判断是否为有效短语"""
        # 简单的停用词列表
        stopwords = {'的', '了', '是', '在', '和', '就', '都', '而', '及', '与', '或', '但', '因为', '所以'}
        
        # 检查是否包含停用词
        for word in stopwords:
            if word in phrase:
                return False
        
        # 长度检查
        if len(phrase) < 2 or len(phrase) > 20:
            return False
        
        return True
    
    def load_qa_pairs(self, file_path: str) -> List[Dict]:
        """
        从文件加载问答对
        
        Args:
            file_path: 文件路径
            
        Returns:
            问答对列表
        """
        qa_pairs = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 支持多种格式
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if 'question' in item and 'answer' in item:
                            qa_pairs.append(item)
                        elif 'q' in item and 'a' in item:
                            qa_pairs.append({
                                'question': item['q'],
                                'answer': item['a'],
                                'id': item.get('id'),
                                'metadata': item.get('metadata', {})
                            })
            elif isinstance(data, dict):
                # 可能是包含问答对列表的字典
                if 'qa_pairs' in data:
                    qa_pairs = data['qa_pairs']
                elif 'data' in data:
                    qa_pairs = data['data']
                    
            self.logger.info(f"Loaded {len(qa_pairs)} QA pairs from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading QA pairs from {file_path}: {e}")
            raise
        
        return qa_pairs
    
    def save_qa_pairs(self, qa_pairs: List[Dict], file_path: str, format: str = 'json'):
        """
        保存问答对到文件
        
        Args:
            qa_pairs: 问答对列表
            file_path: 输出文件路径
            format: 输出格式 ('json', 'jsonl')
        """
        try:
            if format == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            elif format == 'jsonl':
                with open(file_path, 'w', encoding='utf-8') as f:
                    for qa_pair in qa_pairs:
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            self.logger.info(f"Saved {len(qa_pairs)} QA pairs to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving QA pairs to {file_path}: {e}")
            raise
    
    def validate_qa_pair(self, qa_pair: Dict) -> Tuple[bool, List[str]]:
        """
        验证问答对的有效性
        
        Args:
            qa_pair: 问答对字典
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必要字段
        if 'question' not in qa_pair:
            errors.append("Missing 'question' field")
        if 'answer' not in qa_pair:
            errors.append("Missing 'answer' field")
        
        if errors:
            return False, errors
        
        question = qa_pair['question']
        answer = qa_pair['answer']
        
        # 检查类型
        if not isinstance(question, str):
            errors.append("Question must be a string")
        if not isinstance(answer, str):
            errors.append("Answer must be a string")
        
        # 检查长度
        if len(question) < self.min_length:
            errors.append(f"Question too short (min: {self.min_length})")
        if len(answer) < self.min_length:
            errors.append(f"Answer too short (min: {self.min_length})")
        if len(question) > self.max_length:
            errors.append(f"Question too long (max: {self.max_length})")
        if len(answer) > self.max_length:
            errors.append(f"Answer too long (max: {self.max_length})")
        
        # 检查内容质量
        if question.strip() == answer.strip():
            errors.append("Question and answer are identical")
        
        # 检查是否包含有效内容
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z]', question):
            errors.append("Question contains no valid text")
        if not re.search(r'[\u4e00-\u9fa5a-zA-Z]', answer):
            errors.append("Answer contains no valid text")
        
        return len(errors) == 0, errors
    
    def batch_validate(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        批量验证问答对
        
        Args:
            qa_pairs: 问答对列表
            
        Returns:
            (有效问答对列表, 无效问答对列表)
        """
        valid_pairs = []
        invalid_pairs = []
        
        for i, qa_pair in enumerate(qa_pairs):
            is_valid, errors = self.validate_qa_pair(qa_pair)
            
            if is_valid:
                valid_pairs.append(qa_pair)
            else:
                invalid_pair = qa_pair.copy()
                invalid_pair['validation_errors'] = errors
                invalid_pair['original_index'] = i
                invalid_pairs.append(invalid_pair)
        
        self.logger.info(f"Validation complete: {len(valid_pairs)} valid, {len(invalid_pairs)} invalid")
        
        return valid_pairs, invalid_pairs
    
    def deduplicate_qa_pairs(self, qa_pairs: List[Dict], threshold: float = 0.95) -> List[Dict]:
        """
        去除重复的问答对
        
        Args:
            qa_pairs: 问答对列表
            threshold: 相似度阈值
            
        Returns:
            去重后的问答对列表
        """
        if not qa_pairs:
            return []
        
        # 简单的基于文本完全匹配的去重
        seen = set()
        unique_pairs = []
        
        for qa_pair in qa_pairs:
            # 创建唯一标识
            key = f"{qa_pair['question']}:::{qa_pair['answer']}"
            
            if key not in seen:
                seen.add(key)
                unique_pairs.append(qa_pair)
        
        removed_count = len(qa_pairs) - len(unique_pairs)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate QA pairs")
        
        return unique_pairs
    
    def split_dataset(self, qa_pairs: List[Dict], train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """
        分割数据集
        
        Args:
            qa_pairs: 问答对列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            包含train, val, test的字典
        """
        import random
        
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # 打乱数据
        shuffled_pairs = qa_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # 计算分割点
        n = len(shuffled_pairs)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # 分割数据
        splits = {
            'train': shuffled_pairs[:train_end],
            'val': shuffled_pairs[train_end:val_end],
            'test': shuffled_pairs[val_end:]
        }
        
        self.logger.info(f"Dataset split: train={len(splits['train'])}, "
                        f"val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits