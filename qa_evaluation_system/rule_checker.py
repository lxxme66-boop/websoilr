"""
Rule-based checking module for QA evaluation
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from collections import Counter
import jieba
from zhon import hanzi
import string

logger = logging.getLogger(__name__)


class RuleChecker:
    """Rule-based checker for QA pairs"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules_config = config.get('rules', {})
        self.thresholds = config.get('thresholds', {})
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Load sensitive words if available
        self.sensitive_words = self._load_sensitive_words()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.patterns = {
            'url': re.compile(r'https?://[^\s]+'),
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'code_block': re.compile(r'```[\s\S]*?```|`[^`]+`'),
            'html_tag': re.compile(r'<[^>]+>'),
            'repeated_chars': re.compile(r'(.)\1{4,}'),  # 5+ repeated chars
            'repeated_words': re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE),
            'special_chars': re.compile(r'[^\w\s\u4e00-\u9fff' + re.escape(hanzi.punctuation) + ']'),
            'phone_number': re.compile(r'\b1[3-9]\d{9}\b'),
            'id_card': re.compile(r'\b\d{17}[\dXx]\b'),
        }
    
    def _load_sensitive_words(self) -> set:
        """Load sensitive words list"""
        # In production, load from file
        # For now, return a basic set
        return {
            '违法', '暴力', '色情', '赌博', '诈骗',
            '反动', '邪教', '恐怖', '毒品', '枪支'
        }
    
    def check_qa_pair(self, question: str, answer: str) -> Dict[str, Any]:
        """Check a QA pair against all rules"""
        results = {
            'question_checks': self._check_text(question, 'question'),
            'answer_checks': self._check_text(answer, 'answer'),
            'qa_checks': self._check_qa_relation(question, answer),
            'overall_passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Aggregate issues
        for check_type in ['question_checks', 'answer_checks', 'qa_checks']:
            checks = results[check_type]
            if not checks['passed']:
                results['overall_passed'] = False
                results['issues'].extend(checks['issues'])
            results['warnings'].extend(checks.get('warnings', []))
        
        # Calculate rule-based score
        results['rule_score'] = self._calculate_rule_score(results)
        
        return results
    
    def _check_text(self, text: str, text_type: str) -> Dict[str, Any]:
        """Check individual text (question or answer)"""
        checks = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'details': {}
        }
        
        # Length check
        length_check = self._check_length(text, text_type)
        checks['details']['length'] = length_check
        if not length_check['passed']:
            checks['passed'] = False
            checks['issues'].append(length_check['message'])
        
        # Format checks
        if self.rules_config.get('check_special_chars', True):
            special_check = self._check_special_chars(text)
            checks['details']['special_chars'] = special_check
            if special_check['count'] > 5:
                checks['warnings'].append(f"包含{special_check['count']}个特殊字符")
        
        if self.rules_config.get('check_urls', True):
            url_check = self._check_urls(text)
            checks['details']['urls'] = url_check
            if url_check['count'] > 0 and text_type == 'question':
                checks['warnings'].append("问题中包含URL")
        
        if self.rules_config.get('check_emails', True):
            email_check = self._check_emails(text)
            checks['details']['emails'] = email_check
            if email_check['count'] > 0:
                checks['warnings'].append("包含邮箱地址")
        
        if self.rules_config.get('check_code_blocks', True):
            code_check = self._check_code_blocks(text)
            checks['details']['code_blocks'] = code_check
        
        # Content checks
        if self.rules_config.get('check_repetition', True):
            repetition_check = self._check_repetition(text)
            checks['details']['repetition'] = repetition_check
            if repetition_check['has_repetition']:
                checks['warnings'].append("存在重复内容")
        
        if self.rules_config.get('check_sensitive_info', True):
            sensitive_check = self._check_sensitive_info(text)
            checks['details']['sensitive_info'] = sensitive_check
            if sensitive_check['has_sensitive']:
                checks['passed'] = False
                checks['issues'].append("包含敏感信息")
        
        if self.rules_config.get('check_profanity', True):
            profanity_check = self._check_profanity(text)
            checks['details']['profanity'] = profanity_check
            if profanity_check['has_profanity']:
                checks['passed'] = False
                checks['issues'].append("包含敏感词汇")
        
        # Language check
        language_check = self._check_language(text)
        checks['details']['language'] = language_check
        
        # Quality indicators
        quality_check = self._check_text_quality(text)
        checks['details']['quality'] = quality_check
        
        return checks
    
    def _check_length(self, text: str, text_type: str) -> Dict[str, Any]:
        """Check text length"""
        length = len(text)
        min_key = f"min_{text_type}_length"
        max_key = f"max_{text_type}_length"
        
        min_length = self.thresholds.get(min_key, 10)
        max_length = self.thresholds.get(max_key, 2000)
        
        result = {
            'length': length,
            'passed': min_length <= length <= max_length,
            'message': ''
        }
        
        if length < min_length:
            result['message'] = f"{text_type}太短（{length}字符，最少需要{min_length}字符）"
        elif length > max_length:
            result['message'] = f"{text_type}太长（{length}字符，最多允许{max_length}字符）"
        
        return result
    
    def _check_special_chars(self, text: str) -> Dict[str, Any]:
        """Check for special characters"""
        matches = self.patterns['special_chars'].findall(text)
        return {
            'count': len(matches),
            'chars': list(set(matches))[:10]  # First 10 unique special chars
        }
    
    def _check_urls(self, text: str) -> Dict[str, Any]:
        """Check for URLs"""
        matches = self.patterns['url'].findall(text)
        return {
            'count': len(matches),
            'urls': matches
        }
    
    def _check_emails(self, text: str) -> Dict[str, Any]:
        """Check for email addresses"""
        matches = self.patterns['email'].findall(text)
        return {
            'count': len(matches),
            'emails': ['***@***' for _ in matches]  # Masked for privacy
        }
    
    def _check_code_blocks(self, text: str) -> Dict[str, Any]:
        """Check for code blocks"""
        matches = self.patterns['code_block'].findall(text)
        return {
            'count': len(matches),
            'total_length': sum(len(m) for m in matches)
        }
    
    def _check_repetition(self, text: str) -> Dict[str, Any]:
        """Check for repetitive content"""
        # Check repeated characters
        char_matches = self.patterns['repeated_chars'].findall(text)
        
        # Check repeated words
        word_matches = self.patterns['repeated_words'].findall(text)
        
        # Check repeated sentences
        sentences = re.split(r'[。！？.!?]', text)
        sentence_counter = Counter(s.strip() for s in sentences if s.strip())
        repeated_sentences = [s for s, count in sentence_counter.items() if count > 1]
        
        return {
            'has_repetition': bool(char_matches or word_matches or repeated_sentences),
            'repeated_chars': char_matches[:5],
            'repeated_words': word_matches[:5],
            'repeated_sentences': repeated_sentences[:3]
        }
    
    def _check_sensitive_info(self, text: str) -> Dict[str, Any]:
        """Check for sensitive information"""
        has_phone = bool(self.patterns['phone_number'].search(text))
        has_id = bool(self.patterns['id_card'].search(text))
        
        return {
            'has_sensitive': has_phone or has_id,
            'types': [t for t, has in [('phone', has_phone), ('id_card', has_id)] if has]
        }
    
    def _check_profanity(self, text: str) -> Dict[str, Any]:
        """Check for profanity and sensitive words"""
        found_words = []
        text_lower = text.lower()
        
        for word in self.sensitive_words:
            if word in text_lower:
                found_words.append(word)
        
        return {
            'has_profanity': bool(found_words),
            'count': len(found_words)
        }
    
    def _check_language(self, text: str) -> Dict[str, Any]:
        """Check language composition"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text)
        
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0
        
        primary_lang = self.rules_config.get('primary_language', 'zh')
        is_expected_lang = (primary_lang == 'zh' and chinese_ratio > 0.5) or \
                          (primary_lang == 'en' and english_ratio > 0.5)
        
        return {
            'chinese_ratio': chinese_ratio,
            'english_ratio': english_ratio,
            'is_expected_language': is_expected_lang,
            'is_mixed': chinese_ratio > 0.1 and english_ratio > 0.1
        }
    
    def _check_text_quality(self, text: str) -> Dict[str, Any]:
        """Check text quality indicators"""
        # Check for question marks in questions
        has_question_mark = '？' in text or '?' in text
        
        # Check for proper punctuation
        has_punctuation = bool(re.search(r'[。！？，、；：.!?,;:]', text))
        
        # Check for complete sentences
        ends_properly = text.rstrip()[-1:] in '。！？.!?' if text else False
        
        # Calculate entropy (character diversity)
        char_counts = Counter(text)
        total_chars = len(text)
        entropy = -sum((count/total_chars) * (count/total_chars) 
                      for count in char_counts.values()) if total_chars > 0 else 0
        
        return {
            'has_question_mark': has_question_mark,
            'has_punctuation': has_punctuation,
            'ends_properly': ends_properly,
            'character_diversity': entropy
        }
    
    def _check_qa_relation(self, question: str, answer: str) -> Dict[str, Any]:
        """Check relationship between question and answer"""
        checks = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'details': {}
        }
        
        # Check if answer is longer than question (usually expected)
        if len(answer) < len(question) * 0.5:
            checks['warnings'].append("答案似乎过于简短")
        
        # Check for common words (basic relevance)
        q_words = set(jieba.cut(question))
        a_words = set(jieba.cut(answer))
        common_words = q_words.intersection(a_words)
        
        # Remove stopwords
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '什么', '吗', '呢', '吧', '啊'}
        common_words = common_words - stopwords
        
        overlap_ratio = len(common_words) / len(q_words) if len(q_words) > 0 else 0
        
        checks['details']['word_overlap'] = {
            'common_words': list(common_words)[:10],
            'overlap_ratio': overlap_ratio
        }
        
        if overlap_ratio < 0.1:
            checks['warnings'].append("问题和答案之间词汇重叠度较低")
        
        # Check if answer actually answers the question
        question_words = ['什么', '为什么', '怎么', '如何', '哪', '谁', '何时', '多少']
        is_question = any(word in question for word in question_words)
        
        if is_question and len(answer) < 20:
            checks['warnings'].append("答案可能过于简单")
        
        return checks
    
    def _calculate_rule_score(self, check_results: Dict[str, Any]) -> float:
        """Calculate overall rule-based score"""
        if not check_results['overall_passed']:
            return 0.3  # Failed critical checks
        
        # Start with perfect score
        score = 1.0
        
        # Deduct for issues and warnings
        issue_count = len(check_results['issues'])
        warning_count = len(check_results['warnings'])
        
        score -= issue_count * 0.2
        score -= warning_count * 0.05
        
        # Factor in quality indicators
        q_quality = check_results['question_checks']['details'].get('quality', {})
        a_quality = check_results['answer_checks']['details'].get('quality', {})
        
        if not q_quality.get('has_punctuation', True):
            score -= 0.05
        if not a_quality.get('ends_properly', True):
            score -= 0.05
        
        # Factor in language check
        q_lang = check_results['question_checks']['details'].get('language', {})
        a_lang = check_results['answer_checks']['details'].get('language', {})
        
        if not q_lang.get('is_expected_language', True):
            score -= 0.1
        if not a_lang.get('is_expected_language', True):
            score -= 0.1
        
        return max(0.0, min(1.0, score))