"""
Utility functions for QA evaluation system
"""

import os
import json
import yaml
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import jieba
from collections import Counter


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_config.get('file', 'logs/qa_evaluation.log')),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    config = replace_env_vars(config)
    
    return config


def replace_env_vars(config: Any) -> Any:
    """Recursively replace environment variables in config"""
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.environ.get(env_var, config)
    else:
        return config


def load_qa_pairs(file_path: str) -> List[Dict[str, str]]:
    """Load QA pairs from various file formats"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.suffix == '.jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif file_path.suffix in ['.csv', '.tsv']:
        sep = '\t' if file_path.suffix == '.tsv' else ','
        df = pd.read_csv(file_path, sep=sep)
        data = df.to_dict('records')
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Standardize format
    qa_pairs = []
    for item in data:
        if 'question' in item and 'answer' in item:
            qa_pairs.append({
                'question': str(item['question']).strip(),
                'answer': str(item['answer']).strip(),
                'id': item.get('id', None),
                'metadata': {k: v for k, v in item.items() if k not in ['question', 'answer', 'id']}
            })
    
    return qa_pairs


def save_results(results: List[Dict[str, Any]], output_path: str, format: str = 'json') -> None:
    """Save evaluation results to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    elif format == 'csv':
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    elif format == 'excel':
        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {format}")


def create_cache_key(data: Any) -> str:
    """Create a cache key from data"""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()


class Cache:
    """Simple file-based cache"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
    
    def clear(self) -> None:
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Chinese punctuation
    text = re.sub(r'[^\w\s\u4e00-\u9fff。，、；：？！""''（）《》【】…—·]', '', text)
    
    return text.strip()


def calculate_text_stats(text: str) -> Dict[str, Any]:
    """Calculate various text statistics"""
    # Basic stats
    char_count = len(text)
    word_count = len(list(jieba.cut(text)))
    
    # Sentence count (simple heuristic)
    sentence_endings = ['。', '！', '？', '.', '!', '?']
    sentence_count = sum(1 for char in text if char in sentence_endings)
    sentence_count = max(sentence_count, 1)
    
    # Average word length
    words = list(jieba.cut(text))
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Vocabulary diversity
    unique_words = len(set(words))
    vocab_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Language detection
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    is_chinese = chinese_chars > english_chars
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'vocab_diversity': vocab_diversity,
        'is_chinese': is_chinese,
        'chinese_ratio': chinese_chars / char_count if char_count > 0 else 0
    }


def batch_process(items: List[Any], process_func: callable, batch_size: int = 32, 
                  show_progress: bool = True, desc: str = "Processing") -> List[Any]:
    """Process items in batches"""
    results = []
    
    if show_progress:
        pbar = tqdm(total=len(items), desc=desc)
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
        
        if show_progress:
            pbar.update(len(batch))
    
    if show_progress:
        pbar.close()
    
    return results


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to [0, 1] range"""
    return max(min_val, min(score, max_val))


def weighted_average(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculate weighted average of scores"""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(scores.get(key, 0) * weight for key, weight in weights.items())
    return weighted_sum / total_weight


def get_quality_level(score: float, thresholds: Dict[str, float]) -> str:
    """Get quality level based on score and thresholds"""
    if score >= thresholds.get('excellent', 0.9):
        return '优秀'
    elif score >= thresholds.get('good', 0.75):
        return '良好'
    elif score >= thresholds.get('medium', 0.6):
        return '中等'
    elif score >= thresholds.get('poor', 0.4):
        return '较差'
    else:
        return '差'


def export_top_qa_pairs(qa_pairs: List[Dict], scores: List[Dict], 
                       output_path: str, threshold: float = 0.7,
                       top_percentage: float = 0.3) -> None:
    """Export top quality QA pairs"""
    # Combine QA pairs with scores
    qa_with_scores = []
    for qa, score in zip(qa_pairs, scores):
        qa_with_score = qa.copy()
        qa_with_score['evaluation'] = score
        qa_with_scores.append(qa_with_score)
    
    # Sort by overall score
    qa_with_scores.sort(key=lambda x: x['evaluation']['overall_score'], reverse=True)
    
    # Filter by threshold and percentage
    filtered_qa = [qa for qa in qa_with_scores if qa['evaluation']['overall_score'] >= threshold]
    
    top_count = int(len(filtered_qa) * top_percentage)
    top_qa = filtered_qa[:top_count]
    
    # Save to file
    save_results(top_qa, output_path, format='json')
    
    return len(top_qa)


def create_evaluation_report(results: List[Dict], output_path: str) -> Dict[str, Any]:
    """Create comprehensive evaluation report"""
    # Calculate statistics
    scores = [r['overall_score'] for r in results]
    
    report = {
        'summary': {
            'total_qa_pairs': len(results),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        },
        'quality_distribution': {},
        'dimension_analysis': {},
        'common_issues': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Quality distribution
    quality_counts = Counter([r['quality_level'] for r in results])
    report['quality_distribution'] = dict(quality_counts)
    
    # Dimension analysis
    dimensions = ['question_quality', 'answer_quality', 'qa_relevance']
    for dim in dimensions:
        dim_scores = [r['detailed_scores'].get(dim, 0) for r in results]
        report['dimension_analysis'][dim] = {
            'average': np.mean(dim_scores),
            'median': np.median(dim_scores),
            'std': np.std(dim_scores)
        }
    
    # Common issues
    all_issues = []
    for r in results:
        all_issues.extend(r.get('issues', []))
    issue_counts = Counter(all_issues)
    report['common_issues'] = dict(issue_counts.most_common(10))
    
    # Save report
    save_results(report, output_path, format='json')
    
    return report