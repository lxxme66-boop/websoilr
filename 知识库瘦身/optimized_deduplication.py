#!/usr/bin/env python3
"""
精准文档去重系统 - 优化版
用于检测和去除文档集合中的重复内容
"""

import jieba
import jieba.analyse
import time
import json
import logging
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import re
import hashlib
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


class DocumentDeduplicator:
    """文档去重主类"""
    
    # 内置中文停用词列表
    DEFAULT_STOPWORDS = {
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '着', '或', '一个', '没有',
        '这', '那', '在', '也', '有', '更', '让', '去', '说', '看', '啊', '呀', '吧',
        '但', '我', '你', '他', '她', '它', '我们', '你们', '他们', '自己', '什么', '怎么',
        '因为', '所以', '但是', '然后', '而且', '如果', '虽然', '尽管', '即使', '为了',
        '可以', '可能', '会', '能', '能够', '要', '应该', '必须', '得', '需要'
    }
    
    def __init__(self, 
                 fingerprint_threshold: float = 0.9,
                 minhash_threshold: float = 0.5,
                 tfidf_threshold: float = 0.7,
                 num_perm: int = 128,
                 n_jobs: int = -1):
        """
        初始化去重器
        
        Args:
            fingerprint_threshold: 指纹匹配阈值
            minhash_threshold: MinHash LSH阈值
            tfidf_threshold: TF-IDF相似度阈值
            num_perm: MinHash排列数
            n_jobs: 并行任务数，-1表示使用所有CPU核心
        """
        self.fingerprint_threshold = fingerprint_threshold
        self.minhash_threshold = minhash_threshold
        self.tfidf_threshold = tfidf_threshold
        self.num_perm = num_perm
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        
        self.logger = self._setup_logger()
        self._setup_matplotlib()
        self._setup_jieba()
        
    def _setup_logger(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger('DocumentDeduplicator')
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler("precision_deduplication.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_matplotlib(self):
        """配置matplotlib中文显示"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            self.logger.warning("无法设置中文字体，将使用默认设置")
    
    def _setup_jieba(self):
        """设置jieba分词器"""
        # 创建停用词文件
        stopwords_path = "stopwords.txt"
        if not os.path.exists(stopwords_path):
            with open(stopwords_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.DEFAULT_STOPWORDS))
            self.logger.info(f"已创建停用词文件，包含 {len(self.DEFAULT_STOPWORDS)} 个停用词")
        
        # 设置停用词
        try:
            jieba.analyse.set_stop_words(stopwords_path)
            self.logger.info("已加载停用词文件")
        except Exception as e:
            self.logger.error(f"加载停用词文件失败: {e}")
            self.logger.warning("将使用内置停用词列表")
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """文本预处理：清洗、标准化"""
        if not text:
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除邮箱
        text = re.sub(r'\S+@\S+', '', text)
        
        # 保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、；：""''（）\[\]【】]', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text: str, topK: int = 25) -> List[str]:
        """智能关键词提取"""
        if not text:
            return []
        
        # 预处理文本
        text = self.preprocess_text(text)
        
        # 根据文本长度选择算法
        if len(text) < 300:
            return self._extract_keywords_bm25(text, topK)
        else:
            return self._extract_keywords_tfidf(text, topK)
    
    def _extract_keywords_bm25(self, text: str, topK: int) -> List[str]:
        """使用BM25算法提取关键词"""
        try:
            # 分词并过滤
            words = [
                word for word in jieba.cut(text) 
                if len(word) > 1 and word not in self.DEFAULT_STOPWORDS
            ]
            
            if not words:
                return []
            
            # 创建BM25模型
            bm25 = BM25Okapi([words])
            
            # 计算每个词的BM25分数
            word_scores = {}
            for word in set(words):
                word_scores[word] = bm25.get_scores([word])[0]
            
            # 按分数排序
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:topK]]
            
        except Exception as e:
            self.logger.error(f"BM25关键词提取出错: {e}")
            return []
    
    def _extract_keywords_tfidf(self, text: str, topK: int) -> List[str]:
        """使用TF-IDF算法提取关键词"""
        try:
            return jieba.analyse.extract_tags(
                text,
                topK=topK,
                withWeight=False,
                allowPOS=('n', 'vn', 'v', 'eng', 'nz', 'an')
            )
        except Exception as e:
            self.logger.error(f"TF-IDF关键词提取出错: {e}")
            return []
    
    def load_documents(self, file_path: str) -> List[Dict]:
        """从JSON文件加载文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            self.logger.info(f"从 {file_path} 加载了 {len(documents)} 篇文档")
            
            # 并行计算文档指纹
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for doc in documents:
                    future = executor.submit(self._compute_fingerprint, doc)
                    futures.append((future, doc))
                
                for future, doc in futures:
                    doc['fingerprint'] = future.result()
            
            return documents
            
        except Exception as e:
            self.logger.error(f"加载JSON文件失败: {e}")
            return []
    
    def _compute_fingerprint(self, doc: Dict) -> str:
        """计算文档指纹"""
        doc_text = doc.get('Abstract', '') + doc.get('论文标题', '')
        return hashlib.md5(doc_text.encode('utf-8')).hexdigest()
    
    def deduplicate(self, documents: List[Dict]) -> List[Dict]:
        """执行三阶段去重"""
        self.logger.info("开始精准去重处理...")
        start_time = time.time()
        
        # 阶段1: 指纹去重
        duplicates_stage1, unique_docs = self._fingerprint_deduplication(documents)
        
        # 阶段2: MinHash LSH
        candidate_pairs = self._minhash_deduplication(unique_docs)
        
        # 阶段3: TF-IDF精确计算
        duplicates_stage3 = self._tfidf_deduplication(unique_docs, candidate_pairs)
        
        # 合并结果
        all_duplicates = duplicates_stage1 + duplicates_stage3
        
        # 记录时间
        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)
        self.logger.info(f"精准去重完成! 耗时: {int(mins)}分{int(secs)}秒")
        self.logger.info(f"总共发现 {len(all_duplicates)} 个重复文档")
        
        return all_duplicates
    
    def _fingerprint_deduplication(self, documents: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """阶段1: 基于指纹的快速去重"""
        self.logger.info(f"阶段1: 基于指纹的快速去重 (阈值: {self.fingerprint_threshold*100}%)")
        
        fingerprint_map = {}
        duplicates = []
        unique_docs = []
        duplicate_ids = set()
        
        for doc in documents:
            fp = doc['fingerprint']
            if fp in fingerprint_map:
                duplicates.append({
                    'duplicate_id': doc['doc_id'],
                    'representative_id': fingerprint_map[fp],
                    'similarity': 1.0,
                    'method': 'fingerprint'
                })
                duplicate_ids.add(doc['doc_id'])
            else:
                fingerprint_map[fp] = doc['doc_id']
        
        # 收集唯一文档
        for doc in documents:
            if doc['doc_id'] not in duplicate_ids:
                unique_docs.append(doc)
        
        self.logger.info(f"阶段1发现 {len(duplicates)} 个重复文档")
        return duplicates, unique_docs
    
    def _minhash_deduplication(self, documents: List[Dict]) -> Set[Tuple[str, str]]:
        """阶段2: MinHash LSH相似文档检测"""
        self.logger.info(f"阶段2: MinHash LSH 相似文档检测 (阈值: {self.minhash_threshold*100}%)")
        
        # 创建LSH
        lsh = MinHashLSH(threshold=self.minhash_threshold, num_perm=self.num_perm)
        minhashes = {}
        
        # 并行创建MinHash
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            for doc in documents:
                future = executor.submit(self._create_minhash, doc)
                futures[future] = doc['doc_id']
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="创建MinHash"):
                doc_id = futures[future]
                mh = future.result()
                if mh:
                    minhashes[doc_id] = mh
                    lsh.insert(doc_id, mh)
        
        # 查找相似文档对
        candidate_pairs = set()
        for doc_id, mh in tqdm(minhashes.items(), desc="查询相似文档"):
            similar_ids = lsh.query(mh)
            for similar_id in similar_ids:
                if similar_id != doc_id:
                    pair = tuple(sorted([doc_id, similar_id]))
                    candidate_pairs.add(pair)
        
        self.logger.info(f"阶段2发现 {len(candidate_pairs)} 个候选文档对")
        return candidate_pairs
    
    def _create_minhash(self, doc: Dict) -> Optional[MinHash]:
        """为单个文档创建MinHash"""
        try:
            text = doc.get('Abstract', '') + " " + doc.get('论文标题', '')
            keywords = self.extract_keywords(text, topK=30)
            
            if not keywords:
                return None
            
            mh = MinHash(num_perm=self.num_perm)
            for word in keywords:
                mh.update(word.encode('utf-8'))
            
            return mh
        except Exception as e:
            self.logger.error(f"创建MinHash失败: {e}")
            return None
    
    def _tfidf_deduplication(self, documents: List[Dict], candidate_pairs: Set[Tuple[str, str]]) -> List[Dict]:
        """阶段3: TF-IDF余弦相似度精确计算"""
        self.logger.info(f"阶段3: TF-IDF余弦相似度精确计算 (阈值: {self.tfidf_threshold*100}%)")
        
        if not candidate_pairs:
            return []
        
        # 准备文档文本和映射
        doc_id_to_idx = {}
        doc_texts = []
        
        for idx, doc in enumerate(documents):
            text = self.preprocess_text(doc.get('Abstract', '') + " " + doc.get('论文标题', ''))
            doc_texts.append(text)
            doc_id_to_idx[doc['doc_id']] = idx
        
        # 创建TF-IDF矩阵
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        
        # 批量计算相似度
        duplicates = []
        batch_size = 1000
        pairs_list = list(candidate_pairs)
        
        for i in tqdm(range(0, len(pairs_list), batch_size), desc="计算精确相似度"):
            batch = pairs_list[i:i+batch_size]
            
            for doc_id1, doc_id2 in batch:
                if doc_id1 not in doc_id_to_idx or doc_id2 not in doc_id_to_idx:
                    continue
                
                idx1 = doc_id_to_idx[doc_id1]
                idx2 = doc_id_to_idx[doc_id2]
                
                # 计算余弦相似度
                similarity = cosine_similarity(
                    tfidf_matrix[idx1:idx1+1], 
                    tfidf_matrix[idx2:idx2+1]
                )[0][0]
                
                if similarity >= self.tfidf_threshold:
                    representative_id = min(doc_id1, doc_id2)
                    duplicate_id = doc_id1 if doc_id1 != representative_id else doc_id2
                    
                    duplicates.append({
                        'duplicate_id': duplicate_id,
                        'representative_id': representative_id,
                        'similarity': float(similarity),
                        'method': 'tfidf'
                    })
        
        self.logger.info(f"阶段3发现 {len(duplicates)} 个重复文档")
        return duplicates
    
    def analyze_and_export(self, documents: List[Dict], duplicates: List[Dict]):
        """分析并导出结果"""
        if not duplicates:
            self.logger.warning("未发现重复文档")
            return
        
        # 创建文档ID到标题的映射
        id_to_title = {doc['doc_id']: doc.get('论文标题', '无标题') for doc in documents}
        
        # 导出详细重复信息
        self._export_duplicate_details(duplicates, id_to_title)
        
        # 可视化分析
        self._visualize_duplicates(duplicates)
        
        # 导出唯一文档
        self._export_unique_documents(documents, duplicates)
        
        # 显示高相似度文档
        self._show_high_similarity_pairs(duplicates, id_to_title)
    
    def _export_duplicate_details(self, duplicates: List[Dict], id_to_title: Dict[str, str]):
        """导出重复文档详细信息"""
        duplicate_info = []
        for dup in duplicates:
            duplicate_info.append({
                '重复文档ID': dup['duplicate_id'],
                '代表文档ID': dup['representative_id'],
                '相似度': f"{dup['similarity']*100:.1f}%",
                '方法': dup['method'],
                '重复文档标题': id_to_title.get(dup['duplicate_id'], '未知'),
                '代表文档标题': id_to_title.get(dup['representative_id'], '未知')
            })
        
        # 保存到CSV
        df = pd.DataFrame(duplicate_info)
        df.to_csv('duplicate_documents.csv', index=False, encoding='utf-8-sig')
        self.logger.info("重复文档详细信息已保存到 duplicate_documents.csv")
        
        # 保存重复文档ID列表
        duplicate_ids = [{'doc_id': dup['duplicate_id']} for dup in duplicates]
        duplicate_ids_df = pd.DataFrame(duplicate_ids)
        duplicate_ids_df.to_csv('duplicate_ids.csv', index=False, encoding='utf-8-sig')
        self.logger.info("重复文档ID列表已保存到 duplicate_ids.csv")
    
    def _visualize_duplicates(self, duplicates: List[Dict]):
        """可视化重复文档分析"""
        # 统计重复组大小
        group_sizes = defaultdict(int)
        for dup in duplicates:
            group_sizes[dup['representative_id']] += 1
        
        # 统计分布
        size_distribution = defaultdict(int)
        for size in group_sizes.values():
            size_distribution[str(size + 1)] += 1
        
        if not size_distribution:
            return
        
        # 绘制分布图
        sizes = sorted(size_distribution.keys(), key=lambda x: int(x))
        counts = [size_distribution[size] for size in sizes]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(sizes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # 添加数据标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.title('重复文档组大小分布', fontsize=16, fontweight='bold')
        plt.xlabel('组大小（文档数量）', fontsize=12)
        plt.ylabel('组数量', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 添加统计信息
        total_groups = sum(counts)
        total_duplicates = len(duplicates)
        plt.text(0.02, 0.98, f'总组数: {total_groups}\n总重复文档数: {total_duplicates}',
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('duplicate_group_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("重复组分布图已保存为 duplicate_group_distribution.png")
        
        # 绘制相似度分布图
        self._plot_similarity_distribution(duplicates)
    
    def _plot_similarity_distribution(self, duplicates: List[Dict]):
        """绘制相似度分布图"""
        similarities = [dup['similarity'] for dup in duplicates]
        
        if not similarities:
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        plt.title('文档相似度分布', fontsize=16, fontweight='bold')
        plt.xlabel('相似度', fontsize=12)
        plt.ylabel('文档对数量', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 添加统计信息
        mean_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=2, label=f'平均值: {mean_sim:.2f}')
        plt.axvline(median_sim, color='blue', linestyle='dashed', linewidth=2, label=f'中位数: {median_sim:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("相似度分布图已保存为 similarity_distribution.png")
    
    def _export_unique_documents(self, documents: List[Dict], duplicates: List[Dict]):
        """导出唯一文档"""
        duplicate_ids = {dup['duplicate_id'] for dup in duplicates}
        unique_docs = [doc for doc in documents if doc['doc_id'] not in duplicate_ids]
        
        output_file = "unique_documents.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_docs, f, ensure_ascii=False, indent=2)
        
        reduction_rate = (len(documents) - len(unique_docs)) / len(documents) * 100
        self.logger.info(f"唯一文档已保存到 {output_file}")
        self.logger.info(f"原始文档数: {len(documents)}, 唯一文档数: {len(unique_docs)}, "
                        f"减少: {len(documents) - len(unique_docs)} ({reduction_rate:.1f}%)")
    
    def _show_high_similarity_pairs(self, duplicates: List[Dict], id_to_title: Dict[str, str], top_n: int = 5):
        """显示高相似度文档对"""
        high_sim_duplicates = sorted(
            [dup for dup in duplicates if dup['similarity'] >= 0.9],
            key=lambda x: x['similarity'],
            reverse=True
        )
        
        if high_sim_duplicates:
            self.logger.info(f"\n高相似度重复文档（相似度≥90%）前{min(top_n, len(high_sim_duplicates))}个：")
            for i, dup in enumerate(high_sim_duplicates[:top_n], 1):
                self.logger.info(f"\n{i}. 相似度: {dup['similarity']*100:.1f}%")
                self.logger.info(f"   重复文档: {dup['duplicate_id']} - {id_to_title.get(dup['duplicate_id'], '未知')}")
                self.logger.info(f"   代表文档: {dup['representative_id']} - {id_to_title.get(dup['representative_id'], '未知')}")


def main():
    """主函数"""
    # 创建去重器实例
    deduplicator = DocumentDeduplicator(
        fingerprint_threshold=0.9,
        minhash_threshold=0.5,
        tfidf_threshold=0.7,
        num_perm=128,
        n_jobs=-1  # 使用所有CPU核心
    )
    
    deduplicator.logger.info("=" * 50)
    deduplicator.logger.info("精准文档去重系统 - 优化版")
    deduplicator.logger.info("=" * 50)
    
    # 加载文档
    json_file = "data.json"
    documents = deduplicator.load_documents(json_file)
    
    if not documents:
        deduplicator.logger.error("没有加载到文档，程序退出")
        return
    
    # 执行去重
    duplicates = deduplicator.deduplicate(documents)
    
    # 分析和导出结果
    deduplicator.analyze_and_export(documents, duplicates)
    
    deduplicator.logger.info("\n去重处理完成！")
    deduplicator.logger.info("生成的文件：")
    deduplicator.logger.info("  - precision_deduplication.log: 详细日志")
    deduplicator.logger.info("  - duplicate_documents.csv: 重复文档详细信息")
    deduplicator.logger.info("  - duplicate_ids.csv: 重复文档ID列表")
    deduplicator.logger.info("  - unique_documents.json: 去重后的唯一文档")
    deduplicator.logger.info("  - duplicate_group_distribution.png: 重复组分布图")
    deduplicator.logger.info("  - similarity_distribution.png: 相似度分布图")


if __name__ == "__main__":
    # 检查依赖
    required_packages = {
        'jieba': 'jieba',
        'sklearn': 'scikit-learn',
        'datasketch': 'datasketch',
        'tqdm': 'tqdm',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'rank_bm25': 'rank-bm25'
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少必要的依赖包：{', '.join(missing_packages)}")
        print(f"请运行以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        exit(1)
    
    main()