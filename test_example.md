# 测试示例

## 1. 准备测试文件

创建一个测试文件夹和英文文档：

```bash
# 创建测试文件夹
mkdir -p test_english_docs

# 创建一个长文档测试文件
cat > test_english_docs/long_document.txt << 'EOF'
Introduction to Artificial Intelligence

Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

History of AI

The concept of artificial intelligence dates back to ancient times, but the field as we know it today began in the 1950s. Alan Turing proposed the Turing Test in 1950 as a criterion for intelligence in machines. The term "artificial intelligence" was coined by John McCarthy in 1956 at the Dartmouth Conference.

Types of AI

1. Narrow AI (Weak AI): This type of AI is designed to perform a specific task, such as facial recognition, internet searches, or driving a car. Most AI systems today are narrow AI.

2. General AI (Strong AI): This refers to AI that has human-level intelligence and can understand, learn, and apply knowledge across different domains. General AI does not yet exist.

3. Super AI: This hypothetical form of AI would surpass human intelligence in all aspects. It remains in the realm of science fiction.

Machine Learning

Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. There are three main types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data, where the desired output is known.

2. Unsupervised Learning: The algorithm finds patterns in unlabeled data without predefined categories.

3. Reinforcement Learning: The algorithm learns through trial and error by receiving rewards or penalties for its actions.

Deep Learning

Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers. These networks are inspired by the structure and function of the human brain. Deep learning has been particularly successful in areas such as image recognition, natural language processing, and game playing.

Applications of AI

AI has numerous applications across various industries:

1. Healthcare: AI is used for disease diagnosis, drug discovery, and personalized treatment plans.

2. Finance: AI helps in fraud detection, algorithmic trading, and risk assessment.

3. Transportation: Self-driving cars and traffic optimization systems rely on AI.

4. Education: AI-powered tutoring systems and personalized learning platforms.

5. Entertainment: Recommendation systems for movies, music, and content creation.

Ethical Considerations

As AI becomes more prevalent, several ethical concerns have emerged:

1. Bias and Fairness: AI systems can perpetuate or amplify existing biases in data.

2. Privacy: The collection and use of personal data by AI systems raise privacy concerns.

3. Job Displacement: Automation through AI may lead to job losses in certain sectors.

4. Accountability: Determining responsibility when AI systems make decisions or errors.

5. Security: AI systems can be vulnerable to attacks or misuse.

Future of AI

The future of AI holds both promises and challenges. Potential developments include:

1. More sophisticated natural language processing
2. Advanced robotics and automation
3. Improved healthcare diagnostics and treatments
4. Enhanced scientific research capabilities
5. Better climate modeling and environmental solutions

However, it is crucial to develop AI responsibly, ensuring that it benefits humanity while minimizing potential risks and negative impacts.

Conclusion

Artificial Intelligence is transforming our world in unprecedented ways. As we continue to advance in this field, it is essential to balance innovation with ethical considerations and ensure that AI development serves the greater good of humanity.
EOF

echo "✅ 测试文件创建完成"
```

## 2. 运行翻译

```bash
# 方法1：使用运行脚本
./run_translate.sh

# 方法2：直接运行Python
python3 translate_deepseek_improved.py
```

## 3. 交互示例

运行后的交互过程：

```
🚀 改进版DeepSeek翻译工具
==================================================
可用模型:
1. DeepSeek-R1-Distill-Qwen-1.5B (最快)
2. DeepSeek-R1-Distill-Qwen-7B (推荐)
3. DeepSeek-R1-Distill-Qwen-14B (高质量)
4. gte_Qwen2-7B-instruct (替代选项)

请选择模型 (1-4): 2
✅ 选择模型: DeepSeek-R1-Distill-Qwen-7B

📁 输入英文txt文件夹路径: test_english_docs
📂 输出文件夹路径 [默认: ./chinese_translations]: 

文本分割选项:
1. 自动分割 (推荐，每1000字符)
2. 按段落分割 (适合结构化文档)
3. 智能分割 (根据句子边界)
选择分割策略 (1-3) [默认: 1]: 1

开始加载模型: DeepSeek-R1-Distill-Qwen-7B
🔄 加载tokenizer...
🔄 加载模型...
✅ 模型加载成功

📁 找到 1 个文件

[1/1] 翻译: long_document.txt
  文件大小: 3821 字符
  分割成 4 个片段
  翻译片段 1/4 (1000 字符) ✓
  翻译片段 2/4 (1000 字符) ✓
  翻译片段 3/4 (1000 字符) ✓
  翻译片段 4/4 (821 字符) ✓
✅ 完成: long_document_中文.txt (耗时: 45.2秒)

==================================================
翻译完成！
成功: 1/1 个文件
总耗时: 45.2秒
输出目录: /workspace/chinese_translations
```

## 4. 检查结果

```bash
# 查看翻译结果
cat chinese_translations/long_document_中文.txt

# 对比原文和译文长度
echo "原文长度: $(wc -c < test_english_docs/long_document.txt) 字符"
echo "译文长度: $(wc -c < chinese_translations/long_document_中文.txt) 字符"
```

## 5. 批量翻译示例

```bash
# 创建多个测试文件
mkdir -p batch_test
echo "This is a short test document." > batch_test/test1.txt
echo "Another test file with more content..." > batch_test/test2.txt

# 运行批量翻译
python3 translate_deepseek_improved.py
# 输入: batch_test
# 输出: batch_translations
# 选择模型和分割策略...
```

## 注意事项

1. **首次运行**会下载模型，需要较长时间和足够的磁盘空间
2. **长文档**会自动分割成多个片段，确保完整翻译
3. **内存需求**：7B模型需要约16GB内存，14B模型需要约32GB内存
4. **GPU加速**：有CUDA GPU会显著提升翻译速度