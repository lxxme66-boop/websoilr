"""
生成问题答案函数调用流程图
展示 _generate_answer_with_llm 和 _generate_answer_with_llm_context 的调用关系
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# 创建图形
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义颜色
color_llm = '#FFE5B4'  # 浅桃色
color_context = '#B4E5FF'  # 浅蓝色
color_question = '#E5FFB4'  # 浅绿色
color_answer = '#FFB4E5'  # 浅紫色

# 主标题
ax.text(5, 9.5, '问题生成器答案生成函数调用流程', 
        fontsize=20, ha='center', weight='bold')

# === 左侧：_generate_answer_with_llm 流程 ===
# 主函数框
llm_box = FancyBboxPatch((0.5, 7), 2, 1, 
                         boxstyle="round,pad=0.1",
                         facecolor=color_llm,
                         edgecolor='black',
                         linewidth=2)
ax.add_patch(llm_box)
ax.text(1.5, 7.5, '_generate_answer_with_llm', 
        ha='center', va='center', fontsize=10, weight='bold')

# 调用场景
scenarios_llm = [
    ('事实型问题\n(模板)', 5.5),
    ('多跳问题', 4.5),
    ('模板问题', 3.5)
]

for i, (scenario, y) in enumerate(scenarios_llm):
    box = FancyBboxPatch((0.5, y-0.4), 2, 0.8,
                        boxstyle="round,pad=0.05",
                        facecolor=color_question,
                        edgecolor='gray')
    ax.add_patch(box)
    ax.text(1.5, y, scenario, ha='center', va='center', fontsize=9)
    # 箭头
    ax.arrow(1.5, y+0.4, 0, 0.6, head_width=0.1, head_length=0.1, 
             fc='black', ec='black')

# 内部处理
process_box = FancyBboxPatch((0.5, 2), 2, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor='#FFE5CC',
                           edgecolor='gray')
ax.add_patch(process_box)
ax.text(1.5, 2.4, '创建简单prompt\n直接生成答案', 
        ha='center', va='center', fontsize=8)

# 箭头到处理
ax.arrow(1.5, 3.1, 0, -0.3, head_width=0.1, head_length=0.1, 
         fc='black', ec='black')

# === 右侧：_generate_answer_with_llm_context 流程 ===
# 主函数框
context_box = FancyBboxPatch((6, 7), 2.5, 1,
                           boxstyle="round,pad=0.1",
                           facecolor=color_context,
                           edgecolor='black',
                           linewidth=2)
ax.add_patch(context_box)
ax.text(7.25, 7.5, '_generate_answer_with_llm_context',
        ha='center', va='center', fontsize=10, weight='bold')

# 调用场景
scenarios_context = [
    ('比较型答案', 5.8),
    ('推理型答案', 5.2),
    ('详细比较答案\n(LLM)', 4.6),
    ('详细推理答案\n(LLM)', 4.0),
    ('富上下文答案', 3.4)
]

for i, (scenario, y) in enumerate(scenarios_context):
    box = FancyBboxPatch((6, y-0.25), 2.5, 0.5,
                        boxstyle="round,pad=0.05",
                        facecolor=color_answer,
                        edgecolor='gray')
    ax.add_patch(box)
    ax.text(7.25, y, scenario, ha='center', va='center', fontsize=8)
    # 箭头
    ax.arrow(7.25, y+0.25, 0, 0.45, head_width=0.1, head_length=0.05,
             fc='black', ec='black')

# 上下文信息框
context_info_box = FancyBboxPatch((5.5, 2), 3.5, 0.8,
                                boxstyle="round,pad=0.05",
                                facecolor='#CCE5FF',
                                edgecolor='gray')
ax.add_patch(context_info_box)
ax.text(7.25, 2.4, '构建丰富上下文\n(实体信息、关系链、领域知识)',
        ha='center', va='center', fontsize=8)

# 箭头到上下文
ax.arrow(7.25, 3.15, 0, -0.35, head_width=0.1, head_length=0.1,
         fc='black', ec='black')

# === 中间连接：可选的上下文增强 ===
# 虚线箭头
ax.plot([2.5, 6], [7.5, 7.5], 'k--', alpha=0.5)
ax.text(4.25, 7.7, '需要额外上下文时', ha='center', fontsize=8, style='italic')

# === 底部：输出说明 ===
output_box1 = FancyBboxPatch((0.5, 0.5), 2, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor='#F0F0F0',
                           edgecolor='black')
ax.add_patch(output_box1)
ax.text(1.5, 0.9, '快速答案\n(适合模板问题)', 
        ha='center', va='center', fontsize=8)

output_box2 = FancyBboxPatch((6, 0.5), 2.5, 0.8,
                           boxstyle="round,pad=0.05",
                           facecolor='#F0F0F0',
                           edgecolor='black')
ax.add_patch(output_box2)
ax.text(7.25, 0.9, '高质量答案\n(适合复杂问题)',
        ha='center', va='center', fontsize=8)

# 箭头到输出
ax.arrow(1.5, 1.8, 0, -0.5, head_width=0.1, head_length=0.1,
         fc='black', ec='black')
ax.arrow(7.25, 1.8, 0, -0.5, head_width=0.1, head_length=0.1,
         fc='black', ec='black')

# === 添加策略说明 ===
strategy_text = """
策略分配：
• 40% 模板问题 → _generate_answer_with_llm
• 60% LLM问题 → _generate_answer_with_llm_context
"""
ax.text(9.5, 8.5, strategy_text, fontsize=9, ha='right', 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray'))

# === 添加图例 ===
legend_elements = [
    mpatches.Patch(color=color_llm, label='直接生成函数'),
    mpatches.Patch(color=color_context, label='上下文生成函数'),
    mpatches.Patch(color=color_question, label='问题类型'),
    mpatches.Patch(color=color_answer, label='答案类型')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# 保存图形
plt.tight_layout()
plt.savefig('answer_generation_flow.png', dpi=300, bbox_inches='tight')
plt.savefig('answer_generation_flow.pdf', bbox_inches='tight')
print("流程图已生成：answer_generation_flow.png 和 answer_generation_flow.pdf")

# === 生成详细的调用统计图 ===
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：问题类型分布
question_types = ['事实型\n(模板)', '事实型\n(LLM)', '比较型\n(模板)', 
                  '比较型\n(LLM)', '推理型\n(模板)', '推理型\n(LLM)', '多跳型']
answer_methods = ['直接', '直接', '上下文', '上下文', '上下文', '上下文', '直接']
colors = [color_llm if m == '直接' else color_context for m in answer_methods]

ax1.bar(range(len(question_types)), [1]*len(question_types), color=colors)
ax1.set_xticks(range(len(question_types)))
ax1.set_xticklabels(question_types, rotation=45, ha='right')
ax1.set_ylabel('答案生成方法')
ax1.set_title('不同问题类型的答案生成方法', fontsize=12)
ax1.set_ylim(0, 1.5)

# 添加标注
for i, (qt, am) in enumerate(zip(question_types, answer_methods)):
    ax1.text(i, 0.5, am, ha='center', va='center', fontsize=10, weight='bold')

# 右图：性能对比
categories = ['生成速度', 'Token消耗', '答案质量', '上下文丰富度']
llm_scores = [0.9, 0.9, 0.6, 0.3]  # _generate_answer_with_llm
context_scores = [0.5, 0.4, 0.9, 0.95]  # _generate_answer_with_llm_context

x = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x - width/2, llm_scores, width, label='_generate_answer_with_llm', 
                color=color_llm)
bars2 = ax2.bar(x + width/2, context_scores, width, label='_generate_answer_with_llm_context',
                color=color_context)

ax2.set_ylabel('相对得分')
ax2.set_title('两种答案生成方法的性能对比', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend()
ax2.set_ylim(0, 1.1)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('answer_generation_comparison.png', dpi=300, bbox_inches='tight')
print("对比图已生成：answer_generation_comparison.png")

plt.show()