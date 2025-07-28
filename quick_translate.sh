#!/bin/bash

# DeepSeek 翻译工具快速启动脚本

echo "🚀 DeepSeek 批量翻译工具"
echo "========================"
echo ""
echo "请选择运行模式："
echo "1. 交互式运行（可以看到实时进度）"
echo "2. 后台运行（适合大量文件）"
echo "3. Screen 会话运行（可断开连接）"
echo ""

read -p "选择模式 (1-3): " mode

case $mode in
    1)
        echo "启动交互式翻译..."
        python translate_deepseek_improved.py
        ;;
    2)
        echo "启动后台翻译..."
        nohup python translate_deepseek_improved.py > translate_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        pid=$!
        echo "✅ 翻译进程已启动 (PID: $pid)"
        echo "📄 日志文件: translate_$(date +%Y%m%d_%H%M%S).log"
        echo ""
        echo "查看进度: tail -f translate_*.log"
        echo "停止翻译: kill $pid"
        ;;
    3)
        session_name="translate_$(date +%Y%m%d_%H%M%S)"
        echo "创建 Screen 会话: $session_name"
        screen -dmS $session_name bash -c "python translate_deepseek_improved.py; exec bash"
        echo "✅ Screen 会话已创建"
        echo ""
        echo "连接会话: screen -r $session_name"
        echo "分离会话: Ctrl+A 然后按 D"
        echo "列出会话: screen -ls"
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "提示："
echo "- 建议使用 7B 模型进行批量翻译（平衡速度和质量）"
echo "- 对于长文档，选择'小片段'可以获得更完整的翻译"
echo "- 翻译结果保存在 chinese_translations 文件夹"