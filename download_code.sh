#!/bin/bash

echo "=== WebSailor Domain Dataset 代码下载脚本 ==="
echo ""

# 创建时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="websailor_domain_dataset_${TIMESTAMP}.tar.gz"

echo "正在打包代码文件..."
tar -czf "$ARCHIVE_NAME" websailor_domain_dataset/ README.md

if [ $? -eq 0 ]; then
    echo "✅ 代码已成功打包到: $ARCHIVE_NAME"
    echo ""
    echo "文件大小:"
    ls -lh "$ARCHIVE_NAME"
    echo ""
    echo "🎉 您可以通过以下方式下载这个文件:"
    echo "1. 如果在本地环境，直接复制该文件"
    echo "2. 如果在远程环境，使用 scp 或其他文件传输工具"
    echo "3. 如果在云环境，使用相应的文件下载功能"
else
    echo "❌ 打包失败，请检查文件权限"
fi