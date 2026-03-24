#!/bin/bash

# Agent Web Interface 启动脚本

echo "=========================================="
echo "  Agent Web Interface"
echo "=========================================="
echo ""

# 检查 Flask 是否已安装
if ! python -c "import flask" 2>/dev/null; then
    echo "错误: Flask 未安装"
    echo "请运行: pip install flask"
    exit 1
fi

# 检查端口 5000 是否被占用
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "警告: 端口 5000 已被占用"
    echo "正在尝试终止占用端口的进程..."
    
    # 查找并终止占用端口的进程
    PID=$(lsof -ti :5000)
    if [ -n "$PID" ]; then
        kill -9 $PID 2>/dev/null
        echo "已终止进程 $PID"
        sleep 1
    fi
fi

# 启动 Web 服务器
echo "正在启动 Web 服务器..."
echo "服务器地址: http://localhost:5000"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "=========================================="
echo ""

# 启动服务器
python web_server.py
