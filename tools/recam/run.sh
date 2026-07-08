#!/bin/bash

# 脚本：start_multiple_tmux.sh
# 用法：./start_multiple_tmux.sh <n> <script> [bash_args...]

# 检查参数数量
if [ $# -lt 2 ]; then
    echo "用法: $0 <会话数量> <脚本路径> [Bash脚本参数...]"
    echo "示例: $0 5 task.sh --arg1 value1"
    exit 1
fi

NUM_SESSIONS=$1
BASH_SCRIPT=$2
shift 2  # 移除前两个参数，剩下的都是Bash脚本的参数
BASH_ARGS="$@"

# 检查Bash脚本是否存在
if [ ! -f "$BASH_SCRIPT" ]; then
    echo "错误: Bash脚本 '$BASH_SCRIPT' 不存在"
    exit 1
fi

# 检查是否安装了tmux
if ! command -v tmux &> /dev/null; then
    echo "错误: 未找到tmux，请先安装tmux"
    exit 1
fi

# 基础会话名称
BASE_SESSION_NAME="bash_session"

echo "准备启动 $NUM_SESSIONS 个tmux会话..."
echo "Bash脚本: $BASH_SCRIPT"
echo "Bash参数: $BASH_ARGS"
echo ""

# 创建并启动tmux会话
for ((i=0; i<NUM_SESSIONS; i++)); do
    SESSION_NAME="${BASE_SESSION_NAME}_${i}"
    GPU_ID=$((i % 8))
    echo "创建会话: $SESSION_NAME for $GPU_ID"
    
    # 创建新会话并运行Bash脚本
    tmux new-session -d -s "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=$GPU_ID bash '$BASH_SCRIPT' $BASH_ARGS; exec bash"
    
    # 检查是否创建成功
    if [ $? -eq 0 ]; then
        echo "  ✓ 会话 $SESSION_NAME 已创建并启动"
    else
        echo "  ✗ 创建会话 $SESSION_NAME 失败"
    fi
done

echo ""
echo "所有会话已创建完成!"
echo ""
echo "可用命令:"
echo "  1. 连接到指定会话: tmux attach -t ${BASE_SESSION_NAME}_1"
echo "  2. 查看所有会话: tmux list-sessions"
echo "  3. 关闭所有会话: for i in {0..$NUM_SESSIONS}; do tmux kill-session -t ${BASE_SESSION_NAME}_\${i}; done"
echo "  4. 向所有会话发送命令: for i in {0..$NUM_SESSIONS}; do tmux send-keys -t ${BASE_SESSION_NAME}_\${i} '命令' C-m; done"