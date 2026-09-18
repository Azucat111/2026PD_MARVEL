#!/bin/bash
# MARVEL Linux训练环境准备脚本
# 将Windows开发的仿真平台文件复制到linux/目录，供Linux训练使用

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WINDOWS_ROOT="$PROJECT_ROOT/MARVEL-main (1)/MARVEL-main"
LINUX_ROOT="$PROJECT_ROOT/linux"

echo "================================================"
echo "  MARVEL Linux训练环境准备"
echo "================================================"
echo "源目录: $WINDOWS_ROOT"
echo "目标目录: $LINUX_ROOT"
echo ""

# 1. 复制配置系统
echo "[1/7] 复制配置文件..."
cp -r "$WINDOWS_ROOT/configs" "$LINUX_ROOT/"
echo "  ✓ configs/ 复制完成"

# 2. 复制仿真平台模块
echo "[2/7] 复制仿真平台模块..."
cp "$WINDOWS_ROOT/utils/simulation_runtime.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/dynamics_models.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/sensor_models.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/communication_model.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/obstacle_manager.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/task_manager.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/evaluator.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/scenario_config.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/safety_shield.py" "$LINUX_ROOT/utils/" 2>/dev/null || true
cp "$WINDOWS_ROOT/utils/policy_adapter.py" "$LINUX_ROOT/utils/" 2>/dev/null || true
echo "  ✓ 仿真平台模块复制完成"

# 3. 复制MARVEL原始核心
echo "[3/7] 复制MARVEL核心模块..."
cp "$WINDOWS_ROOT/utils/model.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/agent.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/env.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/node_manager.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/motion_model.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/sensor.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/utils.py" "$LINUX_ROOT/utils/"
cp "$WINDOWS_ROOT/utils/runner.py" "$LINUX_ROOT/utils/" 2>/dev/null || true
cp "$WINDOWS_ROOT/utils/multi_agent_worker.py" "$LINUX_ROOT/utils/" 2>/dev/null || true
cp "$WINDOWS_ROOT/parameter.py" "$LINUX_ROOT/"
echo "  ✓ MARVEL核心模块复制完成"

# 4. 复制地图（如果存在）
echo "[4/7] 复制地图文件..."
if [ -d "$WINDOWS_ROOT/DungeonMaps" ]; then
    cp -r "$WINDOWS_ROOT/DungeonMaps" "$LINUX_ROOT/"
    echo "  ✓ DungeonMaps 复制完成"
fi
if [ -d "$WINDOWS_ROOT/maps" ]; then
    cp -r "$WINDOWS_ROOT/maps" "$LINUX_ROOT/"
    echo "  ✓ maps 复制完成"
fi
if [ -d "$WINDOWS_ROOT/maps_medium" ]; then
    cp -r "$WINDOWS_ROOT/maps_medium" "$LINUX_ROOT/"
    echo "  ✓ maps_medium 复制完成"
fi

# 5. 复制其他依赖
echo "[5/7] 复制其他依赖..."
cp "$WINDOWS_ROOT/utils/quads.py" "$LINUX_ROOT/utils/" 2>/dev/null || true
echo "  ✓ 其他依赖复制完成"

# 6. 创建__init__.py
echo "[6/7] 创建__init__.py..."
touch "$LINUX_ROOT/utils/__init__.py"
touch "$LINUX_ROOT/scripts/__init__.py"
echo "  ✓ __init__.py 创建完成"

# 7. 设置权限
echo "[7/7] 设置可执行权限..."
chmod +x "$LINUX_ROOT/scripts/"*.py 2>/dev/null || true
chmod +x "$LINUX_ROOT/scripts/"*.sh 2>/dev/null || true
echo "  ✓ 权限设置完成"

echo ""
echo "================================================"
echo "  ✓ Linux训练环境准备完成"
echo "================================================"
echo ""
echo "下一步："
echo "1. 将 $LINUX_ROOT 打包传输到Linux服务器"
echo "   tar -czf marvel_linux.tar.gz linux/"
echo ""
echo "2. 在Linux上解压并安装依赖"
echo "   tar -xzf marvel_linux.tar.gz"
echo "   cd linux"
echo "   pip install -r requirements.txt"
echo ""
echo "3. 开始训练"
echo "   python scripts/train_multi_task.py --scenario configs/scenarios/urban_rescue_simple.yaml"
echo ""
