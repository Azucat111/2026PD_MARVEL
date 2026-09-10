#!/bin/bash
# 卓翼杯比赛系统启动脚本。

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")"; pwd)
cd "$ROOT_DIR"
export ZHUOYI_ROOT="$ROOT_DIR"

PSP_ROOT="/home/ubuntu/PX4PSP"
export PSP_PATH="$PSP_ROOT"

SCENE="3v3"
DIFF="low"
SEED=""

usage() {
  echo "用法: ./run.sh --scene <3v3|10v10> --diff <low|mid|high> [--seed <number>]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      [ $# -ge 2 ] || { echo "[ERROR] --scene 缺少参数"; exit 1; }
      SCENE="$2"
      shift 2
      ;;
    --diff)
      [ $# -ge 2 ] || { echo "[ERROR] --diff 缺少参数"; exit 1; }
      DIFF="$2"
      shift 2
      ;;
    --seed)
      [ $# -ge 2 ] || { echo "[ERROR] --seed 缺少参数"; exit 1; }
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1"
      usage
      exit 1
      ;;
  esac
done

case "$SCENE" in
  3v3)   N_INT=3;  N_TGT=3;;
  10v10) N_INT=10; N_TGT=10;;
  *)
    echo "[ERROR] --scene 只能是 3v3 或 10v10"
    exit 1
    ;;
esac

case "$DIFF" in
  low|mid|high) ;;
  *)
    echo "[ERROR] --diff 只能是 low、mid 或 high"
    exit 1
    ;;
esac

SCENE_CFG="config/scene_${SCENE}.yaml"
if [ ! -f "$SCENE_CFG" ]; then
  echo "[ERROR] 缺少场景配置: $SCENE_CFG"
  exit 1
fi

if [ -z "$SEED" ]; then
  SEED=$(( (RANDOM << 15 | RANDOM) ))
fi
export ZHUOYI_SEED="$SEED"

RUN_STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_LOG_DIR="$ROOT_DIR/logs/${RUN_STAMP}_${SCENE}_${DIFF}"
mkdir -p "$RUN_LOG_DIR"
ln -sfn "$RUN_LOG_DIR" "$ROOT_DIR/logs/latest"
export ZHUOYI_LOG_DIR="$RUN_LOG_DIR"

eval "$(
python3 - "$SCENE_CFG" <<'PY'
import shlex
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

arena = cfg.get("arena") or {}
sensors = cfg.get("sensors") or {}

print("SCENE_MAP=" + shlex.quote(str(arena.get("map", "DesertTownEX"))))
print("SCENE_GIMBAL_ENABLED=" + ("1" if bool(sensors.get("gimbal", False)) else "0"))
PY
)"

export UE4_MAP="$SCENE_MAP"
export COMPETITION_CONFIG="$ROOT_DIR/config/CompetitionConfig_${SCENE}.json"
COMPETITION_CONFIG_ARCHIVE="$RUN_LOG_DIR/$(basename "$COMPETITION_CONFIG")"
VISION_CONFIG="$ROOT_DIR/config/Config_${SCENE}.json"
ROSTRANS_PARAM_SOURCE="$ROOT_DIR/config/rostrans_${SCENE}.param"
ROSTRANS_PARAM_ACTIVE="$ROOT_DIR/config/rostrans.param"
ROSTRANS_PARAM_ARCHIVE="$RUN_LOG_DIR/rostrans.param"
if [ ! -f "$COMPETITION_CONFIG" ]; then
  echo "[ERROR] 缺少 UE 竞赛碰撞配置: $COMPETITION_CONFIG"
  exit 1
fi
if [ ! -f "$VISION_CONFIG" ]; then
  echo "[ERROR] 缺少视觉配置: $VISION_CONFIG"
  exit 1
fi
if [ ! -f "$ROSTRANS_PARAM_SOURCE" ]; then
  echo "[ERROR] 缺少数据接口配置: $ROSTRANS_PARAM_SOURCE"
  exit 1
fi

python3 - "$COMPETITION_CONFIG" "$N_INT" "$N_TGT" <<'PY'
import json
import sys

path = sys.argv[1]
n_int = int(sys.argv[2])
n_tgt = int(sys.argv[3])

with open(path, encoding="utf-8-sig") as f:
    cfg = json.load(f)

copters = cfg.get("Copters") or []
team1 = [item for item in copters if int(item.get("TeamID", 0)) == 1]
team2 = [item for item in copters if int(item.get("TeamID", 0)) == 2]
if len(copters) != n_int + n_tgt or len(team1) != n_int or len(team2) != n_tgt:
    raise SystemExit(
        f"[ERROR] {path} vehicle/team count mismatch: "
        f"copters={len(copters)} team1={len(team1)} team2={len(team2)} "
        f"expected total={n_int + n_tgt} team1={n_int} team2={n_tgt}"
    )
PY

if [ "$SCENE_GIMBAL_ENABLED" = "1" ]; then
  export START_INTERCEPTOR_PX4="${START_INTERCEPTOR_PX4:-0}"
  export START_RFLYSIM_WAIT_S="${START_RFLYSIM_WAIT_S:-8}"
  export START_COPTERSIM_INTERVAL_S="${START_COPTERSIM_INTERVAL_S:-0.5}"
else
  export START_INTERCEPTOR_PX4="${START_INTERCEPTOR_PX4:-0}"
  export START_RFLYSIM_WAIT_S="${START_RFLYSIM_WAIT_S:-5}"
  export START_COPTERSIM_INTERVAL_S="${START_COPTERSIM_INTERVAL_S:-0.25}"
fi

cp "$COMPETITION_CONFIG" "$COMPETITION_CONFIG_ARCHIVE"
cp "$ROSTRANS_PARAM_SOURCE" "$ROSTRANS_PARAM_ACTIVE"
cp "$ROSTRANS_PARAM_ACTIVE" "$ROSTRANS_PARAM_ARCHIVE"

{
  echo "run_stamp=$RUN_STAMP"
  echo "scene=$SCENE"
  echo "diff=$DIFF"
  echo "seed=$ZHUOYI_SEED"
  echo "psp_root=$PSP_ROOT"
  echo "ue4_map=$UE4_MAP"
  echo "gimbal_enabled=$SCENE_GIMBAL_ENABLED"
  echo "competition_config=$COMPETITION_CONFIG"
  echo "competition_config_archive=$COMPETITION_CONFIG_ARCHIVE"
  echo "vision_config=$VISION_CONFIG"
  echo "rostrans_param=$ROSTRANS_PARAM_SOURCE"
  echo "rostrans_param_archive=$ROSTRANS_PARAM_ARCHIVE"
  echo "start_interceptor_px4=$START_INTERCEPTOR_PX4"
  echo "start_rflysim_wait_s=$START_RFLYSIM_WAIT_S"
  echo "start_coptersim_interval_s=$START_COPTERSIM_INTERVAL_S"
  echo "pwd=$ROOT_DIR"
  date
} >"$RUN_LOG_DIR/run_info.txt"

RUNTIME_LOG_FORWARD_PID=""
COMPETITION_RUNTIME_PID=""
ROSTRANS_PID=""

tail_log() {
  local log_file="$1"
  [ -f "$log_file" ] && tail -80 "$log_file" || true
}

stop_runtime_log_forwarder() {
  if [ -n "${RUNTIME_LOG_FORWARD_PID:-}" ]; then
    kill "$RUNTIME_LOG_FORWARD_PID" 2>/dev/null || true
    RUNTIME_LOG_FORWARD_PID=""
  fi
}

cleanup_and_exit() {
  local code="${1:-0}"
  stop_runtime_log_forwarder
  echo "[run] 正在停止"
  bash "$ROOT_DIR/stop.sh" >"$RUN_LOG_DIR/stop.log" 2>&1 || true
  echo "[run] 已停止"
  exit "$code"
}

fail_with_log() {
  local message="$1"
  local log_file="$2"
  local code="${3:-1}"
  echo "[ERROR] $message"
  tail_log "$log_file"
  cleanup_and_exit "$code"
}

start_runtime_log_forwarder() {
  local log_file="$RUN_LOG_DIR/competition_runtime.log"
  (
    tail -n0 -F "$log_file" 2>/dev/null | while IFS= read -r line; do
      case "$line" in
        *初始化开始*|*比赛开始*|*靶机开始突围*|*"score hits="*|*"match finished"*|*FATAL*|*ERROR*|*WARN*)
          echo "$line"
          ;;
      esac
    done
  ) &
  RUNTIME_LOG_FORWARD_PID=$!
}

start_roscore() {
  echo "[2/6] 启动 ROS"
  if rostopic list >/dev/null 2>&1; then
    return
  fi

  roscore >"$RUN_LOG_DIR/roscore.log" 2>&1 &
  for _ in $(seq 1 20); do
    if rostopic list >/dev/null 2>&1; then
      return
    fi
    sleep 0.5
  done

  echo "[WARN] ROS 10 秒内未就绪，继续启动"
}

start_competition_runtime() {
  echo "[4/6] 启动裁判系统"
  local runtime_bin="$ROOT_DIR/bin/competition_runtime"
  local runtime_log="$RUN_LOG_DIR/competition_runtime.log"

  if [ ! -x "$runtime_bin" ]; then
    fail_with_log "缺少二进制裁判入口: $runtime_bin" "$runtime_log"
  fi

  : >"$runtime_log"
  start_runtime_log_forwarder
  "$runtime_bin" \
    --scene "$SCENE" \
    --diff "$DIFF" \
    --config "$COMPETITION_CONFIG" \
    >>"$runtime_log" 2>&1 &

  COMPETITION_RUNTIME_PID=$!
  sleep 1

  if ! kill -0 "$COMPETITION_RUNTIME_PID" 2>/dev/null; then
    fail_with_log "裁判系统启动后立即退出" "$runtime_log"
  fi
}

start_rostrans() {
  echo "[5/6] 启动数据接口"

  sudo mkdir -p /opt/rostrans/logs 2>/dev/null || true
  sudo chmod 1777 /opt/rostrans/logs 2>/dev/null || true

  (
    cd "$ROOT_DIR/config"
    exec rostrans --vision_sensor_config="$VISION_CONFIG"
  ) >"$RUN_LOG_DIR/rostrans.log" 2>&1 &

  ROSTRANS_PID=$!
  sleep 2

  if ! kill -0 "$ROSTRANS_PID" 2>/dev/null; then
    fail_with_log "数据接口启动后立即退出" "$RUN_LOG_DIR/rostrans.log"
  fi
}

wait_runtime_initialization() {
  echo "[6/6] 等待靶机初始化"
  local runtime_log="$RUN_LOG_DIR/competition_runtime.log"
  local deadline=$((SECONDS + 180))
  local init_line

  while [ "$SECONDS" -lt "$deadline" ]; do
    init_line=$(grep -m1 "初始化成功" "$runtime_log" 2>/dev/null || true)
    if [ -n "$init_line" ]; then
      echo "$init_line"
      echo "[run] 初始化完成，比赛系统已就绪"
      return
    fi
    if ! kill -0 "$COMPETITION_RUNTIME_PID" 2>/dev/null; then
      fail_with_log "裁判系统初始化期间退出" "$runtime_log"
    fi
    sleep 1
  done

  echo "[WARN] 180 秒内未看到靶机初始化成功，继续监控；详情见 $runtime_log"
}

capture_ros_snapshot() {
  (
    sleep 5
    rostopic list >"$RUN_LOG_DIR/rostopic_list.txt" 2>&1 || true
    : >"$RUN_LOG_DIR/rostopic_info.txt"

    topics=()
    for i in $(seq 1 "$N_INT"); do
      topics+=("/radar/interceptor${i}/position")
      if [ "$i" = "1" ]; then
        ns="/mavros"
      else
        ns="/mavros${i}"
      fi
      topics+=("${ns}/local_position/odom")
      topics+=("${ns}/state")
      topics+=("${ns}/setpoint_raw/local")
    done

    for i in $(seq 1 "$N_TGT"); do
      topics+=("/radar/target${i}/position")
    done

    for topic in "${topics[@]}"; do
      {
        echo "===== $topic ====="
        rostopic info "$topic" 2>&1 || true
        echo
      } >>"$RUN_LOG_DIR/rostopic_info.txt"
    done
  ) &
}

check_required_pid() {
  local pid="$1"
  local name="$2"
  local log_file="$3"
  if ! kill -0 "$pid" 2>/dev/null; then
    fail_with_log "关键进程退出: $name pid=$pid" "$log_file"
  fi
}

monitor_required_processes() {
  local vehicle_num=$((N_INT + N_TGT))
  local expected_px4=$N_TGT
  if [ "$START_INTERCEPTOR_PX4" = "1" ]; then
    expected_px4=$vehicle_num
  fi
  local sim_strikes=0

  while true; do
    check_required_pid "$COMPETITION_RUNTIME_PID" "competition_runtime" "$RUN_LOG_DIR/competition_runtime.log"
    check_required_pid "$ROSTRANS_PID" "rostrans" "$RUN_LOG_DIR/rostrans.log"

    local rfly copter px4
    rfly=$(pgrep -x RflySim3D 2>/dev/null | wc -l)
    copter=$(pgrep -f "CopterSim" 2>/dev/null | wc -l)
    px4=$(pgrep -x px4 2>/dev/null | wc -l)
    if [ "$rfly" -lt 1 ] || [ "$copter" -lt "$vehicle_num" ] || [ "$px4" -lt "$expected_px4" ]; then
      sim_strikes=$((sim_strikes + 1))
      echo "[WARN] 仿真进程异常(${sim_strikes}/2): RflySim3D=${rfly} CopterSim=${copter}/${vehicle_num} px4=${px4}/${expected_px4}"
      if [ "$sim_strikes" -ge 2 ]; then
        fail_with_log "仿真关键进程异常" "$RUN_LOG_DIR/start_coptersim.log"
      fi
    else
      sim_strikes=0
    fi

    sleep 2
  done
}

trap 'cleanup_and_exit 0' INT TERM

echo "=========================================="
echo "  卓翼杯比赛系统 scene=$SCENE diff=$DIFF"
echo "  随机种子: $ZHUOYI_SEED"
echo "  日志目录: $RUN_LOG_DIR"
echo "=========================================="

echo "[1/6] 清理环境"
bash "$ROOT_DIR/stop.sh" >"$RUN_LOG_DIR/stop_before_start.log" 2>&1 || true
bash sim/udp_port_free.sh "$((N_INT + N_TGT))" >"$RUN_LOG_DIR/udp_port_free.log" 2>&1 || true

start_roscore

echo "[3/6] 启动仿真平台"
set +e
bash sim/start_coptersim.sh "$SCENE" >"$RUN_LOG_DIR/start_coptersim.log" 2>&1
SIM_START_RC=$?
set -e
if [ "$SIM_START_RC" -ne 0 ]; then
  fail_with_log "仿真平台启动失败" "$RUN_LOG_DIR/start_coptersim.log" "$SIM_START_RC"
fi

start_competition_runtime
start_rostrans
capture_ros_snapshot
wait_runtime_initialization

echo "[run] 按 Ctrl-C 或运行 ./stop.sh 停止"
echo "[run] 最新日志: $ROOT_DIR/logs/latest"

monitor_required_processes
