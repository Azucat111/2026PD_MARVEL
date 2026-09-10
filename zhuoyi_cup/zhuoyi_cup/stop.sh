#!/bin/bash
set +e

echo "[stop] 清理卓翼杯进程与端口 ..."

PATS_F=(
  "bin/competition_runtime"
  '(^|[^-])rostrans'
  "CopterSim"
  "scene_setup.py"
  "foxglove_bridge"
  "roscore" "rosmaster" "rosout"
)
PATS_X=("px4" "RflySim3D" "QGroundControl")

_killall() {
  local sig="$1" p
  for p in "${PATS_F[@]}"; do pkill "$sig" -f "$p" 2>/dev/null; done
  for p in "${PATS_X[@]}"; do pkill "$sig" -x "$p" 2>/dev/null; done
}

_killall -TERM
sleep 1
_killall -KILL

if command -v fuser >/dev/null 2>&1; then
  ports=(20006 20007)
  for v in $(seq 20100 20141); do ports+=("$v"); done
  for v in $(seq 30100 30141); do ports+=("$v"); done
  for v in $(seq 40100 40141); do ports+=("$v"); done
  for v in "${ports[@]}"; do fuser -k "${v}/udp" 2>/dev/null; done
  fuser -k 11311/tcp 2>/dev/null
  sleep 0.3
fi

if command -v ss >/dev/null 2>&1 && ss -lun 2>/dev/null | grep -q ':20006 '; then
  echo "[stop][WARN] 20006/udp 仍被占用:"
  ss -lunp 2>/dev/null | grep ':20006 ' || true
else
  echo "[stop] 20006/udp 已释放"
fi

echo "[stop] done"
