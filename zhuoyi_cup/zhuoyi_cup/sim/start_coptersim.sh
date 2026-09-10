#!/bin/bash
set -e

SCENE="${1:-3v3}"
SIM_DIR=$(cd "$(dirname "$0")"; pwd)
ROOT_DIR=$(cd "${SIM_DIR}/.."; pwd)
LOG_DIR="${ZHUOYI_LOG_DIR:-${SIM_DIR}/../logs}"
mkdir -p "$LOG_DIR"

PSP_ROOT="${PSP_PATH:?PSP_PATH 未设置，请通过 ./run.sh 启动}"
export PSP_PATH="$PSP_ROOT"

COPTERSIM_DIR="${COPTERSIM_DIR:-${PSP_ROOT}/CopterSimNoUI}"

INTERCEPTOR_CLASS_3D=102000003
TARGET_CLASS_3D=103000003
INTERCEPTOR_MODEL="MulticopterNOpx4"
INTERCEPTOR_SIM_MODE=3
INTERCEPTOR_UDP_MODE="Mavlink_Vision"
TARGET_MODEL=0
TARGET_SIM_MODE=2
TARGET_UDP_MODE=2
PX4_SITL_FRAME="iris"
IS_BROADCAST=0
COPTERSIM_MODE_ARG=1
UE_LOAD_UI=1
UE_RES_X=1280
UE_RES_Y=720
START_RFLYSIM_WAIT_S="${START_RFLYSIM_WAIT_S:-6}"
START_COPTERSIM_INTERVAL_S="${START_COPTERSIM_INTERVAL_S:-0.4}"
START_INTERCEPTOR_PX4="${START_INTERCEPTOR_PX4:-1}"

case "$SCENE" in
  3v3)   N_INT=3;  N_TGT=3;;
  10v10) N_INT=10; N_TGT=10;;
  *) echo "[ERROR] 未知场景: $SCENE"; exit 1;;
esac
VEHICLE_NUM=$((N_INT + N_TGT))
START_INDEX=1

echo "[sim] 场景=$SCENE 拦截机=$N_INT 靶机=$N_TGT"

eval "$(
python3 - "$SCENE" "$SIM_DIR" <<'PY'
import os
import sys

scene_name = sys.argv[1]
sim_dir = sys.argv[2]
repo = os.path.dirname(sim_dir)
for path in (os.path.join(repo, "referee"), os.path.join(repo, "bin")):
    if path not in sys.path:
        sys.path.insert(0, path)
import scene
import scenario

cfg = scene.load_scene(scene_name)
master_seed = scenario.master_seed_from_env()
ints = scene.expand_positions(cfg["interceptor"])
tgts = scenario.resolve_target_spawns(cfg, master_seed)
if len(ints) != int(cfg["interceptor"]["count"]) or len(tgts) != int(cfg["target"]["count"]):
    raise SystemExit("scene 位置数量与 count 不一致")

pos_x = [p[0] for p in ints + tgts]
pos_y = [p[1] for p in ints + tgts]
yaw = [0.0] * len(ints) + [180.0] * len(tgts)

def bash_array(name, values):
    body = " ".join(f'"{v:.6f}"' for v in values)
    print(f"{name}=({body})")

bash_array("PosX", pos_x)
bash_array("PosY", pos_y)
bash_array("YawArr", yaw)
PY
)"

ld_path="${LD_LIBRARY_PATH:-}"

pkill -x RflySim3D || true
echo "[sim] 启动 RflySim3D"
cd "${PSP_ROOT}/RflySimUE5/RflySim3D/Binaries/Linux"
RFLYSIM_LOG="$LOG_DIR/rflysim3d.log"
UE_ARGS=(
  "RflySim3D"
  "${UE4_MAP}"
  "-game"
  "-ResX=${UE_RES_X}"
  "-ResY=${UE_RES_Y}"
  "-WINDOWED"
  "-LoadUI=${UE_LOAD_UI}"
  "-cmd=RflyChangeMapbyName-${UE4_MAP}"
)
if [ -n "${COMPETITION_CONFIG:-}" ]; then
  UE_ARGS+=("-CompetitionConfig=${COMPETITION_CONFIG}")
fi
printf '[sim] RflySim3D args:' >"$RFLYSIM_LOG"
printf ' %q' "${UE_ARGS[@]}" >>"$RFLYSIM_LOG"
printf '\n' >>"$RFLYSIM_LOG"
if [ -n "${COMPETITION_CONFIG:-}" ]; then
  printf '[sim] CompetitionConfig: %q\n' "$COMPETITION_CONFIG" >>"$RFLYSIM_LOG"
fi
./RflySim3D "${UE_ARGS[@]}" >>"$RFLYSIM_LOG" 2>&1 &
RFLYSIM_PID=$!

_rfly_deadline=$(( SECONDS + START_RFLYSIM_WAIT_S ))
while [ "$SECONDS" -lt "$_rfly_deadline" ]; do
  if grep -q "CompetitionConfig: loaded" "$RFLYSIM_LOG" 2>/dev/null; then
    echo "[sim] RflySim3D 已就绪"
    sleep 1
    break
  fi
  sleep 0.5
done
if ! kill -0 "$RFLYSIM_PID" 2>/dev/null; then
  echo "[WARN] RflySim3D 进程已退出，日志: ${RFLYSIM_LOG}"
fi

pkill -f "CopterSim" || true
export LD_LIBRARY_PATH=${COPTERSIM_DIR}/lib:${ld_path}
cd "${COPTERSIM_DIR}"
chmod +x ./CopterSim 2>/dev/null || true
COPTERSIM_LOG_DIR="$LOG_DIR/coptersim"
mkdir -p "$COPTERSIM_LOG_DIR"

cntr=$START_INDEX
endNum=$((VEHICLE_NUM + START_INDEX))
while [ "$cntr" -lt "$endNum" ]; do
  idx=$((cntr - START_INDEX))
  if [ "$cntr" -le "$N_INT" ]; then
    MODEL="$INTERCEPTOR_MODEL"; SMODE="$INTERCEPTOR_SIM_MODE"; UMODE="$INTERCEPTOR_UDP_MODE"; CLS="$INTERCEPTOR_CLASS_3D"; ROLE="interceptor"
  else
    MODEL="$TARGET_MODEL"; SMODE="$TARGET_SIM_MODE"; UMODE="$TARGET_UDP_MODE"; CLS="$TARGET_CLASS_3D"; ROLE="target"
  fi
  COPTERSIM_LOG="$COPTERSIM_LOG_DIR/coptersim_${cntr}.log"
  (
    echo "[sim] CopterSim #${cntr} ${ROLE}"
    printf '[sim] command:'
    printf ' %q' ./CopterSim "$COPTERSIM_MODE_ARG" "$cntr" "$CLS" "$MODEL" "$SMODE" \
      "$UE4_MAP" "$IS_BROADCAST" "${PosX[$idx]}" "${PosY[$idx]}" "${YawArr[$idx]}" 1 "$UMODE"
    printf '\n'
    exec ./CopterSim "$COPTERSIM_MODE_ARG" "$cntr" "$CLS" "$MODEL" "$SMODE" \
      "$UE4_MAP" "$IS_BROADCAST" "${PosX[$idx]}" "${PosY[$idx]}" "${YawArr[$idx]}" 1 "$UMODE"
  ) >"$COPTERSIM_LOG" 2>&1 &
  sleep "$START_COPTERSIM_INTERVAL_S"
  cntr=$((cntr + 1))
done

echo "[sim] 等待 UE 注册飞机 ${VEHICLE_NUM} 架"
_so_deadline=$(( SECONDS + ${SCENE_SETUP_WAIT_TIMEOUT_S:-90} ))
while :; do
  _so_created=$(grep -acE "CreateCopter: Request CopterID=" "$RFLYSIM_LOG" 2>/dev/null || true)
  _so_created=${_so_created:-0}
  if [ "$_so_created" -ge "$VEHICLE_NUM" ]; then
    echo "[sim] UE 飞机注册完成 ${_so_created}/${VEHICLE_NUM}"
    break
  fi
  if [ "$SECONDS" -ge "$_so_deadline" ]; then
    echo "[WARN] 等待飞机注册超时 ${_so_created}/${VEHICLE_NUM}"
    break
  fi
  sleep 1
done

cd "${PSP_ROOT}/Firmware"
chmod +x BkFile/EnvOri.sh Tools/sitl_multiple_run_rfly.sh
./BkFile/EnvOri.sh
PX4_BOOT_LOG="$LOG_DIR/px4_sitl_boot.log"
set +e
if [ "$START_INTERCEPTOR_PX4" = "1" ]; then
  echo "[sim] 启动 PX4 SITL x ${VEHICLE_NUM}"
  ./Tools/sitl_multiple_run_rfly.sh "${VEHICLE_NUM}" "${START_INDEX}" "${PX4_SITL_FRAME}" >"$PX4_BOOT_LOG" 2>&1
else
  TARGET_START=$((N_INT + START_INDEX))
  echo "[sim] 启动 PX4 SITL x ${N_TGT}"
  ./Tools/sitl_multiple_run_rfly.sh "${N_TGT}" "${TARGET_START}" "${PX4_SITL_FRAME}" >"$PX4_BOOT_LOG" 2>&1
fi
PX4_RC=$?
set -e
if [ "$PX4_RC" -ne 0 ]; then
  echo "[ERROR] PX4 SITL 启动失败，日志: ${PX4_BOOT_LOG}"
  tail -40 "$PX4_BOOT_LOG" || true
  exit "$PX4_RC"
fi

sleep "${SCENE_SETUP_SETTLE_S:-2}"
echo "[sim] 设置 UE 场景"
SCENE_SETUP_LOG="$LOG_DIR/scene_setup.log"
set +e
python3 -u "$SIM_DIR/scene_setup.py" --scene "$SCENE" --config-dir "$ROOT_DIR/config" >"$SCENE_SETUP_LOG" 2>&1
SCENE_SETUP_RC=$?
set -e
if [ "$SCENE_SETUP_RC" -ne 0 ]; then
  echo "[WARN] UE 场景设置失败，日志: ${SCENE_SETUP_LOG}"
fi
if ! kill -0 "$RFLYSIM_PID" 2>/dev/null; then
  echo "[WARN] RflySim3D 在场景设置后退出，日志: ${RFLYSIM_LOG}"
fi

echo "[sim] 仿真平台启动完成"
