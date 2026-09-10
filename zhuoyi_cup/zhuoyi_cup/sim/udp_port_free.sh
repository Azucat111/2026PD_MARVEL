#!/bin/bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    exec sudo "$0" "$@"
fi

VERBOSE=0
if [ "${1:-}" = "--verbose" ] || [ "${1:-}" = "-v" ]; then
    VERBOSE=1
    shift
fi

num=${1:-1}
if [ -z "$num" ] || [ "$num" -le 0 ] 2>/dev/null; then
    echo "[port-free][ERROR] 飞机数量必须大于 0"
    exit 1
fi

declare -A UDP_PORTS=()
declare -A TCP_PORTS=()
declare -A PID_REASONS=()

add_udp() { UDP_PORTS["$1"]=1; }
add_tcp() { TCP_PORTS["$1"]=1; }

for port in 20005 20006 20007 20008 20009 20010 20011 14550; do
    add_udp "$port"
done

for ((i=0; i<num; i++)); do
    add_tcp $((4560 + i))
    add_udp $((20100 + i * 2))
    add_udp $((20100 + i * 2 + 1))
    add_udp $((30100 + i * 2))
    add_udp $((30100 + i * 2 + 1))
    add_udp $((16540 + i))
    add_udp $((17540 + i))
    add_udp $((18570 + i))
    add_udp $((6001 + i))
done

echo "[port-free] cleanup ROS/RflySim related processes ..."
pkill -f "roscore" 2>/dev/null || true
pkill -f "rosmaster" 2>/dev/null || true
pkill -f "rosout" 2>/dev/null || true
pkill -f "bt_ros" 2>/dev/null || true
pkill -f "ego_planner" 2>/dev/null || true
pkill -f "main.py" 2>/dev/null || true
pkill -f "det.py" 2>/dev/null || true
pkill -f "aruco_detect_node" 2>/dev/null || true

append_pid_reason() {
    local pid=$1
    local reason=$2
    local old="${PID_REASONS[$pid]-}"
    if [ -n "$old" ]; then
        PID_REASONS["$pid"]="${old},${reason}"
    else
        PID_REASONS["$pid"]="$reason"
    fi
}

extract_pids_from_line() {
    local line=$1
    local reason=$2
    local rest=$line
    local pid

    while [[ "$rest" =~ pid=([0-9]+) ]]; do
        pid="${BASH_REMATCH[1]}"
        append_pid_reason "$pid" "$reason"
        rest="${rest#*pid=$pid}"
    done
}

scan_udp_once() {
    local line port
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        for port in "${!UDP_PORTS[@]}"; do
            if [[ " $line " == *":$port "* ]]; then
                extract_pids_from_line "$line" "udp/$port"
                break
            fi
        done
    done < <(ss -H -lunp 2>/dev/null || true)
}

scan_tcp_once() {
    local line port
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        for port in "${!TCP_PORTS[@]}"; do
            if [[ " $line " == *":$port "* ]]; then
                extract_pids_from_line "$line" "tcp/$port"
                break
            fi
        done
    done < <(ss -H -tlnp 2>/dev/null || true)
}

scan_target_ports() {
    PID_REASONS=()
    scan_udp_once
    scan_tcp_once
}

echo "[port-free] target ports: udp=${#UDP_PORTS[@]} tcp=${#TCP_PORTS[@]} vehicles=$num"
scan_target_ports

if [ "${#PID_REASONS[@]}" -eq 0 ]; then
    echo "[port-free] no target ports occupied."
else
    for pid in "${!PID_REASONS[@]}"; do
        echo "[port-free] kill PID $pid (${PID_REASONS[$pid]})"
        kill -9 "$pid" 2>/dev/null || echo "[port-free][WARN] failed to kill PID $pid"
    done
fi

sleep 0.1
scan_target_ports
if [ "${#PID_REASONS[@]}" -eq 0 ]; then
    echo "[port-free] done."
else
    echo "[port-free][WARN] some ports are still occupied:"
    for pid in "${!PID_REASONS[@]}"; do
        echo "  PID $pid (${PID_REASONS[$pid]})"
    done
fi

if [ "$VERBOSE" = "1" ]; then
    echo "[port-free] udp ports: ${!UDP_PORTS[*]}"
    echo "[port-free] tcp ports: ${!TCP_PORTS[*]}"
fi
