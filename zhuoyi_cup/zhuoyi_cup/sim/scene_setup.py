#!/usr/bin/env python3
"""设置 UE 场景显示和 3v3 吊舱链路。"""
import argparse
import os
import sys
import time
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(ROOT_DIR, "referee"),
              os.path.join(ROOT_DIR, "bin")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import scene
import scenario


DEFAULT_SAFETY_ZONE_TYPE_ID = 100100853
INTERCEPTOR_LABEL_RGB = [255, 0, 0]
TARGET_LABEL_RGB = [0, 0, 255]
LABEL_FONT_SIZE = 32
UE_MAX_FPS = 30


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    print(f"[scene_setup][{_now()}] {msg}", flush=True)


def _object_cfg(scene_cfg, key, defaults):
    return dict(defaults, **scene_cfg.get("scene_objects", {}).get(key, {}))


def _ue_transform(scene_cfg):
    cfg = scene_cfg.get("scene_objects", {}).get("ue_transform", {})
    return {
        "swap_xy": bool(cfg.get("swap_xy", False)),
        "x_sign": float(cfg.get("x_sign", 1.0)),
        "y_sign": float(cfg.get("y_sign", 1.0)),
        "x_offset": float(cfg.get("x_offset", 0.0)),
        "y_offset": float(cfg.get("y_offset", 0.0)),
    }


def _apply_transform(pos, transform):
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    if transform["swap_xy"]:
        x, y = y, x
    return [
        transform["x_sign"] * x + transform["x_offset"],
        transform["y_sign"] * y + transform["y_offset"],
        z,
    ]


def _send_safety_zone(ue, scene_cfg):
    safe = scenario.resolve_safe_zone(scene_cfg, scenario.master_seed_from_env())
    cfg = _object_cfg(scene_cfg, "safety_zone", {
        "type_id": DEFAULT_SAFETY_ZONE_TYPE_ID,
        "id": int(scene_cfg.get("competition", {}).get("safety_zone_copter_id", 100)),
        "z": 0.0,
        "height": 250.0,
        "color_rgba": [1.0, 0.0, 0.0, 0.5],
    })
    transform = _ue_transform(scene_cfg)
    pos = _apply_transform([float(safe["n"]), float(safe["e"]), float(cfg["z"])], transform)
    radius = float(safe.get("radius", cfg.get("radius", 30.0)))
    height = float(cfg.get("height", 250.0))
    color = [float(v) for v in cfg.get("color_rgba", [1.0, 0.0, 0.0, 0.5])]
    obj_id = int(cfg["id"])
    type_id = int(cfg.get("type_id", DEFAULT_SAFETY_ZONE_TYPE_ID))

    _log(
        f"safety_zone id={obj_id} type={type_id} "
        f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
        f"radius={radius:.2f} height={height:.2f}"
    )
    ue.sendUE4Pos(obj_id, type_id, 0, pos)
    time.sleep(0.2)
    ue.sendUE4ExtAct(obj_id, [
        radius, height, color[0], color[1], color[2], color[3],
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ])


def _send_team_labels(ue, scene_cfg, attempt, repeats):
    n_int = int(scene_cfg.get("interceptor", {}).get("count", 0))
    n_tgt = int(scene_cfg.get("target", {}).get("count", 0))
    ue.sendUE4Cmd(b"RflyChangeViewKeyCmd S 1")
    for cid in range(1, n_int + 1):
        ue.sendUE4LabelID(cid, str(cid), LABEL_FONT_SIZE, INTERCEPTOR_LABEL_RGB, -1)
    for cid in range(n_int + 1, n_int + n_tgt + 1):
        ue.sendUE4LabelID(cid, str(cid), LABEL_FONT_SIZE, TARGET_LABEL_RGB, -1)
    safety_id = int(scene_cfg.get("competition", {}).get("safety_zone_copter_id", 100))
    ue.sendUE4LabelID(safety_id, "", 1, [255, 255, 255], -1)
    _log(f"labels pass={attempt}/{repeats} interceptor=red target=blue")


def setup_visuals(scene_cfg, *, repeats=3, interval=0.75):
    repeats = int(repeats)
    _log(f"visual setup scene={scene_cfg.get('scene')}")
    try:
        import UE4CtrlAPI
        ue = UE4CtrlAPI.UE4CtrlAPI()
        ue.sendUE4Cmd(f"t.MaxFPS {UE_MAX_FPS}", -1)
        _log(f"UE max FPS set to {UE_MAX_FPS}")
        _send_safety_zone(ue, scene_cfg)
        for attempt in range(1, repeats + 1):
            if attempt > 1:
                time.sleep(interval)
            _send_team_labels(ue, scene_cfg, attempt, repeats)
        return 0
    except Exception as exc:
        _log(f"[ERROR] visual setup failed: {exc}")
        traceback.print_exc()
        return 1


def activate_gimbal(config_dir, *, start_id=1, config_name="Config_3v3.json"):
    old_cwd = os.getcwd()
    try:
        os.chdir(config_dir)
        import ReqCopterSim
        import VisionCaptureApi

        req = ReqCopterSim.ReqCopterSim()
        target_ip = req.getSimIpID(int(start_id))
        config_path = os.path.join(config_dir, config_name)
        _log(f"activate gimbal: CopterSim #{start_id} -> {target_ip}; config={config_path}")
        vis = VisionCaptureApi.VisionCaptureApi(target_ip)
        vis.jsonLoad(config_path)
        vis.jsonLoad(1, config_path)
        return 0
    except Exception as exc:
        _log(f"[ERROR] gimbal activation failed: {exc}")
        traceback.print_exc()
        return 1
    finally:
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser(description="设置 UE 场景显示和吊舱链路")
    parser.add_argument("--scene", default="3v3", choices=["3v3", "10v10"])
    parser.add_argument("--config-dir", default=os.path.join(ROOT_DIR, "config"))
    args = parser.parse_args()

    scene_cfg = scene.load_scene(args.scene)
    rc = setup_visuals(scene_cfg)

    if bool(scene_cfg.get("sensors", {}).get("gimbal", False)):
        start_id = int(os.environ.get("ROSTRANS_START_ID", "1"))
        rc |= activate_gimbal(args.config_dir, start_id=start_id, config_name=f"Config_{args.scene}.json")
    else:
        _log("gimbal activation skipped")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
