"""Open-loop evaluation of the G1 left-arm pick-apple GR00T policy.

For every step t of the chosen episode we feed the dataset's ground-truth
observation (cam_high + cam_left_wrist images, left-arm/left-hand state)
to the policy server, take the first step of the predicted action chunk,
and compare it against the dataset's ground-truth action.

Outputs:
  * <save-video>            — 4-panel MP4 (cam_high, cam_wrist, MuJoCo GT,
                              MuJoCo Pred) with per-step joint error printed.
  * <out-dir>/openloop_ep{N}_left_arm.png
  * <out-dir>/openloop_ep{N}_left_hand.png
  * RMSE / L2 summary on stdout.

The arm/hand convention:
  * left_arm  action is RELATIVE  → integrate cumulatively on top of the
    state at query time to obtain absolute joint targets, then compare to
    the dataset's absolute action.
  * left_hand action is ABSOLUTE  → compare directly.

Pre-req: GR00T policy server already running, env `unitree_lerobot_clean`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import msgpack_numpy as mnp
import mujoco
import numpy as np
import pandas as pd
import zmq
from torchcodec.decoders import VideoDecoder


DATASET_ROOT = Path("/home/anhnx10/work/unitree_lerobot/data_converted/pick_and_put_v4_converted")
G1_XML = Path("/home/anhnx10/work/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.xml")
OUTPUTS_ROOT = Path("/home/anhnx10/work/unitree_lerobot/outputs")
DEFAULT_VIDEO_DIR = OUTPUTS_ROOT / "openloop-videos"
DEFAULT_PLOT_DIR = OUTPUTS_ROOT / "openloop_plots"

ARM_LEFT = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
ARM_RIGHT = [n.replace("left_", "right_") for n in ARM_LEFT]
HAND_LEFT = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
HAND_RIGHT = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]
STATE28_JOINTS = ARM_LEFT + ARM_RIGHT + HAND_LEFT + HAND_RIGHT
LA_SLICE = slice(0, 7)
LH_SLICE = slice(14, 21)
LA_NAMES = ["shoulder-pitch", "shoulder-roll", "shoulder-yaw",
            "elbow", "wrist-roll", "wrist-pitch", "wrist-yaw"]
LH_NAMES = ["thumb-0", "thumb-1", "thumb-2",
            "middle-0", "middle-1", "index-0", "index-1"]


def decode_modality_config(blob: dict) -> dict:
    out = {}
    for k, v in blob.items():
        out[k] = v["as_json"] if isinstance(v, dict) and v.get("__ModalityConfig__") else v
    return out


class RawPolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.ctx = zmq.Context()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(mnp.packb(req))
        resp = mnp.unpackb(self.sock.recv(), raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except zmq.error.Again:
            self._connect()
            return False

    def get_modality_config(self):
        return decode_modality_config(self._call("get_modality_config"))

    def get_action(self, observation):
        resp = self._call("get_action", {"observation": observation, "options": None})
        return resp[0], resp[1]


def make_state_to_qpos(model: mujoco.MjModel) -> np.ndarray:
    out = []
    for jn in STATE28_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"joint not found: {jn}")
        out.append(int(model.jnt_qposadr[jid]))
    return np.asarray(out, dtype=np.int64)


def init_qpos(model, data):
    data.qpos[:] = 0.0
    data.qpos[0:3] = (0.0, 0.0, 0.793)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)


def apply_state(model, data, state28, qpos_idx):
    data.qpos[qpos_idx] = state28
    mujoco.mj_forward(model, data)


def build_camera():
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0.25, 0.0, 1.0])
    cam.distance = 1.5
    cam.azimuth = 150.0
    cam.elevation = -10.0
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=5)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="Query the policy every N frames (set 8 to mimic action_horizon=8).")
    ap.add_argument("--ema-alpha", type=float, default=0.7,
                    help="EMA smoothing on predicted action: a_t = alpha*a_pred + (1-alpha)*a_{t-1}. "
                         "alpha=1.0 disables smoothing. Recommended 0.6-0.8.")
    ap.add_argument("--save-video", default=None,
                    help="Default: outputs/openloop-videos/openloop_g1_ep{episode}.mp4")
    ap.add_argument("--out-dir", default=None,
                    help="Default: outputs/openloop_plots")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video is None:
        DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        save_video = str(DEFAULT_VIDEO_DIR / f"openloop_g1_ep{args.episode}.mp4")
    else:
        save_video = args.save_video
        Path(save_video).parent.mkdir(parents=True, exist_ok=True)

    # ---- Dataset --------------------------------------------------------
    ep_path = DATASET_ROOT / "data/chunk-000" / f"episode_{args.episode:06d}.parquet"
    df = pd.read_parquet(ep_path)
    with open(DATASET_ROOT / "meta/tasks.jsonl") as f:
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in f}
    task_text = tasks[int(df["task_index"].iloc[0])]
    n_total = len(df)
    N = n_total if args.max_steps is None else min(args.max_steps, n_total)
    print(f"[dataset] episode={args.episode}  frames={n_total}  using={N}  task='{task_text}'")

    cam_high = VideoDecoder(str(DATASET_ROOT / "videos/chunk-000/observation.images.cam_high"
                                / f"episode_{args.episode:06d}.mp4"))
    cam_wrist = VideoDecoder(str(DATASET_ROOT / "videos/chunk-000/observation.images.cam_left_wrist"
                                 / f"episode_{args.episode:06d}.mp4"))

    states_28 = np.stack(
        [np.asarray(s, dtype=np.float32) for s in df["observation.state"].iloc[:N]], axis=0)
    actions_28 = np.stack(
        [np.asarray(a, dtype=np.float32) for a in df["action"].iloc[:N]], axis=0)

    # ---- Server ---------------------------------------------------------
    client = RawPolicyClient(args.host, args.port, timeout_ms=60000)
    if not client.ping():
        raise SystemExit(f"GR00T server not reachable on {args.host}:{args.port}")
    print("[server] ping ok  modality:", list(client.get_modality_config().keys()))

    # ---- MuJoCo ---------------------------------------------------------
    model_gt = mujoco.MjModel.from_xml_path(str(G1_XML))
    model_pr = mujoco.MjModel.from_xml_path(str(G1_XML))
    data_gt, data_pr = mujoco.MjData(model_gt), mujoco.MjData(model_pr)
    qpos_idx = make_state_to_qpos(model_gt)
    init_qpos(model_gt, data_gt)
    init_qpos(model_pr, data_pr)
    Hm, Wm = 360, 480
    renderer_gt = mujoco.Renderer(model_gt, Hm, Wm)
    renderer_pr = mujoco.Renderer(model_pr, Hm, Wm)
    cam = build_camera()

    # ---- Video writer (4 panel) -----------------------------------------
    # Cameras (top row) keep their native 640x480; resize to 480x360 so the
    # canvas is balanced with the MuJoCo renders (480x360).
    panel_w, panel_h = 480, 360
    gap = 8
    header = 50
    footer = 32
    canvas_w = panel_w * 2 + gap * 3
    canvas_h = header + panel_h * 2 + gap * 3 + footer
    writer = cv2.VideoWriter(save_video,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             15.0, (canvas_w, canvas_h))

    # ---- Storage --------------------------------------------------------
    pred_la_first = np.full((N, 7), np.nan, dtype=np.float32)
    pred_lh_first = np.full((N, 7), np.nan, dtype=np.float32)
    raw_la_first  = np.full((N, 7), np.nan, dtype=np.float32)
    raw_lh_first  = np.full((N, 7), np.nan, dtype=np.float32)
    pred_state28 = np.full((N, 28), np.nan, dtype=np.float32)

    t_query_total = 0.0
    n_queries = 0
    alpha = float(args.ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise SystemExit(f"--ema-alpha must be in (0, 1], got {alpha}")
    ema_la = None  # type: ignore[var-annotated]
    ema_lh = None  # type: ignore[var-annotated]
    print(f"[ema] alpha={alpha:.2f} ({'smoothing on' if alpha < 1.0 else 'disabled'})")

    for t in range(0, N, args.stride):
        img_h_full = cam_high[t].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
        img_w_full = cam_wrist[t].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
        st = states_28[t]

        obs = {
            "video": {
                "cam_high":       img_h_full[None, None],
                "cam_left_wrist": img_w_full[None, None],
            },
            "state": {
                "left_arm":  st[LA_SLICE][None, None].astype(np.float32),
                "left_hand": st[LH_SLICE][None, None].astype(np.float32),
            },
            "language": {"annotation.human.task_description": [[task_text]]},
        }
        tq = time.perf_counter()
        action_dict, _ = client.get_action(obs)
        t_query_total += time.perf_counter() - tq
        n_queries += 1

        chunk_la = np.asarray(action_dict["left_arm"],  dtype=np.float32).reshape(-1, 7)
        chunk_lh = np.asarray(action_dict["left_hand"], dtype=np.float32).reshape(-1, 7)
        abs_la_chunk = st[LA_SLICE][None, :] + np.cumsum(chunk_la, axis=0)   # (16, 7) absolute
        abs_lh_chunk = chunk_lh                                              # (16, 7) absolute

        # Fill the stride-window from this query, applying EMA smoothing
        # across the per-step action stream: a_t = alpha*a_pred + (1-alpha)*a_{t-1}.
        for k in range(args.stride):
            tk = t + k
            if tk >= N or k >= abs_la_chunk.shape[0]:
                break
            raw_la = abs_la_chunk[k]
            raw_lh = abs_lh_chunk[k]
            raw_la_first[tk] = raw_la
            raw_lh_first[tk] = raw_lh
            if ema_la is None:
                ema_la = raw_la.copy()
                ema_lh = raw_lh.copy()
            else:
                ema_la = alpha * raw_la + (1.0 - alpha) * ema_la
                ema_lh = alpha * raw_lh + (1.0 - alpha) * ema_lh
            pred_la_first[tk] = ema_la
            pred_lh_first[tk] = ema_lh
            full = states_28[tk].copy()
            full[LA_SLICE] = ema_la
            full[LH_SLICE] = ema_lh
            pred_state28[tk] = full

        if n_queries % 20 == 0:
            print(f"  step {t+1}/{N}  queries={n_queries}  avg={1000*t_query_total/n_queries:.1f} ms")

    print(f"[done] queries={n_queries}  avg latency={1000*t_query_total/n_queries:.1f} ms")

    # ---- Render every frame to the 4-panel video ----------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in range(N):
        img_h = cam_high[t].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
        img_w = cam_wrist[t].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
        img_h = cv2.resize(img_h, (panel_w, panel_h))
        img_w = cv2.resize(img_w, (panel_w, panel_h))
        img_h_bgr = cv2.cvtColor(img_h, cv2.COLOR_RGB2BGR)
        img_w_bgr = cv2.cvtColor(img_w, cv2.COLOR_RGB2BGR)

        # MuJoCo GT pose (uses dataset GT action as the target qpos for clarity).
        # We use action[t] rather than state[t] so each frame shows the *command*
        # that drives the next motion — matches the predicted target unit.
        apply_state(model_gt, data_gt, actions_28[t],  qpos_idx)
        pred_full = pred_state28[t] if not np.isnan(pred_state28[t, 0]) else states_28[t]
        apply_state(model_pr, data_pr, pred_full,      qpos_idx)
        renderer_gt.update_scene(data_gt, camera=cam)
        renderer_pr.update_scene(data_pr, camera=cam)
        frame_gt = cv2.cvtColor(renderer_gt.render(), cv2.COLOR_RGB2BGR)
        frame_pr = cv2.cvtColor(renderer_pr.render(), cv2.COLOR_RGB2BGR)
        frame_gt = cv2.resize(frame_gt, (panel_w, panel_h))
        frame_pr = cv2.resize(frame_pr, (panel_w, panel_h))

        canvas = np.full((canvas_h, canvas_w, 3), 25, dtype=np.uint8)

        x0, x1 = gap, gap + panel_w
        x2, x3 = gap * 2 + panel_w, gap * 2 + panel_w * 2
        y_top = header
        y_mid = header + panel_h + gap

        canvas[y_top:y_top + panel_h, x0:x1] = img_h_bgr
        canvas[y_top:y_top + panel_h, x2:x3] = img_w_bgr
        canvas[y_mid:y_mid + panel_h, x0:x1] = frame_gt
        canvas[y_mid:y_mid + panel_h, x2:x3] = frame_pr

        cv2.putText(canvas, "cam_high",        (x0 + 6, header - 8), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "cam_left_wrist",  (x2 + 6, header - 8), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "G1 GROUND TRUTH", (x0 + 6, y_mid - 6),  font, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "G1 PREDICTION",   (x2 + 6, y_mid - 6),  font, 0.55, (160, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"open-loop  episode {args.episode}  step {t+1}/{N}",
                    (gap, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, task_text, (gap, canvas_h - 10), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(canvas)

        if args.show:
            cv2.imshow("openloop", canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[done] video: {save_video}")

    # ---- Per-joint plots -----------------------------------------------
    gt_la = actions_28[:, LA_SLICE]
    gt_lh = actions_28[:, LH_SLICE]
    ts = np.arange(N)

    def plot_group(gt, pred, names, title, fname):
        n = gt.shape[1]
        fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n), sharex=True)
        fig.suptitle(f"{title}  —  episode {args.episode}  ({N} steps)", fontsize=12)
        for i in range(n):
            ax = axes[i] if n > 1 else axes
            ax.plot(ts, gt[:, i], color="#1f77b4", linewidth=1.4, label="ground truth")
            ax.plot(ts, pred[:, i], color="#d62728", linewidth=1.0,
                    linestyle="--", label="predicted (open-loop)")
            ax.set_ylabel(names[i], fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("timestep")
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / fname, dpi=110)
        plt.close(fig)
        print(f"  saved {out_dir / fname}")

    plot_group(gt_la, pred_la_first, LA_NAMES, "Left arm joint target (rad)",
               f"openloop_ep{args.episode}_left_arm.png")
    plot_group(gt_lh, pred_lh_first, LH_NAMES, "Left hand joint target (rad)",
               f"openloop_ep{args.episode}_left_hand.png")

    valid = ~np.isnan(pred_la_first[:, 0])
    err_la = pred_la_first[valid] - gt_la[valid]
    err_lh = pred_lh_first[valid] - gt_lh[valid]
    raw_err_la = raw_la_first[valid] - gt_la[valid]
    raw_err_lh = raw_lh_first[valid] - gt_lh[valid]

    def jitter(x):
        # mean abs diff between consecutive frames, per-joint then averaged
        d = np.abs(np.diff(x, axis=0))
        return d.mean()

    rmse_la_raw  = np.sqrt((raw_err_la ** 2).mean(axis=0))
    rmse_lh_raw  = np.sqrt((raw_err_lh ** 2).mean(axis=0))
    rmse_la_ema  = np.sqrt((err_la ** 2).mean(axis=0))
    rmse_lh_ema  = np.sqrt((err_lh ** 2).mean(axis=0))
    j_la_gt   = jitter(gt_la[valid])
    j_lh_gt   = jitter(gt_lh[valid])
    j_la_raw  = jitter(raw_la_first[valid])
    j_lh_raw  = jitter(raw_lh_first[valid])
    j_la_ema  = jitter(pred_la_first[valid])
    j_lh_ema  = jitter(pred_lh_first[valid])

    lines = []
    lines.append(f"[config] episode={args.episode}  N={N}  stride={args.stride}  ema_alpha={alpha:.2f}")
    lines.append("\n[summary] per-joint RMSE (rad)  — raw (no smoothing)")
    lines.append("  left_arm : " + "  ".join(f"{v:.4f}" for v in rmse_la_raw))
    lines.append("  left_hand: " + "  ".join(f"{v:.4f}" for v in rmse_lh_raw))
    lines.append("\n[summary] per-joint RMSE (rad)  — EMA smoothed")
    lines.append("  left_arm : " + "  ".join(f"{v:.4f}" for v in rmse_la_ema))
    lines.append("  left_hand: " + "  ".join(f"{v:.4f}" for v in rmse_lh_ema))
    lines.append("\n[summary] L2 mean error (rad)")
    lines.append(f"  raw : left_arm={np.linalg.norm(raw_err_la, axis=1).mean():.4f}  "
                 f"left_hand={np.linalg.norm(raw_err_lh, axis=1).mean():.4f}")
    lines.append(f"  ema : left_arm={np.linalg.norm(err_la,     axis=1).mean():.4f}  "
                 f"left_hand={np.linalg.norm(err_lh,     axis=1).mean():.4f}")
    lines.append("\n[summary] jitter (mean |Δaction| between consecutive frames, rad)")
    lines.append(f"  ground truth : left_arm={j_la_gt:.5f}  left_hand={j_lh_gt:.5f}")
    lines.append(f"  raw pred     : left_arm={j_la_raw:.5f}  left_hand={j_lh_raw:.5f}")
    lines.append(f"  ema pred     : left_arm={j_la_ema:.5f}  left_hand={j_lh_ema:.5f}")
    lines.append(f"  jitter reduction: left_arm={100*(1-j_la_ema/max(j_la_raw,1e-9)):.1f}%  "
                 f"left_hand={100*(1-j_lh_ema/max(j_lh_raw,1e-9)):.1f}%")

    report = "\n".join(lines)
    print("\n" + report)
    log_path = out_dir / f"openloop_ep{args.episode}_alpha{alpha:.2f}.log"
    log_path.write_text(report + "\n")
    print(f"\n[log] saved {log_path}")


if __name__ == "__main__":
    main()
