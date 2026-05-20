"""Open-loop eval of the G1 pick-apple policy on a GR00T 1.6 server.

Schema vs the 1.5 script (eval_g1_left_hand_pick_apple_openloop.py):
  * Server uses GR00T's own MsgSerializer (msgpack + np.save), NOT
    msgpack_numpy. A msgpack_numpy client receives raw ndarrays back as
    dict blobs the server cannot decode.
  * Observation uses Gr00tSimPolicyWrapper FLAT keys:
        video.ego_view           (1, 1, H, W, 3) uint8
        state.left_leg           (1, 1, 6)  float32  ← we don't have legs,
        state.right_leg          (1, 1, 6)  float32     zero-pad
        state.waist              (1, 1, 3)  float32  ← zero-pad
        state.left_arm           (1, 1, 7)  float32  ← dataset[0:7]
        state.right_arm          (1, 1, 7)  float32  ← dataset[7:14]
        state.left_hand          (1, 1, 7)  float32  ← dataset[14:21] (dex3 L)
        state.right_hand         (1, 1, 7)  float32  ← dataset[21:28] (dex3 R)
        annotation.human.task_description: list[str], len 1
  * Action chunk horizon = 30 (was 16). Returned keys are PREFIXED with
    `action.`:
        action.left_arm        (1, 30, 7)  RELATIVE (single-anchor)
        action.right_arm       (1, 30, 7)  RELATIVE
        action.left_hand       (1, 30, 7)  ABSOLUTE
        action.right_hand      (1, 30, 7)  ABSOLUTE
        action.waist           (1, 30, 3)  ABSOLUTE
        action.base_height_command (1, 30, 1)
        action.navigate_command    (1, 30, 3)

Relative→absolute (same convention as 1.5): target[k] = state[t] + chunk[k]
(NOT cumsum, NOT per-step delta). Hand is absolute, use directly.

The dataset only has 28-dim upper-body state. Legs / waist are zero-padded
when sent to the model; the model should ignore them for upper-body tasks
but its predictions there are not meaningful for this eval.

Pre-req: GR00T 1.6 server running, e.g.
  python gr00t/eval/run_gr00t_server.py \
    --model-path cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 --use-sim-policy-wrapper \
    --device cuda:0 --host 0.0.0.0 --port 5555
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import msgpack
import mujoco
import numpy as np
import pandas as pd
import zmq
from torchcodec.decoders import VideoDecoder


DATASET_ROOT = Path("/home/anhnx10/work/unitree_lerobot/data_converted/pick_and_put_v4_converted")
G1_XML = Path("/home/anhnx10/work/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.xml")
OUTPUTS_ROOT = Path("/home/anhnx10/work/unitree_lerobot/outputs")
DEFAULT_VIDEO_DIR = OUTPUTS_ROOT / "openloop-videos-n16"
DEFAULT_PLOT_DIR = OUTPUTS_ROOT / "openloop_plots_n16"
DEFAULT_TASK = "pick apple and put in the plate"  # model: GR00T-N1.6-G1-PnPAppleToPlate

# Upper-body joint slices in the dataset's 28-dim state.
LA_SLICE = slice(0, 7)
RA_SLICE = slice(7, 14)
LH_SLICE = slice(14, 21)
RH_SLICE = slice(21, 28)

# Whole-body dims expected by the n1.6 UNITREE_G1 embodiment.
DIM_LEG = 6
DIM_WAIST = 3
DIM_ARM = 7
DIM_HAND = 7
CHUNK_H = 30

# MuJoCo joint ordering for the 28-dim dataset state (matches the 1.5 script
# so we can reuse its renderer).
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
LA_NAMES = ["shoulder-pitch", "shoulder-roll", "shoulder-yaw",
            "elbow", "wrist-roll", "wrist-pitch", "wrist-yaw"]
LH_NAMES = ["thumb-0", "thumb-1", "thumb-2",
            "middle-0", "middle-1", "index-0", "index-1"]


# ---- GR00T MsgSerializer (matches server_client.py exactly) ---------------
def _encode_custom(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_custom(obj):
    if not isinstance(obj, dict):
        return obj
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    if "__ModalityConfig_class__" in obj:
        return obj.get("as_json", obj)
    return obj


class Gr00tClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.ctx = zmq.Context()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: dict | None = None) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self.sock.send(msgpack.packb(req, default=_encode_custom))
        resp = msgpack.unpackb(self.sock.recv(), object_hook=_decode_custom)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except zmq.error.Again:
            self._connect()
            return False

    def get_modality_config(self):
        return self._call("get_modality_config")

    def get_action(self, observation: dict) -> tuple[dict, Any]:
        resp = self._call("get_action", {"observation": observation, "options": None})
        # ReplayPolicy and Gr00tPolicy return (action_dict, info)
        if isinstance(resp, (list, tuple)) and len(resp) == 2:
            return resp[0], resp[1]
        # Some configurations return the action dict directly.
        return resp, None


def build_observation(cam_high_rgb: np.ndarray,
                      state28: np.ndarray,
                      task_text: str) -> dict:
    """Pack a single-frame observation in the FLAT format the
    Gr00tSimPolicyWrapper expects.

    Legs / waist are zero-padded because the dataset only has upper-body.
    Right arm + right hand are taken from the dataset (the policy was trained
    on full-body states, so we send the real measurements we DO have rather
    than zeros).
    """
    s = state28.astype(np.float32)
    return {
        "video.ego_view":    cam_high_rgb[None, None],
        "state.left_leg":    np.zeros((1, 1, DIM_LEG), dtype=np.float32),
        "state.right_leg":   np.zeros((1, 1, DIM_LEG), dtype=np.float32),
        "state.waist":       np.zeros((1, 1, DIM_WAIST), dtype=np.float32),
        "state.left_arm":    s[LA_SLICE][None, None],
        "state.right_arm":   s[RA_SLICE][None, None],
        "state.left_hand":   s[LH_SLICE][None, None],
        "state.right_hand":  s[RH_SLICE][None, None],
        "annotation.human.task_description": [task_text],
    }


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
                         "alpha=1.0 disables smoothing.")
    ap.add_argument("--task", default=None,
                    help=f"Override task description sent to the model. "
                         f"Default = '{DEFAULT_TASK}' (matches the deployed PnPAppleToPlate checkpoint, "
                         f"NOT the dataset task text which may say 'box').")
    ap.add_argument("--save-video", default=None,
                    help="Default: outputs/openloop-videos-n16/openloop_g1_n16_ep{episode}.mp4")
    ap.add_argument("--out-dir", default=None,
                    help="Default: outputs/openloop_plots_n16")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video is None:
        DEFAULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        save_video = str(DEFAULT_VIDEO_DIR / f"openloop_g1_n16_ep{args.episode}.mp4")
    else:
        save_video = args.save_video
        Path(save_video).parent.mkdir(parents=True, exist_ok=True)

    # ---- Dataset --------------------------------------------------------
    ep_path = DATASET_ROOT / "data/chunk-000" / f"episode_{args.episode:06d}.parquet"
    df = pd.read_parquet(ep_path)
    with open(DATASET_ROOT / "meta/tasks.jsonl") as f:
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in f}
    dataset_task = tasks[int(df["task_index"].iloc[0])]
    sent_task = args.task if args.task else DEFAULT_TASK
    n_total = len(df)
    N = n_total if args.max_steps is None else min(args.max_steps, n_total)
    print(f"[dataset] episode={args.episode}  frames={n_total}  using={N}")
    print(f"[task]    dataset='{dataset_task}'")
    print(f"[task]    sent_to_model='{sent_task}'")

    cam_high = VideoDecoder(str(DATASET_ROOT / "videos/chunk-000/observation.images.cam_high"
                                / f"episode_{args.episode:06d}.mp4"))

    states_28 = np.stack(
        [np.asarray(s, dtype=np.float32) for s in df["observation.state"].iloc[:N]], axis=0)
    actions_28 = np.stack(
        [np.asarray(a, dtype=np.float32) for a in df["action"].iloc[:N]], axis=0)

    # ---- Server ---------------------------------------------------------
    client = Gr00tClient(args.host, args.port, timeout_ms=60000)
    if not client.ping():
        raise SystemExit(f"GR00T server not reachable on {args.host}:{args.port}")
    mc = client.get_modality_config()
    print(f"[server] ping ok  modality keys: {list(mc.keys())}")
    action_keys = mc["action"]["modality_keys"] if isinstance(mc["action"], dict) else mc["action"].modality_keys
    delta_idx = mc["action"]["delta_indices"] if isinstance(mc["action"], dict) else mc["action"].delta_indices
    print(f"[server] action modality_keys = {action_keys}  chunk_H = {len(delta_idx)}")
    if len(delta_idx) != CHUNK_H:
        print(f"[warn] server chunk_H={len(delta_idx)} != script default {CHUNK_H}")

    # ---- MuJoCo (only for visualization) -------------------------------
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

    # ---- Video writer (3-panel: cam_high + MuJoCo GT + MuJoCo Pred) ---
    # No wrist cam is sent to the model here, so we drop that panel and
    # make the cam_high panel bigger.
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
    pred_la = np.full((N, DIM_ARM), np.nan, dtype=np.float32)
    pred_lh = np.full((N, DIM_HAND), np.nan, dtype=np.float32)
    raw_la  = np.full((N, DIM_ARM), np.nan, dtype=np.float32)
    raw_lh  = np.full((N, DIM_HAND), np.nan, dtype=np.float32)
    pred_state28 = np.full((N, 28), np.nan, dtype=np.float32)

    t_query_total = 0.0
    n_queries = 0
    alpha = float(args.ema_alpha)
    if not (0.0 < alpha <= 1.0):
        raise SystemExit(f"--ema-alpha must be in (0, 1], got {alpha}")
    ema_la = None
    ema_lh = None
    print(f"[ema] alpha={alpha:.2f} ({'smoothing on' if alpha < 1.0 else 'disabled'})")

    for t in range(0, N, args.stride):
        img_h_full = cam_high[t].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
        st = states_28[t]

        obs = build_observation(img_h_full, st, sent_task)
        tq = time.perf_counter()
        action_dict, _ = client.get_action(obs)
        t_query_total += time.perf_counter() - tq
        n_queries += 1

        # Action dict keys are prefixed "action.X" with shape (B, H, D).
        chunk_la_raw = np.asarray(action_dict["action.left_arm"],  dtype=np.float32)
        chunk_lh_raw = np.asarray(action_dict["action.left_hand"], dtype=np.float32)
        # Squeeze the batch dim if present.
        chunk_la = chunk_la_raw[0] if chunk_la_raw.ndim == 3 else chunk_la_raw  # (H, 7)
        chunk_lh = chunk_lh_raw[0] if chunk_lh_raw.ndim == 3 else chunk_lh_raw  # (H, 7)

        # N1.6+ server (including Gr00tSimPolicyWrapper) already calls
        # state_action_processor.unapply_action server-side, so both arm and
        # hand come back ABSOLUTE. Do NOT add state again — that was the
        # N1.5 convention. See [[project-n17-server-unapplies]] memory.
        abs_la_chunk = chunk_la
        abs_lh_chunk = chunk_lh

        for k in range(args.stride):
            tk = t + k
            if tk >= N or k >= abs_la_chunk.shape[0]:
                break
            r_la = abs_la_chunk[k]
            r_lh = abs_lh_chunk[k]
            raw_la[tk] = r_la
            raw_lh[tk] = r_lh
            if ema_la is None:
                ema_la = r_la.copy()
                ema_lh = r_lh.copy()
            else:
                ema_la = alpha * r_la + (1.0 - alpha) * ema_la
                ema_lh = alpha * r_lh + (1.0 - alpha) * ema_lh
            pred_la[tk] = ema_la
            pred_lh[tk] = ema_lh
            full = states_28[tk].copy()
            full[LA_SLICE] = ema_la
            full[LH_SLICE] = ema_lh
            pred_state28[tk] = full

        if n_queries % 20 == 0:
            print(f"  step {t+1}/{N}  queries={n_queries}  avg={1000*t_query_total/n_queries:.1f} ms")

    print(f"[done] queries={n_queries}  avg latency={1000*t_query_total/n_queries:.1f} ms")

    # ---- Render every frame to video -----------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    for t in range(N):
        img_h = cam_high[t].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)
        img_h = cv2.resize(img_h, (panel_w, panel_h))
        img_h_bgr = cv2.cvtColor(img_h, cv2.COLOR_RGB2BGR)

        apply_state(model_gt, data_gt, actions_28[t],  qpos_idx)
        pred_full = pred_state28[t] if not np.isnan(pred_state28[t, 0]) else states_28[t]
        apply_state(model_pr, data_pr, pred_full, qpos_idx)
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

        # Top row: cam_high (left) + a small text panel could go right,
        # but to keep this simple we mirror cam_high on both top panels
        # for visual symmetry with the 1.5 script.
        canvas[y_top:y_top + panel_h, x0:x1] = img_h_bgr
        canvas[y_top:y_top + panel_h, x2:x3] = img_h_bgr
        canvas[y_mid:y_mid + panel_h, x0:x1] = frame_gt
        canvas[y_mid:y_mid + panel_h, x2:x3] = frame_pr

        cv2.putText(canvas, "ego_view (cam_high)",  (x0 + 6, header - 8), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ego_view (cam_high)",  (x2 + 6, header - 8), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "G1 GROUND TRUTH",      (x0 + 6, y_mid - 6),  font, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "G1 PREDICTION (n1.6)", (x2 + 6, y_mid - 6),  font, 0.55, (160, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"open-loop  n1.6  episode {args.episode}  step {t+1}/{N}",
                    (gap, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"sent='{sent_task}'", (gap, canvas_h - 10), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
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
        fig.suptitle(f"{title}  —  episode {args.episode}  ({N} steps, GR00T 1.6)", fontsize=12)
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

    plot_group(gt_la, pred_la, LA_NAMES, "Left arm joint target (rad)",
               f"openloop_n16_ep{args.episode}_left_arm.png")
    plot_group(gt_lh, pred_lh, LH_NAMES, "Left hand joint target (rad)",
               f"openloop_n16_ep{args.episode}_left_hand.png")

    valid = ~np.isnan(pred_la[:, 0])
    err_la = pred_la[valid] - gt_la[valid]
    err_lh = pred_lh[valid] - gt_lh[valid]
    raw_err_la = raw_la[valid] - gt_la[valid]
    raw_err_lh = raw_lh[valid] - gt_lh[valid]

    def jitter(x):
        d = np.abs(np.diff(x, axis=0))
        return d.mean()

    rmse_la_raw = np.sqrt((raw_err_la ** 2).mean(axis=0))
    rmse_lh_raw = np.sqrt((raw_err_lh ** 2).mean(axis=0))
    rmse_la_ema = np.sqrt((err_la ** 2).mean(axis=0))
    rmse_lh_ema = np.sqrt((err_lh ** 2).mean(axis=0))

    lines = []
    lines.append(f"[config] GR00T 1.6  episode={args.episode}  N={N}  stride={args.stride}  ema_alpha={alpha:.2f}")
    lines.append(f"[task]   sent='{sent_task}'   dataset='{dataset_task}'")
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
    lines.append(f"  ground truth : left_arm={jitter(gt_la[valid]):.5f}  left_hand={jitter(gt_lh[valid]):.5f}")
    lines.append(f"  raw pred     : left_arm={jitter(raw_la[valid]):.5f}  left_hand={jitter(raw_lh[valid]):.5f}")
    lines.append(f"  ema pred     : left_arm={jitter(pred_la[valid]):.5f}  left_hand={jitter(pred_lh[valid]):.5f}")

    report = "\n".join(lines)
    print("\n" + report)
    log_path = out_dir / f"openloop_n16_ep{args.episode}_alpha{alpha:.2f}.log"
    log_path.write_text(report + "\n")
    print(f"\n[log] saved {log_path}")


if __name__ == "__main__":
    main()
