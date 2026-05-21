"""Real-life evaluation of the G1 left-arm pick-apple GR00T policy.

Closed-loop deployment on a physical G1:
  * pull cam_high + cam_left_wrist images from the image_server,
  * read current left-arm joint state from the arm controller,
  * read current left-hand state from EE shared memory,
  * send observation to the GR00T policy server (ZMQ),
  * apply EMA smoothing on the predicted action (alpha=0.5 default,
    chosen from the openloop sweep in eval_g1_left_hand_pick_apple_openloop.py),
  * command the left arm via arm_ctrl + arm_ik and the left hand via EE
    shared memory at --frequency Hz.

Conventions (GR00T N1.7 server, embodiment new_embodiment):
  * Both left_arm and left_hand action keys come back ABSOLUTE because the
    N1.7 server calls state_action_processor.unapply_action() inside
    decode_action() (processing_gr00t_n1d7.py:312). Use chunks directly;
    do NOT add state on the client.
  * Right arm is held at its initial pose; right hand at zeros.

History:
  * v1 (wrong): treated arm as `state + cumsum(chunk)` — overshoots ~10x.
  * v2 (wrong, N1.5 era): treated arm as absolute when chunk was relative —
    wrong scale, wrist drifted near zero.
  * v3 (correct for N1.5): `target[k] = state_at_query + chunk[k]`, single-
    reference relative add. Server returned raw relative chunks.
  * v4 (correct for N1.7, current): server already unapplies relative→abs,
    use chunk directly: `target[k] = chunk[k]`. Verified 2026-05-20: at
    ep5 frame 0, chunk_la[0] ≈ dataset action[0] ≈ state[0] (delta ~0.01 rad)
    confirming absolute, not relative.

Pre-req:
  * GR00T policy server already running (default tcp://127.0.0.1:5555).
  * image_server.py running on the G1 development unit.
  * Env: unitree_lerobot_clean.
"""

from __future__ import annotations

import argparse
import collections
import math
import select
import sys
import termios
import threading
import time
import tty
from typing import Any

import msgpack_numpy as mnp
import numpy as np
import zmq

from unitree_lerobot.eval_robot.make_robot import (
    process_images_and_observations,
    setup_image_client,
    setup_robot_interface,
)

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


# ---- Conventions copied from the openloop script --------------------------
LA_DIM = 7   # left arm joints
RA_DIM = 7   # right arm joints
LH_DIM = 7   # left hand joints (dex3)
DEFAULT_TASK = "pick apple and put in the box"

# ---- Physical-arm mapping (THIS UNIT'S SDK ENUM IS MIRRORED) -------------
# Empirical verification 2026-05-21 (multiple slew tests, debug log
# correlation with the wrist-camera landmark):
#   - With standard mapping (LEFT_ARM_SLICE = slice(0, 7)), applying
#     LEFT_ARM_TRAIN_INIT (elbow=90°) to motor[15-21] physically bent the
#     arm WITHOUT the wrist camera (= robot's RIGHT arm).
#   - The model's "left_arm" output therefore needs to land on motor[22-28]
#     (which is wired to robot's LEFT arm on this unit) for the camera-
#     bearing arm to do the picking motion.
# Conclusion: the SDK enum G1_29_JointArmIndex is mirrored on this hardware
# revision (motor[15-21] labeled "kLeftXxx" is actually wired to physical
# right). We compensate at the script layer with these slice constants.
LEFT_ARM_SLICE  = slice(0, LA_DIM)                   # picking arm = motor[15-21] = arm WITH cam
RIGHT_ARM_SLICE = slice(LA_DIM, LA_DIM + RA_DIM)     # idle arm    = motor[22-28] = arm WITHOUT cam

# ---- Training-data init pose ---------------------------------------------
# Mean of frame-0 left-arm state across 50 training episodes of
# pick_and_put_v4_converted. The model expects to start the trajectory
# from roughly this pose; if the robot starts elsewhere, the policy
# predicts a move BACK toward this distribution (which is what causes
# the "arm drops down" complaint when the operator holds the arm
# perpendicular to the body before pressing 's').
LEFT_ARM_TRAIN_INIT = np.array([
    -0.18,   # shoulder-pitch  (slightly tilted back / up)
    +0.04,   # shoulder-roll
    +0.17,   # shoulder-yaw
    +0.00,   # elbow
    +0.04,   # wrist-roll
    -0.52,   # wrist-pitch
    +0.25,   # wrist-yaw
], dtype=np.float64)

# Mean frame-0 left_hand state across 30 training episodes. Several Dex3
# joints idle at NEGATIVE values (knuckles 0, 3, 5, 6); commanding them at
# 0 puts the hand in an OOD pose and the policy then immediately commands
# them back to their natural idle, which looks like the hand "shrugging".
LEFT_HAND_TRAIN_INIT = np.array([
    -0.46,
    +0.80,
    +0.04,
    +0.00,
    -0.08,
    -0.49,
    -0.09,
], dtype=np.float64)

# Right-arm rest pose. The user's task is LEFT-handed; the model does not
# consume right_arm (new_embodiment state only includes left_arm +
# left_hand), so the right arm pose is chosen purely for physical
# convenience.
#
# Values = MEAN across all 102,462 training frames (261 episodes) — the
# right arm is essentially HELD STILL throughout training (std ≤ 0.075 per
# joint), so this is the pose the operator parked it in. Matching it
# guarantees the right arm sits exactly where the model was trained to
# expect (relevant only for cam_high visual distribution, since the model
# doesn't observe right_arm state directly).
RIGHT_ARM_REST = np.array([
    -0.13,   # shoulder-pitch
    -0.03,   # shoulder-roll
    +0.00,   # shoulder-yaw
    +1.30,   # elbow             (operator-verified "arm straight down" pose)
    -0.01,   # wrist-roll
    +0.14,   # wrist-pitch
    -0.01,   # wrist-yaw
], dtype=np.float64)

# ---- Safety limits for the LEFT arm only (right side is untouched) -------
# Joint position limits (rad), per Unitree G1 spec. Tightened slightly inside
# the mechanical bounds so we never command end-stops.
#   ordering matches LA_NAMES in the openloop script:
#   shoulder-pitch, shoulder-roll, shoulder-yaw, elbow,
#   wrist-roll, wrist-pitch, wrist-yaw
LEFT_ARM_LIMITS_LOW  = np.array([-3.05, -1.55, -2.60, -1.04, -1.95, -1.60, -1.60], dtype=np.float64)
LEFT_ARM_LIMITS_HIGH = np.array([ 2.65,  2.20,  2.60,  2.07,  1.95,  1.60,  1.60], dtype=np.float64)

# Dex3 left-hand joint range (rad). Per-joint inspection of training data
# shows several knuckles operate in NEGATIVE territory (down to ~-1.74),
# so the previous [-0.05, +1.75] clamp was muzzling the fingers (they
# couldn't actually close on objects whose grip needs a negative target).
# Picked with ~0.3 rad margin on top of the observed training extremes.
LEFT_HAND_LIMITS_LOW  = np.full(LH_DIM, -2.00, dtype=np.float64)
LEFT_HAND_LIMITS_HIGH = np.full(LH_DIM,  1.50, dtype=np.float64)

# Per-step velocity cap (rad/step). Tuned for the default 20 Hz loop:
#   0.027 rad/step ≈ 0.54 rad/s on the arm (gentle, safe for first deploy);
#   0.067 rad/step ≈ 1.33 rad/s on the hand (enough for grasp open/close).
# If you raise --frequency to 30 Hz, scale these caps by 20/30 = 0.67 so the
# per-second velocity stays the same.
DEFAULT_ARM_DELTA_CAP  = 0.027
DEFAULT_HAND_DELTA_CAP = 0.067

# Emergency stop: if predicted target is farther than this from the current
# measured state, abort. Catches NaN, mis-normalized actions, or wild bias.
DEFAULT_ARM_DEVIATION_ABORT = 0.6   # rad per joint
DEFAULT_HAND_DEVIATION_ABORT = 1.0  # rad per joint


# ---- GR00T policy ZMQ client ---------------------------------------------
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

    def _call(self, endpoint: str, data: dict | None = None):
        req: dict[str, Any] = {"endpoint": endpoint}
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

    def get_action(self, observation: dict) -> tuple[dict, Any]:
        resp = self._call("get_action", {"observation": observation, "options": None})
        return resp[0], resp[1]


def build_observation(cam_high_rgb: np.ndarray,
                      cam_left_wrist_rgb: np.ndarray,
                      left_arm_state: np.ndarray,
                      left_hand_state: np.ndarray,
                      task_text: str) -> dict:
    """Pack a single-frame observation in the format the GR00T server expects.

    Image shape: (1, 1, H, W, 3) uint8 RGB.
    State shape: (1, 1, D) float32.
    """
    # Same observation format as eval_g1_left_hand_pick_apple_openloop.py
    # (validated end-to-end). The server's new_embodiment modality config
    # registers video.modality_keys = ['cam_high', 'cam_left_wrist'] — only
    # those two are required.
    return {
        "video": {
            "cam_high":       cam_high_rgb[None, None],
            "cam_left_wrist": cam_left_wrist_rgb[None, None],
        },
        "state": {
            "left_arm":  left_arm_state.astype(np.float32)[None, None],
            "left_hand": left_hand_state.astype(np.float32)[None, None],
        },
        "language": {"annotation.human.task_description": [[task_text]]},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Robot / image config (mirrors eval_g1.py's EvalRealConfig fields).
    ap.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    ap.add_argument("--motion", default="upper_body",
                    help="motion_mode passed to the arm controller.")
    ap.add_argument("--ee", default="dex3", help="end effector type")
    ap.add_argument("--image-host", default="127.0.0.1")
    ap.add_argument("--frequency", type=float, default=20.0,
                    help="Control loop frequency in Hz. Default 20 Hz makes the "
                         "robot execute the same predicted action chunk ~1.5x "
                         "slower than the training rate (30 Hz) — safer for "
                         "early real-life testing. Bump back to 30 Hz once "
                         "behavior is verified to match the training distribution.")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--base-type", default="legs")

    # GR00T server
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=5555)
    ap.add_argument("--task", default=DEFAULT_TASK)

    # Action chunk handling — async + ACT-style temporal ensemble
    ap.add_argument("--ensemble-tau-ticks", type=float, default=4.0,
                    help="Decay constant (in CONTROL TICKS) for temporal "
                         "ensemble weights w = exp(-age/tau). ACT defaults to "
                         "~10 ticks at 50 Hz (≈200 ms); 4 ticks at 20 Hz keeps "
                         "the same wall-clock half-life. Lower = react faster "
                         "to fresh chunks, higher = smoother but laggier.")
    ap.add_argument("--ensemble-capacity", type=int, default=8,
                    help="Max number of recent chunks kept in the ring buffer "
                         "for temporal ensembling. Must be >= ceil(chunk_H / "
                         "expected_query_stride) so every executable tick is "
                         "covered by at least one chunk.")
    ap.add_argument("--query-stride-ticks", type=int, default=1,
                    help="Minimum control ticks between two model queries. 1 "
                         "= fire as fast as the inference worker can keep up "
                         "(usually ~3 ticks @ 20 Hz given 150 ms latency). "
                         "Raise to throttle GPU load at the cost of less "
                         "overlap in the ensemble.")
    ap.add_argument("--hold-hz", type=float, default=50.0,
                    help="Frequency of the daemon thread that resends the last "
                         "command to the arm + EE controllers. Keeps the motor "
                         "from sagging during inference latency or init pauses.")
    ap.add_argument("--chunk-stride", type=int, default=1,
                    help="DEPRECATED under temporal ensemble — no longer used. "
                         "Kept for CLI compatibility; emits a warning if != 1.")
    ap.add_argument("--prefetch-threshold", type=int, default=0,
                    help="DEPRECATED under temporal ensemble — no longer used. "
                         "Kept for CLI compatibility; emits a warning if != 0.")
    ap.add_argument("--play-delay-ticks", type=int, default=0,
                    help="DEPRECATED under temporal ensemble — the ensemble "
                         "weights handle latency masking. Kept for CLI "
                         "compatibility; emits a warning if != 0.")

    # Smoothing — alpha=0.5 was the best result from the openloop sweep.
    ap.add_argument("--ema-alpha", type=float, default=0.5,
                    help="EMA: a_t = alpha*a_pred + (1-alpha)*a_{t-1}. "
                         "1.0 disables smoothing.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Stop after this many control steps. Default: until Ctrl-C.")

    # ---- Safety knobs (LEFT arm + LEFT hand only) ----------------------
    ap.add_argument("--arm-delta-cap", type=float, default=DEFAULT_ARM_DELTA_CAP,
                    help="Max per-step change for any left-arm joint (rad/step).")
    ap.add_argument("--hand-delta-cap", type=float, default=DEFAULT_HAND_DELTA_CAP,
                    help="Max per-step change for any left-hand joint (rad/step).")
    ap.add_argument("--arm-abort", type=float, default=DEFAULT_ARM_DEVIATION_ABORT,
                    help="Emergency-stop threshold: |target - current| per arm joint (rad).")
    ap.add_argument("--hand-abort", type=float, default=DEFAULT_HAND_DEVIATION_ABORT,
                    help="Emergency-stop threshold: |target - current| per hand joint (rad).")
    ap.add_argument("--ramp-steps", type=int, default=15,
                    help="Soft-start: linearly interpolate from current pose to "
                         "first predicted target over this many control steps.")
    ap.add_argument("--disable-safety", action="store_true",
                    help="DANGEROUS: disable joint-limit + delta-cap + abort checks.")
    ap.add_argument("--init-pose", choices=["current", "training"], default="training",
                    help="'training' (default): move left arm to the training-data "
                         "frame-0 pose before the loop starts — required so the "
                         "policy sees in-distribution state. 'current': use whatever "
                         "pose the robot is in (matches the old behavior; only use "
                         "if you have manually placed the arm at a training-like pose).")
    ap.add_argument("--init-pose-secs", type=float, default=2.0,
                    help="Time (seconds) over which the left arm slews from its "
                         "current pose to LEFT_ARM_TRAIN_INIT before the closed-loop "
                         "starts. Slower = safer.")
    return ap.parse_args()


def safety_clamp(target: np.ndarray, current: np.ndarray,
                 low: np.ndarray, high: np.ndarray,
                 delta_cap: float, abort: float,
                 label: str) -> tuple[np.ndarray, bool, str]:
    """Apply joint limit clamp + per-step velocity cap.

    Returns (clamped_target, abort_flag, reason). abort_flag is True if the
    raw target is so far off that we should refuse to move.
    """
    if not np.all(np.isfinite(target)):
        return current.copy(), True, f"{label}: non-finite values in target {target}"
    dev = np.abs(target - current)
    if np.any(dev > abort):
        bad = int(np.argmax(dev))
        return current.copy(), True, (
            f"{label}: joint {bad} deviation {dev[bad]:.3f} rad > abort {abort:.3f}"
        )
    # Velocity cap: limit Δ to ±delta_cap per joint.
    delta = np.clip(target - current, -delta_cap, delta_cap)
    capped = current + delta
    # Joint position clamp.
    clamped = np.clip(capped, low, high)
    return clamped, False, ""


class ControlState:
    """Thread-shared command target. Updated by main loop / slew, read by
    the hold-pose daemon which is the SOLE writer to the arm + EE hardware.

    Centralizing hardware writes in one thread (a) gives a continuous command
    stream so the G1 controller never sees a >100ms gap (avoids gravity sag),
    and (b) removes any chance of two threads racing on ctrl_dual_arm().
    """

    def __init__(self, arm_full_cmd: np.ndarray, arm_ik,
                 left_hand_cmd: np.ndarray, right_hand_cmd: np.ndarray):
        self._lock = threading.Lock()
        self._arm = np.asarray(arm_full_cmd, dtype=np.float64).copy()
        self._tau = np.asarray(arm_ik.solve_tau(self._arm), dtype=np.float64).copy()
        self._lh  = np.asarray(left_hand_cmd, dtype=np.float64).copy()
        self._rh  = np.asarray(right_hand_cmd, dtype=np.float64).copy()
        self._arm_ik = arm_ik

    def set_arm(self, full_cmd: np.ndarray) -> None:
        full_cmd = np.asarray(full_cmd, dtype=np.float64)
        tau = np.asarray(self._arm_ik.solve_tau(full_cmd), dtype=np.float64)
        with self._lock:
            self._arm = full_cmd
            self._tau = tau

    def set_hands(self, left: np.ndarray, right: np.ndarray) -> None:
        left  = np.asarray(left,  dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        with self._lock:
            self._lh = left
            self._rh = right

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            return (self._arm.copy(), self._tau.copy(),
                    self._lh.copy(),  self._rh.copy())


def hold_pose_loop(arm_ctrl, ee_shared, ctrl_state: ControlState,
                   stop_event: threading.Event, hz: float) -> None:
    """Daemon: resend the latest command to arm + EE at fixed rate.

    Runs faster than the control loop so even during a slow tick (model query,
    OS jitter, image stall) the motor still gets a fresh command every ~20ms.
    """
    period = 1.0 / float(hz)
    while not stop_event.is_set():
        t = time.perf_counter()
        try:
            arm_cmd, arm_tau, lh, rh = ctrl_state.snapshot()
            arm_ctrl.ctrl_dual_arm(arm_cmd, arm_tau)
            if ee_shared:
                with ee_shared["lock"]:
                    ee_shared["left"][:]  = lh.tolist()
                    ee_shared["right"][:] = rh.tolist()
        except Exception as e:
            logger_mp.error(f"[hold] {e}")
        time.sleep(max(0.0, period - (time.perf_counter() - t)))


class ChunkBuffer:
    """Single-slot async pipeline for GR00T action chunks.

    Main loop:  fills `pending_*`, sets `request_event`.
    Worker:     drains pending, runs policy.get_action, fills `ready_*`,
                sets `ready_event`.
    Main loop:  swaps `ready_*` into its local cache, clears `ready_event`.

    At most one query is in flight; the next request only fires when both
    events are clear, so the worker is never starved nor doubled up.

    Both arm and hand chunks are ABSOLUTE joint targets (N1.7 server
    unapplies relative→absolute server-side).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.request_event = threading.Event()
        self.ready_event   = threading.Event()
        # request payload (main -> worker)
        self.pending_obs:    dict | None = None
        self.pending_t_obs:  float = 0.0
        # response payload (worker -> main): absolute (H, D) chunks
        self.ready_la_abs:  np.ndarray | None = None  # (H, LA_DIM)
        self.ready_lh_abs:  np.ndarray | None = None  # (H, LH_DIM)
        self.ready_t_obs:   float = 0.0

    def submit(self, obs: dict, t_obs: float) -> None:
        with self._lock:
            self.pending_obs   = obs
            self.pending_t_obs = t_obs
        self.request_event.set()

    def take_pending(self) -> tuple[dict, float] | None:
        with self._lock:
            obs, t_obs = self.pending_obs, self.pending_t_obs
            self.pending_obs = None
        if obs is None:
            return None
        return obs, t_obs

    def publish(self, la_abs: np.ndarray, lh_abs: np.ndarray, t_obs: float) -> None:
        with self._lock:
            self.ready_la_abs = la_abs
            self.ready_lh_abs = lh_abs
            self.ready_t_obs  = t_obs
        self.ready_event.set()

    def take_ready(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        with self._lock:
            la, lh, t = self.ready_la_abs, self.ready_lh_abs, self.ready_t_obs
            self.ready_la_abs = None
            self.ready_lh_abs = None
        if la is None:
            return None
        return la, lh, t


class ChunkRing:
    """ACT-style temporal ensemble over the K most recent chunks.

    Each chunk c stores absolute joint targets and the wall-clock time
    the obs was captured:
      * `la_abs` — left_arm absolute targets, shape (H, LA_DIM).
      * `lh_abs` — left_hand absolute targets, shape (H, LH_DIM).
      * `t_obs`  — perf_counter when the obs was captured.

    To compute the target for absolute time `t_now`:
      * For each chunk c that still covers t_now (idx in [0, H-1]):
          target_la = c.la_abs[idx]
          target_lh = c.lh_abs[idx]
      * Weighted-average all contributors with w = exp(-age/tau_ticks),
        where age is in control ticks.
    """

    def __init__(self, capacity: int, period: float, tau_ticks: float):
        self._chunks: collections.deque = collections.deque(maxlen=max(1, capacity))
        self._period = float(period)
        self._tau = float(tau_ticks)

    def push(self, la_abs: np.ndarray, lh_abs: np.ndarray, t_obs: float) -> None:
        self._chunks.append({
            "la_abs": la_abs.astype(np.float32, copy=False),
            "lh_abs": lh_abs.astype(np.float32, copy=False),
            "t_obs":  float(t_obs),
        })

    def __len__(self) -> int:
        return len(self._chunks)

    def newest_age_ticks(self, t_now: float) -> float | None:
        if not self._chunks:
            return None
        return (t_now - self._chunks[-1]["t_obs"]) / self._period

    def ensemble(self, t_now: float) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        """Weighted average of all covering chunks' absolute targets at t_now.

        Returns (la_target, lh_target, n_contributing). If no chunk covers
        t_now (all expired or none yet pushed), returns (None, None, 0).
        """
        la_acc: np.ndarray | None = None
        lh_acc: np.ndarray | None = None
        w_total = 0.0
        n = 0
        for c in self._chunks:
            age = (t_now - c["t_obs"]) / self._period
            if age < 0:
                continue
            # Training delta_indices = [0..H-1], so chunk[0] = action at obs-time,
            # chunk[k] = action k ticks after obs. Apply chunk[round(age)] to align
            # with the validated openloop convention (where chunk[0] is rendered at
            # frame t = t_obs). Previously this had a `-1` offset that introduced a
            # constant 1-tick lag (~50 ms @ 20 Hz) in the trajectory playback.
            idx = int(round(age))
            H = c["la_abs"].shape[0]
            if idx < 0 or idx >= H:
                continue
            w = math.exp(-age / self._tau) if self._tau > 0 else 1.0
            la = c["la_abs"][idx]
            lh = c["lh_abs"][idx]
            if la_acc is None:
                la_acc = w * la
                lh_acc = w * lh
            else:
                la_acc = la_acc + w * la
                lh_acc = lh_acc + w * lh
            w_total += w
            n += 1
        if la_acc is None or w_total == 0.0:
            return None, None, 0
        return la_acc / w_total, lh_acc / w_total, n


def inference_worker(policy: RawPolicyClient, buffer: ChunkBuffer,
                     stop_event: threading.Event) -> None:
    """Daemon: pull a request, query the policy, publish the chunk.

    N1.7 server returns ABSOLUTE targets for both arm and hand
    (state_action_processor.unapply_action runs server-side, see
    processing_gr00t_n1d7.py:312). Same convention as
    eval_g1_left_hand_pick_apple_openloop.py.
    """
    while not stop_event.is_set():
        if not buffer.request_event.wait(timeout=0.1):
            continue
        buffer.request_event.clear()
        payload = buffer.take_pending()
        if payload is None:
            continue
        obs, t_obs = payload
        try:
            action_dict, _ = policy.get_action(obs)
        except Exception as e:
            logger_mp.error(f"[inference] {e}")
            continue
        la_abs = np.asarray(action_dict["left_arm"],  dtype=np.float32).reshape(-1, LA_DIM)
        lh_abs = np.asarray(action_dict["left_hand"], dtype=np.float32).reshape(-1, LH_DIM)
        buffer.publish(la_abs, lh_abs, t_obs)


def stdin_quit_listener(quit_event: threading.Event) -> None:
    """Daemon: set quit_event when user presses 'q' on stdin (no Enter needed).

    Switches stdin to cbreak mode (line-buffering off + echo off) so a single
    keystroke triggers the slew-back. Polls via select.select with a 100 ms
    timeout so the loop also wakes up to check `quit_event` and exit cleanly
    when main shuts down.

    Caller (main) is responsible for restoring the original terminal mode
    in its finally block — this listener changes tty flags but does NOT
    restore them on exit, because daemon threads can be killed abruptly
    before their own finally runs, leaving the terminal broken.

    If stdin isn't a TTY (pipe / redirect / IDE without pty), the listener
    no-ops and falls back to Ctrl+C as the only abort path.
    """
    if not sys.stdin.isatty():
        logger_mp.warning("[input] stdin is not a TTY; 'q' hotkey disabled — "
                          "use Ctrl+C to abort.")
        return
    try:
        tty.setcbreak(sys.stdin.fileno())
    except Exception as e:
        logger_mp.error(f"[input-listener] failed to enter cbreak mode: {e}")
        return
    while not quit_event.is_set():
        r, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not r:
            continue
        try:
            ch = sys.stdin.read(1)
        except Exception:
            break
        if not ch:                    # EOF
            break
        if ch.lower() == "q":
            logger_mp.info("[input] 'q' pressed — initiating slew back to init pose")
            quit_event.set()
            break


def slew_back_to_init(arm_ctrl, ee_shared, ctrl_state: "ControlState",
                      init_full_arm: np.ndarray,
                      left_hand_init: np.ndarray,
                      right_hand_init: np.ndarray,
                      period: float,
                      secs: float = 2.0) -> None:
    """Linearly slew the dual arm + both hands from the current sensed pose
    back to the start-of-deployment init pose via ctrl_state.

    Writes go through ctrl_state — the hold_pose_loop daemon picks them up
    at 50 Hz and forwards to arm_ctrl + ee_shared, so no extra UDP socket
    is opened and there's no risk of conflicting with the live hold stream.
    Exceptions are logged but never re-raised; finally already runs.
    """
    try:
        current_full_arm = np.asarray(arm_ctrl.get_current_dual_arm_q(),
                                      dtype=np.float64)
        with ee_shared["lock"]:
            current_lh = np.array(ee_shared["state"][:len(left_hand_init)],
                                  dtype=np.float64)
        n_steps = max(1, int(round(secs / period)))
        logger_mp.info(f"[shutdown] slewing back to init over {n_steps} ticks "
                       f"({secs:.1f}s @ {1.0/period:.1f} Hz)")
        for i in range(1, n_steps + 1):
            t = time.perf_counter()
            w = i / float(n_steps)
            cmd_arm = (1.0 - w) * current_full_arm + w * init_full_arm
            cmd_lh  = (1.0 - w) * current_lh        + w * left_hand_init
            try:
                ctrl_state.set_arm(cmd_arm)
                ctrl_state.set_hands(cmd_lh, right_hand_init)
            except Exception as e:
                logger_mp.error(f"[shutdown-slew] tick {i}: {e}")
            time.sleep(max(0.0, period - (time.perf_counter() - t)))
        ctrl_state.set_arm(init_full_arm)
        ctrl_state.set_hands(left_hand_init, right_hand_init)
        logger_mp.info("[shutdown] back at init pose")
    except Exception as e:
        logger_mp.error(f"[shutdown-slew] failed: {e}")


def main():
    args = parse_args()

    if not (0.0 < args.ema_alpha <= 1.0):
        raise SystemExit(f"--ema-alpha must be in (0, 1], got {args.ema_alpha}")
    alpha = float(args.ema_alpha)

    for deprecated_name, val, default in (
        ("--chunk-stride", args.chunk_stride, 1),
        ("--prefetch-threshold", args.prefetch_threshold, 0),
        ("--play-delay-ticks", args.play_delay_ticks, 0),
    ):
        if val != default:
            logger_mp.warning(
                f"{deprecated_name}={val} is deprecated under the temporal "
                f"ensemble architecture and is ignored. Use --ensemble-tau-ticks "
                f"and --query-stride-ticks instead.")

    if args.ensemble_tau_ticks <= 0:
        raise SystemExit(f"--ensemble-tau-ticks must be > 0, got {args.ensemble_tau_ticks}")
    if args.ensemble_capacity < 1:
        raise SystemExit(f"--ensemble-capacity must be >= 1, got {args.ensemble_capacity}")
    if args.query_stride_ticks < 1:
        raise SystemExit(f"--query-stride-ticks must be >= 1, got {args.query_stride_ticks}")

    # ---- Robot interface FIRST -----------------------------------------
    # arm_ctrl's publish thread starts publishing as soon as it's
    # constructed; we want the hold-pose daemon online before any slow
    # network IO (policy ping / image handshake) can stretch out the
    # window where the arm is left without commands matching the sensed
    # pose. Controller-side fix (robot_arm.py: q_target seeded with the
    # current sensed pose at construction) already prevents the drop, but
    # this reordering is defense-in-depth.
    robot_interface = setup_robot_interface(args)
    arm_ctrl     = robot_interface["arm_ctrl"]
    arm_ik       = robot_interface["arm_ik"]
    ee_shared    = robot_interface["ee_shared_mem"]
    arm_dof      = int(robot_interface["arm_dof"])     # dual arm dof, e.g. 14
    ee_dof       = int(robot_interface["ee_dof"])      # per-side EE dof, 7 for dex3

    if arm_dof < 2 * LA_DIM:
        raise SystemExit(f"arm_dof={arm_dof} too small for left-arm slice of {LA_DIM}")
    if ee_dof != LH_DIM:
        raise SystemExit(f"ee_dof={ee_dof} does not match left_hand dim {LH_DIM}")

    # ---- Initial pose: snap left arm + right arm to training-data init -
    # We must put the left arm into a pose the policy actually saw during
    # training before we let it close the loop. The right arm is also slewed
    # to its training rest pose (folded, tucked at side) so it doesn't poke
    # into the cam_high view and feed the model OOD pixels.
    current_full_arm = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
    init_full_arm = current_full_arm.copy()
    if args.init_pose == "training":
        init_full_arm[LEFT_ARM_SLICE] = LEFT_ARM_TRAIN_INIT
        init_full_arm[RIGHT_ARM_SLICE] = RIGHT_ARM_REST
    logger_mp.info(
        f"[init] mode={args.init_pose}\n"
        f"  current_left_arm  ={current_full_arm[LEFT_ARM_SLICE].tolist()}\n"
        f"  target_left_arm   ={init_full_arm[LEFT_ARM_SLICE].tolist()}\n"
        f"  current_right_arm ={current_full_arm[RIGHT_ARM_SLICE].tolist()}\n"
        f"  target_right_arm  ={init_full_arm[RIGHT_ARM_SLICE].tolist()}"
    )

    # ---- Threading setup -----------------------------------------------
    # Hold-pose daemon must run BEFORE the slew so the motor is fed a
    # continuous command stream all the way: slew → prompt wait → loop →
    # inference gaps → shutdown slew. The control loop never touches the
    # hardware directly; it only updates ctrl_state.
    # Read the actual current left_hand pose so the slew doesn't start from
    # `0` (OOD — see LEFT_HAND_TRAIN_INIT comment).
    with ee_shared["lock"]:
        full_ee_state_init = np.array(ee_shared["state"][:], dtype=np.float64)
    current_left_hand  = full_ee_state_init[:LH_DIM].copy()
    right_hand_zero    = np.zeros(LH_DIM, dtype=np.float64)
    ctrl_state = ControlState(current_full_arm, arm_ik, current_left_hand, right_hand_zero)

    stop_event = threading.Event()

    hold_thread = threading.Thread(
        target=hold_pose_loop,
        args=(arm_ctrl, ee_shared, ctrl_state, stop_event, args.hold_hz),
        name="hold-pose", daemon=True,
    )
    # Listener-related: created here so the finally block can always
    # safely reference them, but listener thread + cbreak mode are only
    # activated AFTER the user presses 's' (so input() above runs in
    # canonical mode).
    quit_event = threading.Event()
    _old_termios = None
    listener_thread: threading.Thread | None = None
    infer_started = False
    infer_thread: threading.Thread | None = None

    period = 1.0 / float(args.frequency)
    n_queries = 0
    t_query_total = 0.0
    step = 0
    end_lh = (LEFT_HAND_TRAIN_INIT.copy() if args.init_pose == "training"
              else current_left_hand.copy())

    try:
        # Start hold thread now so motors are continuously fed from this
        # point until shutdown (no sag during the slew or the prompt wait).
        # Crucially this runs BEFORE the slow policy / image client setup,
        # so the arm is held at its sensed pose during those network IOs.
        hold_thread.start()
        logger_mp.info(f"[threads] hold@{args.hold_hz:.0f}Hz started")

        # ---- GR00T server + image client (slow, network IO) ----------
        # Done after hold_thread starts so any latency here doesn't leave
        # the arm under the controller's default (sensed-pose-seeded)
        # publish without the higher-frequency hold reinforcement.
        policy = RawPolicyClient(args.policy_host, args.policy_port, timeout_ms=60000)
        if not policy.ping():
            raise SystemExit(f"GR00T server not reachable on {args.policy_host}:{args.policy_port}")
        logger_mp.info(f"[server] ping ok at {args.policy_host}:{args.policy_port}")

        image_client, image_config = setup_image_client(args)

        chunk_buffer = ChunkBuffer()
        infer_thread = threading.Thread(
            target=inference_worker,
            args=(policy, chunk_buffer, stop_event),
            name="gr00t-infer", daemon=True,
        )

        # ---- Slew to preparation pose BEFORE the 's' prompt ----------
        # User sees the robot physically move into the training-init pose
        # so they can verify visually (cam_high frame matches training
        # distribution, arms in correct posture) BEFORE committing to
        # closed-loop. Abort with Ctrl+C if anything looks wrong.
        n_init_steps = max(1, int(round(args.init_pose_secs * args.frequency)))
        start_left  = current_full_arm[LEFT_ARM_SLICE].copy()
        end_left    = init_full_arm[LEFT_ARM_SLICE].copy()
        start_right = current_full_arm[RIGHT_ARM_SLICE].copy()
        end_right   = init_full_arm[RIGHT_ARM_SLICE].copy()
        start_lh    = current_left_hand.copy()
        logger_mp.info(
            f"[init] slewing left+right arm + left hand over {n_init_steps} steps "
            f"({args.init_pose_secs:.1f}s @ {args.frequency:.1f}Hz)\n"
            f"  left_arm  : {start_left.tolist()} -> {end_left.tolist()}\n"
            f"  right_arm : {start_right.tolist()} -> {end_right.tolist()}\n"
            f"  left_hand : {start_lh.tolist()} -> {end_lh.tolist()}"
        )
        for i in range(1, n_init_steps + 1):
            t = time.perf_counter()
            w = i / float(n_init_steps)
            cmd = init_full_arm.copy()
            cmd[LEFT_ARM_SLICE] = (1.0 - w) * start_left + w * end_left
            cmd[RIGHT_ARM_SLICE] = (1.0 - w) * start_right + w * end_right
            ctrl_state.set_arm(cmd)
            cur_lh = (1.0 - w) * start_lh + w * end_lh
            ctrl_state.set_hands(cur_lh, right_hand_zero)
            time.sleep(max(0.0, period - (time.perf_counter() - t)))
        ctrl_state.set_arm(init_full_arm)
        ctrl_state.set_hands(end_lh, right_hand_zero)
        # Longer settle so both arms actually reach their commanded positions
        # before inference starts. With the previous 0.3s settle, an arm that
        # started far from RIGHT_ARM_REST (e.g. elbow=1.36 before slew → 0
        # commanded) may still be drifting toward its target when inference
        # begins, producing apparent "movement" on the idle arm.
        time.sleep(1.5)
        # Log post-settle sensed arm positions to confirm both arms reached
        # their targets. Large mismatch with init_full_arm = motor not done
        # tracking yet → consider increasing init_pose_secs.
        post_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float64)
        logger_mp.info(
            f"[init] post-settle sensed:\n"
            f"  left_arm  ={post_q[LEFT_ARM_SLICE].tolist()}\n"
            f"  right_arm ={post_q[RIGHT_ARM_SLICE].tolist()}\n"
            f"  target was LEFT={init_full_arm[LEFT_ARM_SLICE].tolist()},\n"
            f"             RIGHT={init_full_arm[RIGHT_ARM_SLICE].tolist()}"
        )
        logger_mp.info("[init] robot now at preparation pose — ready for prompt")

        # ---- Prompt AFTER slew --------------------------------------
        user_input = input("Robot at preparation pose. Enter 's' to start "
                           "closed-loop GR00T evaluation (once running, "
                           "press 'q' to slew back to init): ")
        if user_input.lower() != "s":
            logger_mp.info("aborted by user")
            return    # finally cleans up hold_thread; slew_back is a no-op (already at init)

        # ---- Activate raw stdin listener + start inference worker ---
        if sys.stdin.isatty():
            try:
                _old_termios = termios.tcgetattr(sys.stdin.fileno())
            except Exception as e:
                logger_mp.warning(f"[input] cannot read termios state: {e}")
        listener_thread = threading.Thread(
            target=stdin_quit_listener, args=(quit_event,),
            name="stdin-quit", daemon=True,
        )
        listener_thread.start()
        infer_thread.start()
        infer_started = True
        logger_mp.info("[threads] gr00t-infer started")

        # ---- EMA + temporal-ensemble state ----------------------------
        ema_la: np.ndarray | None = None
        ema_lh: np.ndarray | None = None
        ring = ChunkRing(
            capacity=int(args.ensemble_capacity),
            period=period,
            tau_ticks=float(args.ensemble_tau_ticks),
        )
        inference_busy: bool = False
        last_query_step = -10**9   # ensure first prefetch fires immediately

        logger_mp.info(
            f"[run] control loop @ {args.frequency:.1f}Hz  alpha={alpha:.2f}  "
            f"ensemble_tau_ticks={args.ensemble_tau_ticks}  "
            f"ensemble_capacity={args.ensemble_capacity}  "
            f"query_stride_ticks={args.query_stride_ticks}"
        )

        # ---- Bootstrap first chunk ------------------------------------
        # Fire the first query NOW so we have something to play when the
        # loop starts. The hold thread keeps the arm at init_full_arm
        # during this wait — no sag.
        def _capture_obs() -> tuple[dict | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            observation, current_arm_q = process_images_and_observations(
                image_client, image_config, arm_ctrl
            )
            if current_arm_q is None:
                return None, None, None, None
            current_arm_q = np.asarray(current_arm_q, dtype=np.float64)
            left_arm_state = current_arm_q[LEFT_ARM_SLICE].astype(np.float32)
            with ee_shared["lock"]:
                full_ee_state = np.array(ee_shared["state"][:], dtype=np.float64)
            left_hand_state = full_ee_state[:LH_DIM].astype(np.float32)
            cam_high_t = observation.get("observation.images.cam_high")
            if cam_high_t is None:
                cam_high_t = observation.get("observation.images.cam_left_high")
            cam_wrist_t = observation.get("observation.images.cam_left_wrist", None)
            if cam_high_t is None or cam_wrist_t is None:
                return None, current_arm_q, left_arm_state, full_ee_state
            cam_high_rgb  = cam_high_t.numpy().astype(np.uint8)
            cam_wrist_rgb = cam_wrist_t.numpy().astype(np.uint8)
            obs = build_observation(
                cam_high_rgb, cam_wrist_rgb,
                left_arm_state, left_hand_state, args.task,
            )
            return obs, current_arm_q, left_arm_state, full_ee_state

        bootstrap_deadline = time.perf_counter() + 5.0
        first_t_obs = 0.0
        while time.perf_counter() < bootstrap_deadline:
            obs, current_arm_q, left_arm_state, _ = _capture_obs()
            if obs is not None:
                first_t_obs = time.perf_counter()
                chunk_buffer.submit(obs, first_t_obs)
                inference_busy = True
                n_queries += 1
                break
            time.sleep(0.05)
        else:
            raise SystemExit("[bootstrap] failed to capture obs for first query within 5s")

        if not chunk_buffer.ready_event.wait(timeout=10.0):
            raise SystemExit("[bootstrap] GR00T server did not return first chunk within 10s")
        ready = chunk_buffer.take_ready()
        chunk_buffer.ready_event.clear()
        inference_busy = False
        if ready is None:
            raise SystemExit("[bootstrap] ChunkBuffer.take_ready returned None")
        boot_la, boot_lh, boot_t_obs = ready
        ring.push(boot_la, boot_lh, boot_t_obs)
        t_query_total += time.perf_counter() - boot_t_obs
        chunk_H = boot_la.shape[0]
        logger_mp.info(f"[bootstrap] first chunk ready in "
                       f"{1000*(time.perf_counter()-boot_t_obs):.0f}ms, H={chunk_H}")

        # ---- Main control loop --------------------------------------
        last_la_cmd = init_full_arm[LEFT_ARM_SLICE].copy()
        last_lh_cmd = end_lh.copy()
        while True:
            if quit_event.is_set():
                logger_mp.info("[main] quit_event set, exiting closed-loop")
                break
            loop_start = time.perf_counter()

            # ---- 1. Capture obs / state (cheap; needed for safety) ---
            obs, current_arm_q, left_arm_state, full_ee_state = _capture_obs()
            if current_arm_q is None:
                logger_mp.warning("arm state unavailable, skipping step")
                time.sleep(period)
                continue
            left_hand_state = full_ee_state[:LH_DIM].astype(np.float32)

            # ---- 2. Push any freshly-arrived chunk into the ring -----
            if chunk_buffer.ready_event.is_set():
                ready = chunk_buffer.take_ready()
                chunk_buffer.ready_event.clear()
                inference_busy = False
                if ready is not None:
                    new_la, new_lh, new_t = ready
                    ring.push(new_la, new_lh, new_t)
                    t_query_total += max(0.0, loop_start - new_t)

            # ---- 3. Temporal-ensemble the target for THIS tick -------
            la_target_raw, lh_target_raw, n_contrib = ring.ensemble(loop_start)
            if la_target_raw is None:
                # No chunk covers this tick — every recent chunk has run
                # out and the worker hasn't published a fresh one in time.
                # Hold the last sent command (hold thread keeps motor fed).
                la_target_raw = last_la_cmd.astype(np.float32)
                lh_target_raw = last_lh_cmd.astype(np.float32)

            if ema_la is None:
                ema_la = la_target_raw.copy()
                ema_lh = lh_target_raw.copy()
            else:
                ema_la = alpha * la_target_raw + (1.0 - alpha) * ema_la
                ema_lh = alpha * lh_target_raw + (1.0 - alpha) * ema_lh

            la_cmd = ema_la.astype(np.float64)
            lh_cmd = ema_lh.astype(np.float64)

            # ---- 3b. Soft-start ramp (first --ramp-steps ticks) ------
            if step < args.ramp_steps:
                w = (step + 1) / float(args.ramp_steps)
                la_cmd = (1.0 - w) * left_arm_state.astype(np.float64) + w * la_cmd
                lh_cmd = (1.0 - w) * left_hand_state.astype(np.float64) + w * lh_cmd

            # ---- 3c. Safety clamps ----------------------------------
            if not args.disable_safety:
                la_cmd, abort_a, reason_a = safety_clamp(
                    la_cmd, current_arm_q[LEFT_ARM_SLICE],
                    LEFT_ARM_LIMITS_LOW, LEFT_ARM_LIMITS_HIGH,
                    args.arm_delta_cap, args.arm_abort, "left_arm",
                )
                lh_cmd, abort_h, reason_h = safety_clamp(
                    lh_cmd, full_ee_state[:LH_DIM],
                    LEFT_HAND_LIMITS_LOW, LEFT_HAND_LIMITS_HIGH,
                    args.hand_delta_cap, args.hand_abort, "left_hand",
                )
                if abort_a or abort_h:
                    logger_mp.error(f"[SAFETY ABORT] {reason_a or reason_h}")
                    break

            # ---- 4. Publish target to hold thread --------------------
            arm_cmd = init_full_arm.copy()             # right arm stays at init
            arm_cmd[LEFT_ARM_SLICE] = la_cmd
            ctrl_state.set_arm(arm_cmd)
            ctrl_state.set_hands(lh_cmd, right_hand_zero)
            last_la_cmd = la_cmd.copy()
            last_lh_cmd = lh_cmd.copy()

            # ---- 5. Fire next prefetch if worker is idle -------------
            # Stride limit + worker-free gating naturally caps query rate
            # to the lesser of (1 / inference_latency) and frequency / stride.
            if (
                obs is not None
                and not inference_busy
                and not chunk_buffer.ready_event.is_set()
                and step - last_query_step >= args.query_stride_ticks
            ):
                chunk_buffer.submit(obs, loop_start)
                inference_busy = True
                n_queries += 1
                last_query_step = step

            step += 1
            if step % 15 == 0:
                avg_q = 1000 * t_query_total / max(n_queries, 1)
                newest_age = ring.newest_age_ticks(time.perf_counter())
                # Wrist joints: 4=roll, 5=pitch, 6=yaw. Print model RAW
                # target, after-EMA cmd, and measured state. Drift = cmd
                # diverging from training range  ([-0.2,+0.3], [-0.7,-0.1],
                # [-0.15,+0.4]).
                logger_mp.info(
                    f"  step {step:>4}  q={n_queries}  avg_q={avg_q:.0f}ms  "
                    f"ring={len(ring)}  ens_n={n_contrib}  "
                    f"newest_age={newest_age:.1f}t"
                )
                logger_mp.info(
                    f"           wrist target raw "
                    f"roll={la_target_raw[4]:+.3f} pitch={la_target_raw[5]:+.3f} yaw={la_target_raw[6]:+.3f}  "
                    f"| cmd roll={la_cmd[4]:+.3f} pitch={la_cmd[5]:+.3f} yaw={la_cmd[6]:+.3f}  "
                    f"| state roll={left_arm_state[4]:+.3f} pitch={left_arm_state[5]:+.3f} yaw={left_arm_state[6]:+.3f}"
                )
                # DEBUG: command vs sensed for both slices. If idle slice
                # (RIGHT) is constant in cmd but the sensed value drifts,
                # the motor is still tracking from a far-away start; bump
                # init_pose_secs. If sensed left arm drifts from cmd, that's
                # the picking arm following the model output (expected).
                la_sp_cmd = arm_cmd[LEFT_ARM_SLICE][0]
                ra_sp_cmd = arm_cmd[RIGHT_ARM_SLICE][0]
                la_el_cmd = arm_cmd[LEFT_ARM_SLICE][3]
                ra_el_cmd = arm_cmd[RIGHT_ARM_SLICE][3]
                la_sp_st = current_arm_q[LEFT_ARM_SLICE][0]
                ra_sp_st = current_arm_q[RIGHT_ARM_SLICE][0]
                la_el_st = current_arm_q[LEFT_ARM_SLICE][3]
                ra_el_st = current_arm_q[RIGHT_ARM_SLICE][3]
                logger_mp.info(
                    f"           LEFT  cmd shoulder_pitch={la_sp_cmd:+.3f} elbow={la_el_cmd:+.3f}  "
                    f"| state shoulder_pitch={la_sp_st:+.3f} elbow={la_el_st:+.3f}"
                )
                logger_mp.info(
                    f"           RIGHT cmd shoulder_pitch={ra_sp_cmd:+.3f} elbow={ra_el_cmd:+.3f}  "
                    f"| state shoulder_pitch={ra_sp_st:+.3f} elbow={ra_el_st:+.3f}"
                )

            if args.max_steps is not None and step >= args.max_steps:
                logger_mp.info(f"reached max_steps={args.max_steps}, stopping")
                break

            # Maintain frequency
            time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        logger_mp.info("interrupted by user")
    except Exception as e:
        logger_mp.error(f"run failed: {e}", exc_info=True)
    finally:
        # Stop the stdin listener and restore the terminal to canonical mode
        # FIRST — the listener may be in cbreak (line-buffering off, echo off);
        # leaving it that way would break the operator's shell after exit.
        quit_event.set()
        if _old_termios is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(),
                                  termios.TCSADRAIN, _old_termios)
            except Exception as e:
                logger_mp.warning(f"[shutdown] failed to restore terminal: {e}")
        # Slew BEFORE stopping the hold thread — the slew writes through
        # ctrl_state, which only reaches the motor while hold_thread is alive.
        # If the user aborted at the prompt the robot is already at init,
        # so the lerp will be effectively a no-op (current ≈ target).
        if hold_thread.is_alive():
            slew_back_to_init(
                arm_ctrl, ee_shared, ctrl_state,
                init_full_arm, end_lh, right_hand_zero,
                period, secs=2.0,
            )
        stop_event.set()
        if hold_thread.is_alive():
            hold_thread.join(timeout=2.0)
        if infer_started and infer_thread.is_alive():
            infer_thread.join(timeout=2.0)
        logger_mp.info(f"[done] steps={step}  queries={n_queries}")


if __name__ == "__main__":
    main()
