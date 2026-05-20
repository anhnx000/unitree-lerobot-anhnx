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
    do NOT add state or cumsum on the client. Verified 2026-05-20: at ep5
    frame 0, chunk_la[0] ≈ dataset action[0] ≈ state[0] (delta ~0.01 rad).
  * Right arm is held at its initial pose; right hand at zeros.

Pre-req:
  * GR00T policy server already running (default tcp://127.0.0.1:5555).
  * image_server.py running on the G1 development unit.
  * Env: unitree_lerobot_clean.
"""

from __future__ import annotations

import argparse
import time
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
LH_DIM = 7   # left hand joints (dex3)
DEFAULT_TASK = "pick apple and put in the box"

# ---- Safety limits for the LEFT arm only (right side is untouched) -------
# Joint position limits (rad), per Unitree G1 spec. Tightened slightly inside
# the mechanical bounds so we never command end-stops.
#   ordering matches LA_NAMES in the openloop script:
#   shoulder-pitch, shoulder-roll, shoulder-yaw, elbow,
#   wrist-roll, wrist-pitch, wrist-yaw
LEFT_ARM_LIMITS_LOW  = np.array([-3.05, -1.55, -2.60, -1.04, -1.95, -1.60, -1.60], dtype=np.float64)
LEFT_ARM_LIMITS_HIGH = np.array([ 2.65,  2.20,  2.60,  2.07,  1.95,  1.60,  1.60], dtype=np.float64)

# Dex3 left-hand joint range (rad). Conservative — dex3 fingers operate
# roughly in [0, 1.7]; we add a small margin.
LEFT_HAND_LIMITS_LOW  = np.full(LH_DIM, -0.05, dtype=np.float64)
LEFT_HAND_LIMITS_HIGH = np.full(LH_DIM,  1.75, dtype=np.float64)

# Per-step velocity cap (rad/step). At 30 Hz: 0.04 rad/step ≈ 1.2 rad/s on
# the arm, 0.10 rad/step ≈ 3 rad/s on the hand (fingers need to close fast).
DEFAULT_ARM_DELTA_CAP  = 0.04
DEFAULT_HAND_DELTA_CAP = 0.10

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
    ap.add_argument("--frequency", type=float, default=30.0,
                    help="Control loop frequency in Hz. Must match training fps "
                         "(this model was trained at 30 Hz).")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--base-type", default="legs")

    # GR00T server
    ap.add_argument("--policy-host", default="127.0.0.1")
    ap.add_argument("--policy-port", type=int, default=5555)
    ap.add_argument("--task", default=DEFAULT_TASK)

    # Action chunk handling
    ap.add_argument("--chunk-stride", type=int, default=1,
                    help="Re-query the model every N executed steps. 1 = query "
                         "every step (highest CPU/GPU load). The server returns "
                         "a 16-step action chunk; stride uses the first N of it.")

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


def main():
    args = parse_args()

    if not (0.0 < args.ema_alpha <= 1.0):
        raise SystemExit(f"--ema-alpha must be in (0, 1], got {args.ema_alpha}")
    alpha = float(args.ema_alpha)

    # ---- GR00T server ----------------------------------------------------
    policy = RawPolicyClient(args.policy_host, args.policy_port, timeout_ms=60000)
    if not policy.ping():
        raise SystemExit(f"GR00T server not reachable on {args.policy_host}:{args.policy_port}")
    logger_mp.info(f"[server] ping ok at {args.policy_host}:{args.policy_port}")

    # ---- Image + robot interfaces ---------------------------------------
    image_client, image_config = setup_image_client(args)
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

    # ---- Initial pose: hold current arm pose, zero left hand ------------
    current_full_arm = arm_ctrl.get_current_dual_arm_q()
    init_full_arm = np.asarray(current_full_arm, dtype=np.float64).copy()
    logger_mp.info(f"[init] dual_arm_q={init_full_arm.tolist()}")

    user_input = input("Enter 's' to start closed-loop GR00T evaluation: ")
    if user_input.lower() != "s":
        logger_mp.info("aborted by user")
        return

    # Move to initial pose explicitly so the first action is well-defined.
    tau = arm_ik.solve_tau(init_full_arm)
    arm_ctrl.ctrl_dual_arm(init_full_arm, tau)
    time.sleep(1.0)

    # Initialize left hand at zero (closed depends on dex3 zero pose — adjust
    # if your hand convention differs).
    left_hand_zero = np.zeros(LH_DIM, dtype=np.float64)
    right_hand_zero = np.zeros(LH_DIM, dtype=np.float64)
    with ee_shared["lock"]:
        ee_shared["left"][:]  = left_hand_zero.tolist()
        ee_shared["right"][:] = right_hand_zero.tolist()

    # ---- EMA state -------------------------------------------------------
    ema_la: np.ndarray | None = None
    ema_lh: np.ndarray | None = None

    # ---- Action chunk cache --------------------------------------------
    # The model returns a chunk of action_horizon predictions per query; we
    # consume them for `--chunk-stride` steps before re-querying. N1.7
    # server returns ABSOLUTE joint targets for both arm and hand
    # (state_action_processor.unapply_action runs server-side), so the chunk
    # is used directly without anchor/cumsum manipulation.
    chunk_la_abs: np.ndarray | None = None    # (H, LA_DIM) absolute
    chunk_lh_abs: np.ndarray | None = None    # (H, LH_DIM) absolute
    chunk_idx = 0

    period = 1.0 / float(args.frequency)
    logger_mp.info(f"[run] control loop at {args.frequency:.1f} Hz, alpha={alpha:.2f}, "
                   f"chunk_stride={args.chunk_stride}")

    step = 0
    n_queries = 0
    t_query_total = 0.0
    try:
        while True:
            loop_start = time.perf_counter()

            # ---- 1. Observation ---------------------------------------
            observation, current_arm_q = process_images_and_observations(
                image_client, image_config, arm_ctrl
            )
            if current_arm_q is None:
                logger_mp.warning("arm state unavailable, skipping step")
                time.sleep(period)
                continue
            current_arm_q = np.asarray(current_arm_q, dtype=np.float64)
            left_arm_state = current_arm_q[:LA_DIM].astype(np.float32)

            with ee_shared["lock"]:
                full_ee_state = np.array(ee_shared["state"][:], dtype=np.float64)
            left_hand_state = full_ee_state[:LH_DIM].astype(np.float32)

            # Images: process_images_and_observations returns torch tensors HxWx3 RGB.
            cam_high_t = observation.get("observation.images.cam_left_high", None)
            cam_wrist_t = observation.get("observation.images.cam_left_wrist", None)
            if cam_high_t is None or cam_wrist_t is None:
                logger_mp.warning("missing camera frame, skipping step")
                time.sleep(period)
                continue
            cam_high_rgb  = cam_high_t.numpy().astype(np.uint8)
            cam_wrist_rgb = cam_wrist_t.numpy().astype(np.uint8)

            # ---- 2. Query policy (or reuse cached chunk) --------------
            need_query = (
                chunk_la_abs is None
                or chunk_idx >= args.chunk_stride
                or chunk_idx >= chunk_la_abs.shape[0]
            )
            if need_query:
                obs = build_observation(
                    cam_high_rgb, cam_wrist_rgb,
                    left_arm_state, left_hand_state,
                    args.task,
                )
                tq = time.perf_counter()
                action_dict, _ = policy.get_action(obs)
                t_query_total += time.perf_counter() - tq
                n_queries += 1

                # N1.7 server already unapplied relative→absolute server-side.
                chunk_la_abs = np.asarray(action_dict["left_arm"],  dtype=np.float32).reshape(-1, LA_DIM)
                chunk_lh_abs = np.asarray(action_dict["left_hand"], dtype=np.float32).reshape(-1, LH_DIM)
                chunk_idx = 0

            # ---- 3. Resolve absolute target for this tick -------------
            k = chunk_idx
            la_target_raw = chunk_la_abs[k]
            lh_target_raw = chunk_lh_abs[k]

            # EMA smoothing on the executed action stream.
            if ema_la is None:
                ema_la = la_target_raw.copy()
                ema_lh = lh_target_raw.copy()
            else:
                ema_la = alpha * la_target_raw + (1.0 - alpha) * ema_la
                ema_lh = alpha * lh_target_raw + (1.0 - alpha) * ema_lh

            la_cmd = ema_la.astype(np.float64)
            lh_cmd = ema_lh.astype(np.float64)

            # ---- 3b. Soft-start ramp (left side only) ----------------
            # Linearly blend from the current measured state toward the
            # predicted target over the first --ramp-steps ticks so the
            # robot doesn't snap to whatever the policy says on frame 0.
            if step < args.ramp_steps:
                w = (step + 1) / float(args.ramp_steps)
                la_cmd = (1.0 - w) * left_arm_state.astype(np.float64) + w * la_cmd
                lh_cmd = (1.0 - w) * left_hand_state.astype(np.float64) + w * lh_cmd

            # ---- 3c. Safety clamps (left arm + left hand only) -------
            if not args.disable_safety:
                la_cmd, abort_a, reason_a = safety_clamp(
                    la_cmd, current_arm_q[:LA_DIM],
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

            # ---- 4. Send to robot -------------------------------------
            arm_cmd = init_full_arm.copy()             # right arm stays at init
            arm_cmd[:LA_DIM] = la_cmd
            tau = arm_ik.solve_tau(arm_cmd)
            arm_ctrl.ctrl_dual_arm(arm_cmd, tau)

            with ee_shared["lock"]:
                ee_shared["left"][:]  = lh_cmd.tolist()
                ee_shared["right"][:] = right_hand_zero.tolist()

            chunk_idx += 1
            step += 1
            if step % 30 == 0:
                avg_q = 1000 * t_query_total / max(n_queries, 1)
                logger_mp.info(f"  step {step}  queries={n_queries}  avg_query={avg_q:.1f} ms")

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
        logger_mp.info(f"[done] steps={step}  queries={n_queries}")


if __name__ == "__main__":
    main()
