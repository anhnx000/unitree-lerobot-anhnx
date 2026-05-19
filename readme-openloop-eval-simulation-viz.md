# GR00T Eval — Open-loop & Closed-loop visualization (G1 Pick Apple)

End-to-end recipe for evaluating the finetuned GR00T policy
`g1-left-arm-pick-apple/checkpoint-5500` against the dataset
`data_converted/pick_and_put_v4_converted` and producing side-by-side
videos of two MuJoCo G1 robots — **ground truth** vs **prediction**.

---

## Output layout

All artifacts land under `/home/anhnx10/work/unitree_lerobot/outputs/`:

```
outputs/
├── closeloop-videos/      # closed-loop simulation videos
│   └── closeloop_g1_ep{N}.mp4
├── openloop-videos/       # open-loop 4-panel comparison videos
│   └── openloop_g1_ep{N}.mp4
└── openloop_plots/        # per-joint GT-vs-Pred plots
    ├── openloop_ep{N}_left_arm.png
    └── openloop_ep{N}_left_hand.png
```

Both eval scripts write here by default — no flag needed.

---

## 1. Start the GR00T policy server

In **terminal #1** (env `isaac_groot`):

```bash
conda activate isaac_groot
cd /home/anhnx10/work/vla_g1/Isaac-GR00T

uv run python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /home/anhnx10/work/vla_g1/Isaac-GR00T/outputs/checkpoints/g1-left-arm-pick-apple/checkpoint-5500 \
  --device cuda:0 \
  --host 0.0.0.0 --port 5555
```

Wait for `✓ Server ready — listening on 0.0.0.0:5555`.

Quick sanity check (in any env with `pyzmq` + `msgpack-numpy`):

```bash
python -c "
import zmq, msgpack_numpy as mnp
s = zmq.Context().socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 5000); s.connect('tcp://127.0.0.1:5555')
s.send(mnp.packb({'endpoint':'ping'})); print(mnp.unpackb(s.recv(), raw=False))
"
```

---

## 2. Open-loop evaluation (per-step compare vs GT)

**What it does** — for every step `t` of an episode:

1. Reads the ground-truth observation from the dataset:
  `cam_high[t]`, `cam_left_wrist[t]`, `state.left_arm[t]`, `state.left_hand[t]`.
2. Sends them to the GR00T server, receives a 16-step action chunk
  (`left_arm` RELATIVE deltas, `left_hand` ABSOLUTE targets).
3. Takes the first step of the chunk → absolute joint target by
  integrating the delta on the current state.
4. Compares to the dataset's ground-truth `action[t]`.

**Outputs**

- `outputs/openloop-videos/openloop_g1_ep{N}.mp4` — 4-panel video
(cam_high, cam_left_wrist on top; MuJoCo G1 GT and MuJoCo G1 Pred below).
- `outputs/openloop_plots/openloop_ep{N}_left_arm.png` and `_left_hand.png` —
7-subplot per-joint GT vs Pred line plots.
- RMSE / L2 summary on stdout.

**Run** (in **terminal #2**, env `unitree_lerobot_clean`):

```bash
conda activate unitree_lerobot_clean
cd /home/anhnx10/work/unitree_lerobot

DISPLAY=:1 MUJOCO_GL=glfw python \
  unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_openloop.py \
  --episode 5 --host 127.0.0.1 --port 5555
```

Useful flags:


| flag                | default                                          | description                                            |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| `--episode N`       | `5`                                              | which episode in `data/chunk-000` to replay            |
| `--max-steps N`     | full episode                                     | cap number of frames (smoke test)                      |
| `--stride N`        | `1`                                              | query policy every N frames (`8` ≈ `action_horizon=8`) |
| `--save-video PATH` | `outputs/openloop-videos/openloop_g1_ep{ep}.mp4` | override video path                                    |
| `--out-dir DIR`     | `outputs/openloop_plots`                         | override plot directory                                |
| `--show`            | off                                              | also open a live OpenCV window                         |


Example with a different episode + stride:

```bash
DISPLAY=:1 MUJOCO_GL=glfw python \
  unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_openloop.py \
  --episode 17 --stride 4 --max-steps 400
```

Headless (no DISPLAY) is fine — drop `DISPLAY=:1` and `--show`.

---

## 3. Closed-loop simulation (drift over time)

Same camera input but the predicted action chunk **drives the predicted
robot state forward** without re-grounding on the dataset state, so the
right-side robot can diverge from GT.

```bash
conda activate unitree_lerobot_clean
DISPLAY=:1 MUJOCO_GL=glfw python \
  unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_simulation.py \
  --episode 5 --host 127.0.0.1 --port 5555 --action-horizon 8
```

Flags:


| flag                 | default                                            | description                                    |
| -------------------- | -------------------------------------------------- | ---------------------------------------------- |
| `--episode N`        | `5`                                                | episode to replay                              |
| `--action-horizon N` | `8`                                                | predicted chunk steps consumed before re-query |
| `--max-steps N`      | full episode                                       | cap frames                                     |
| `--save-video PATH`  | `outputs/closeloop-videos/closeloop_g1_ep{ep}.mp4` | override                                       |
| `--show`             | off                                                | live OpenCV window                             |


Output: 2-panel side-by-side MP4 (GT robot vs predicted robot).

---

## 4. Watching the videos

```bash
# VLC (must be installed via apt, not snap — snap bundle is incompatible with Arrow Lake GPU)
DISPLAY=:1 vlc outputs/openloop-videos/openloop_g1_ep5.mp4
DISPLAY=:1 vlc outputs/closeloop-videos/closeloop_g1_ep5.mp4

# or ffplay
DISPLAY=:1 ffplay outputs/openloop-videos/openloop_g1_ep5.mp4
```

If VSCode native preview is available, just click the `.mp4` in the
file explorer.

---

## 5. Interpreting results

- The console summary prints **per-joint RMSE** and **mean L2 error**.
Typical numbers for episode 5 after 5500 steps of finetuning:
  ```
  left_arm  RMSE per joint:  0.74 0.17 0.14 1.13 0.22 0.45 0.22
  left_hand RMSE per joint:  0.08 0.03 0.00 0.06 0.05 0.05 0.07
  L2 mean: left_arm = 1.46 rad, left_hand = 0.12 rad
  ```
  → elbow + shoulder-pitch dominate the error; hand fingers track well.
- Open-loop error is the *cleanest* signal — closed-loop additionally
accumulates drift, so larger divergence in `closeloop-videos/` is
expected, especially after action_horizon × ~8 steps.

---

## 6. Troubleshooting


| symptom                                                   | fix                                                                                                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `EGLError EGL_NOT_INITIALIZED` from `mujoco.Renderer`     | use `MUJOCO_GL=glfw` + `DISPLAY=:1`, not `egl`                                                                                       |
| `ModuleNotFoundError: msgpack_numpy`                      | `pip install msgpack-numpy` in `unitree_lerobot_clean`                                                                               |
| `server not reachable`                                    | check terminal #1 is still showing `Server ready`; firewall on port 5555                                                             |
| VLC segfault (`libGL error: failed to load driver: iris`) | gỡ snap, cài apt: `sudo snap remove vlc && sudo apt install -y vlc`                                                                  |
| Joint indices look swapped on right hand                  | normal — dataset stores right hand as thumb→index→middle, left hand as thumb→middle→index (mirrored). Scripts handle both correctly. |


---

## 7. File map


| path                                                                    | purpose                                          |
| ----------------------------------------------------------------------- | ------------------------------------------------ |
| `unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_openloop.py`   | open-loop eval + 4-panel video + per-joint plots |
| `unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_simulation.py` | closed-loop sim + 2-panel video                  |
| `unitree_lerobot/eval_robot/assets/g1/g1_body29_hand14.xml`             | MuJoCo G1 (29-DOF body + 14-DOF dual hand)       |
| `data_converted/pick_and_put_v4_converted/`                             | LeRobot v2.1 dataset (261 episodes, 30 fps)      |
| `outputs/`                                                              | all generated artifacts (see §Output layout)     |




- [unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_openloop.py](vscode-webview://1bufgb40k628ih8n0ivnfpej3snfj2nj0jffep008t30lts13bdl/unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_openloop.py) → mặc định lưu vào `outputs/openloop-videos/` + `outputs/openloop_plots/`
- [unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_simulation.py](vscode-webview://1bufgb40k628ih8n0ivnfpej3snfj2nj0jffep008t30lts13bdl/unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_simulation.py) → mặc định lưu vào `outputs/closeloop-videos/`

