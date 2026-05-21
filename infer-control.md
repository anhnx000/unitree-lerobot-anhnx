





# Tăng tốc gắp cho `eval_g1_left_hand_pick_apple_realife.py`

Tài liệu này tổng hợp các flag điều khiển tốc độ closed-loop và cách trade-off với độ ổn định / an toàn.

## Vì sao default chậm

`eval_g1_left_hand_pick_apple_realife.py` mặc định chạy ở `--frequency 20` Hz nhưng model được train ở **30 Hz** (xem `meta/info.json` của `pick_and_put_v4_converted`: `"fps": 30`). Chunk model trả về có `delta_indices=[0..15]` được pace ở 30 Hz (mỗi step cách nhau 33 ms).

Khi chạy ở 20 Hz, mỗi step thực thi cách nhau 50 ms → toàn bộ quỹ đạo chunk bị dãn ra **1.5×** so với lúc training. Cộng thêm velocity cap conservative `arm-delta-cap=0.027 rad/step = 0.54 rad/s` và `hand-delta-cap=0.067 rad/step = 1.33 rad/s`, motion bị giới hạn cứng.

## Các flag ảnh hưởng tốc độ

| Flag | Default | Tác động |
|---|---|---|
| `--frequency` | 20 Hz | Hz của control loop. Đặt = 30 để match training rate (×1.5 tốc độ replay chunk). |
| `--arm-delta-cap` | 0.027 rad/step | Velocity cap arm (rad/step). Wall-clock velocity = cap × frequency. |
| `--hand-delta-cap` | 0.067 rad/step | Velocity cap hand (rad/step). Quan trọng cho open/close gripper. |
| `--ema-alpha` | 0.5 | EMA smoothing. Cao = response nhạy, thấp = mượt hơn nhưng lag. |
| `--ensemble-tau-ticks` | 4 ticks | Decay constant của ACT-style temporal ensemble. Cao = trọng số chunk cũ cao hơn, smoother. |
| `--query-stride-ticks` | 1 | Tần suất query GR00T server (cap ở `1/inference_latency`). |
| `--arm-abort` | 0.6 rad | Emergency-stop nếu \|target - current\| vượt ngưỡng. |
| `--hand-abort` | 1.0 rad | Tương tự cho hand. Tăng nếu chunk thường có jump lớn (vd gripper từ open → close). |

### Lý thuyết quan hệ giữa frequency và velocity cap

- Tại frequency `f` (Hz), velocity wall-clock = `delta-cap × f` (rad/s).
- Default 20 Hz: arm 0.027 × 20 = **0.54 rad/s**, hand 0.067 × 20 = **1.33 rad/s**.
- Nếu chỉ đổi `--frequency 30` mà giữ delta-cap cũ: velocity = 0.027 × 30 = **0.81 rad/s** (tăng tự nhiên 1.5×).
- Muốn motion nhanh hơn nữa, **đồng thời** tăng cả frequency và delta-cap.

## Câu lệnh khuyến nghị

### Bước 1: Bắt đầu với 1.5× (match training)

```bash
cd /home/anhnx10/work/unitree_lerobot && \
python unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_realife.py \
    --frequency 30 \
    --arm-delta-cap 0.05 \
    --hand-delta-cap 0.15 \
    --ema-alpha 0.5 \
    --hand-abort 100
```

Thay đổi vs default:
- `--frequency 30` (×1.5 trajectory speed).
- `--arm-delta-cap 0.05` → 1.5 rad/s arm (~3× default 0.54).
- `--hand-delta-cap 0.15` → 4.5 rad/s hand (~3× default 1.33).
- `--hand-abort 100` → thực tế disable emergency-stop ngón tay vì chunk hand có jumps lớn (vd open→close gripper trong vài frame). Arm abort giữ 0.6.

Test 1-2 episode. Nếu robot mượt + gắp được → có thể push lên bước 2.

### Bước 2: Nhanh hơn nữa (×2-3 tổng tốc độ)

```bash
python unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_realife.py \
    --frequency 30 \
    --arm-delta-cap 0.08 \
    --hand-delta-cap 0.20 \
    --ema-alpha 0.7 \
    --ensemble-tau-ticks 6 \
    --hand-abort 100
```

Thêm thay đổi:
- `--arm-delta-cap 0.08` → 2.4 rad/s arm (×4.4 default).
- `--hand-delta-cap 0.20` → 6.0 rad/s hand.
- `--ema-alpha 0.7` → ít smoothing, response nhạy.
- `--ensemble-tau-ticks 6` → equivalent half-life với tau=4 ở 20Hz (tau ∝ frequency để giữ wall-clock decay không đổi). Giúp ensemble không jitter khi tick rate cao hơn.

### Bước 3: Tối đa tốc độ (cần verify cẩn thận)

```bash
python unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_realife.py \
    --frequency 30 \
    --arm-delta-cap 0.12 \
    --hand-delta-cap 0.30 \
    --ema-alpha 0.8 \
    --ensemble-tau-ticks 6 \
    --hand-abort 100 \
    --arm-abort 0.4
```

- `--arm-delta-cap 0.12` → 3.6 rad/s arm.
- `--hand-delta-cap 0.30` → 9.0 rad/s hand.
- **Giảm `--arm-abort` về 0.4** — quan trọng. Khi velocity cap cao, emergency-stop chặn cần "nhạy" hơn để bắt được command sai sớm.

## Diagnostic — chỗ nào chậm?

Log từ realife mỗi 15 step:
```
step  90  q=18  avg_q=145ms  ring=4  ens_n=3  newest_age=2.5t
       wrist target raw roll=+0.142 pitch=-0.487 yaw=+0.219  | cmd roll=... | state roll=...
```

- **`avg_q` cao (>200ms)** → inference latency cao, GPU load lớn hoặc network chậm. Bump `--query-stride-ticks 2` để giảm tần suất query.
- **`cmd` lag xa `target raw`** → velocity cap đang chặn. Tăng `--arm-delta-cap` / `--hand-delta-cap`.
- **`state` không kịp `cmd`** → motor PD không theo kịp. Hạ velocity cap. Đây là dấu hiệu **đã đẩy quá tốc độ** của motor.

### Vibration trở lại?

Theo thứ tự thử:
1. Hạ `--ema-alpha` về 0.5 (hoặc 0.3).
2. Hạ `--arm-delta-cap` về 0.05 (gentler).
3. Tăng `--ensemble-tau-ticks` lên 8 (smoother ensemble).
4. Verify **không có process slew_to_train_init.py còn sống**: `pgrep -af slew_to_train_init`. Conflict UDP là nguyên nhân #1 của rung loud motor.

## Safety knobs khi push tốc độ

- **`--arm-abort` 0.4** thay vì default 0.6 khi `--arm-delta-cap ≥ 0.08`. Emergency-stop chặn được command sai sớm.
- **`--hand-abort` 100** chỉ áp dụng cho hand vì training data có jumps lớn ở finger. Arm abort luôn ≤ 0.6.
- **`--max-steps`** dùng để test ngắn (vd 100 ticks ≈ 3.3s @ 30Hz) trước khi chạy full. Quan sát physical behavior.

## Tham chiếu nguồn

- Code path: `unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_realife.py`
- Convention chunk N1.7: server đã unapply relative→absolute, client dùng `chunk[k]` trực tiếp (xem [eval_g1_left_hand_pick_apple_realife.py](unitree_lerobot/eval_robot/eval_g1_left_hand_pick_apple_realife.py) docstring v4).
- Training fps: `data_converted/pick_and_put_v4_converted/meta/info.json`.
- Velocity cap defaults: hằng số `DEFAULT_ARM_DELTA_CAP`, `DEFAULT_HAND_DELTA_CAP` trong realife.py.



# Nhắc lại cách chạy code để xem 2 camera head và left wrist

Trước khi deploy realife, verify head_camera (D435I) và left_wrist_camera (D405)
đã mount đúng + cho POV khớp training distribution. Cần 2 terminals.

## Terminal 1 — image_server

Phục vụ frames qua ZMQ (head ở port 55555, left_wrist ở port 55556, request config
ở port 60000). Cờ `--rs` bắt buộc vì cả 2 cam đều là RealSense.

```bash
cd /home/anhnx10/work/unitree_lerobot
python -m unitree_lerobot.eval_robot.image_server.image_server --rs
```

Đợi log in ra "Camera Readiness Summary" với cả `head_camera` và
`left_wrist_camera` ở trạng thái READY. Nếu camera nào FAILED, check
`cam_config_server.yaml` xem serial number đúng chưa
(`udevadm info --query=property --name=/dev/videoN | grep ID_MODEL` để
kiểm tra mapping device).

## Terminal 2 — live viewer

Window OpenCV hiện 2 cam side-by-side (cam_high bên trái, cam_left_wrist
bên phải) với FPS overlay.

```bash
~/anaconda3/envs/unitree_lerobot_clean/bin/python \
    /home/anhnx10/work/unitree_lerobot/outputs/cam_probe/live_viewer.py
```

Phím:
- **`s`** → lưu cặp frame hiện tại ra `outputs/cam_probe/live_head.png` +
  `live_wrist.png` (overwrite). Dùng khi muốn ghi lại snapshot để so sánh
  với training reference.
- **`q`** hoặc **ESC** → quit.

Nếu image_server chạy ở host khác (ví dụ trên Jetson onboard), pass `--host`:

```bash
python /home/anhnx10/work/unitree_lerobot/outputs/cam_probe/live_viewer.py \
    --host 192.168.123.161
```

## Training reference (để so sánh POV)

So sánh với 2 frame extract từ training dataset (episode 5 frame 0):

- `outputs/cam_probe/training_ref_cam_high_ep5_f0.png`
- `outputs/cam_probe/training_ref_cam_left_wrist_ep5_f0.png`

Mục tiêu khớp POV training:
- **cam_high**: top-down workspace, vai/cánh tay trái robot lộ ở góc dưới-trái khung.
- **cam_left_wrist**: ngón Dex3 đen chiếm 1/3 đáy khung, táo ở giữa,
  giỏ vàng góc trên-phải.

POV cam_left_wrist chỉ khớp khi tay trái ở `LEFT_ARM_TRAIN_INIT`. Nếu robot
đang ở pose zero/init mặc định, view sẽ khác — dùng `slew_to_train_init.py`
để đưa tay vào training pose trước khi probe.

## Terminal 3 (optional) — slew about robot vào training-init pose để verify POV

**KHÔNG chạy đồng thời với realife.py** (2 process spam UDP → robot rung).

```bash
~/anaconda3/envs/unitree_lerobot_clean/bin/python \
    /home/anhnx10/work/unitree_lerobot/outputs/cam_probe/slew_to_train_init.py
```

Slew left arm → `LEFT_ARM_TRAIN_INIT`, right arm → `RIGHT_ARM_REST`,
left hand → `LEFT_HAND_TRAIN_INIT`. Sau slew, robot hold pose, bạn xem
camera trong terminal 2 để verify. Ctrl+C để thoát.

