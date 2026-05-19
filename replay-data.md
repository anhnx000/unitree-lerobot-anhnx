## Replay Dataset Trên Robot Thật

Các bước dưới đây dùng cho laptop đang nối Ethernet trực tiếp với robot Unitree G1.

### 1. Kích Hoạt Môi Trường

```bash
conda activate unitree_lerobot_clean
cd /home/anhnx10/work/unitree_lerobot
```

### 2. Set IP Cho Interface Ethernet

Kiểm tra tên interface:

```bash
ip addr
```

Với máy hiện tại, interface Ethernet là `enp129s0`. Gán IP cùng subnet với robot:

```bash
sudo ip addr add 192.168.123.23/24 dev enp129s0
```

Kiểm tra lại:

```bash
ip addr show dev enp129s0
```

Nên thấy dòng tương tự:

```text
inet 192.168.123.23/24 scope global secondary enp129s0
```

Kiểm tra kết nối tới image host mặc định của robot:

```bash
ping -c 2 192.168.123.164
```

### 3. Cài Dependency Image Server

Chỉ cần làm một lần:

```bash
sudo apt update
sudo apt install -y libusb-1.0-0-dev libturbojpeg-dev pkg-config

python -m pip install aiortc pupil-labs-uvc
```

Kiểm tra import:

```bash
python - <<'PY'
import aiortc
import uvc
print("aiortc ok")
print("uvc ok")
PY
```

### 4. Bật Image Server

Repo cần file `unitree_lerobot/cam_config_server.yaml`. Trong lần chạy hiện tại, có thể dùng config local tối thiểu để replay không bật visualization.

Chạy image server:

```bash
python -m unitree_lerobot.eval_robot.image_server.image_server --no-affinity
```

Giữ terminal này chạy. Nếu thành công, log sẽ có:

```text
[Responser] Camera Config Responser initialized at 0.0.0.0:60000
[Image Server] Running... Press Ctrl+C to exit.
```

Mở terminal khác và kiểm tra port:

```bash
python - <<'PY'
import socket
s = socket.socket()
s.settimeout(2)
s.connect(("127.0.0.1", 60000))
print("image server port 60000 OK")
s.close()
PY
```

### 5. Chạy Replay Đến Prompt

Mở terminal mới:

```bash
conda activate unitree_lerobot_clean
cd /home/anhnx10/work/unitree_lerobot
```

Chạy replay:

```bash
python -u unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id=unitreerobotics/G1_Dex3_ToastedBread_Dataset \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex3" \
  --image_host=127.0.0.1 \
  --motion=true \
  --visualization=false
```

Đợi đến khi hiện:

```text
Please enter the start signal (enter 's' to start the subsequent program):
```

Khi robot ở vùng an toàn, nhập:

```text
s
```

### 6. Dừng

Để dừng replay:

```bash
Ctrl+C
```

Để dừng image server:

```bash
Ctrl+C
```

### Ghi Chú An Toàn

- `--motion=true` dùng cho robot đang ở running mode.
- `--visualization=false` được dùng vì camera UVC hiện còn vấn đề permission/format; replay vẫn gửi action từ dataset xuống robot.
- Không nhập `s` nếu tay robot chưa ở vùng an toàn hoặc có người/vật trong workspace.

## Replay Raw Data 2 Camera: Đỉnh Đầu + Tay Trái

Phần này dùng cho dữ liệu raw tự thu có 2 camera:

- `color_0` -> camera đỉnh đầu, map thành `observation.images.cam_left_high`
- `color_1` -> camera bên tay trái, map thành `observation.images.cam_left_wrist`

Config dùng để convert là:

```text
Unitree_G1_Dex3_2Cam_Lefthand
```

Config này giữ nguyên state/action 28 chiều của `G1_29 + Dex3`:

```text
left_arm.qpos  + right_arm.qpos + left_ee.qpos + right_ee.qpos
7              + 7              + 7             + 7
```

### 1. Chuẩn Bị Env

```bash
conda activate unitree_lerobot_clean
cd /home/anhnx10/work/unitree_lerobot
```

### 2. Chọn Episode Raw

Ví dụ replay episode:

```text
data/pick_and_put_p1/pick_and_put/episode_0028
```

Tạo dataset tạm chỉ chứa episode này:

```bash
mkdir -p /home/anhnx10/work/unitree_lerobot/data/replay_episode_0028/pick_and_put

ln -sfn \
  /home/anhnx10/work/unitree_lerobot/data/pick_and_put_p1/pick_and_put/episode_0028 \
  /home/anhnx10/work/unitree_lerobot/data/replay_episode_0028/pick_and_put/episode_0028
```

### 3. Convert Raw Sang LeRobot

Mặc định LeRobot lưu dataset local vào `~/.cache/huggingface/lerobot`. Nếu muốn lưu vào workspace, set `HF_LEROBOT_HOME` trước khi convert:

```bash
export HF_LEROBOT_HOME=/home/anhnx10/work/unitree_lerobot/data_converted
```

Khi đó dataset converted sẽ nằm trong:

```text
/home/anhnx10/work/unitree_lerobot/data_converted/<repo-id>
```

```bash
python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir /home/anhnx10/work/unitree_lerobot/data/replay_episode_0028 \
  --repo-id anhnx10/pick_and_put_episode_0028 \
  --robot_type Unitree_G1_Dex3_2Cam_Lefthand \
  --mode image
```

Kiểm tra dataset đã convert:

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "anhnx10/pick_and_put_episode_0028"
ds = LeRobotDataset(repo_id=repo_id, root="", episodes=[0])
step = ds[0]

print("num_frames:", ds.num_frames)
print("state shape:", tuple(step["observation.state"].shape))
print("action shape:", tuple(step["action"].shape))
print("task:", step["task"])
PY
```

Kết quả mong đợi:

```text
state shape: (28,)
action shape: (28,)
```

### 4. Replay Episode Đã Convert

Đảm bảo image server đang chạy ở terminal khác:

```bash
python -m unitree_lerobot.eval_robot.image_server.image_server --no-affinity
```

Kiểm tra port:

```bash
python - <<'PY'
import socket
s = socket.socket()
s.settimeout(2)
s.connect(("127.0.0.1", 60000))
print("image server port 60000 OK")
s.close()
PY
```

Chạy replay:

```bash
export HF_LEROBOT_HOME=/home/anhnx10/work/unitree_lerobot/data_converted

python -u unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id=anhnx10/pick_and_put_episode_0028 \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex3" \
  --image_host=127.0.0.1 \
  --motion=true \
  --visualization=false
```

Hoặc có thể truyền trực tiếp `--root` tới dataset converted:

```bash
python -u unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id=anhnx10/pick_and_put_episode_0028 \
  --root=/home/anhnx10/work/unitree_lerobot/data_converted/anhnx10/pick_and_put_episode_0028 \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex3" \
  --image_host=127.0.0.1 \
  --motion=true \
  --visualization=false
```

Khi hiện:

```text
Please enter the start signal (enter 's' to start the subsequent program):
```

Nếu vùng robot đã an toàn, nhập:

```text
s
```

### 5. Đổi Sang Episode Khác

Nếu muốn replay episode khác, ví dụ `episode_0017`, đổi các chỗ `0028` thành `0017` và đổi repo id tương ứng:

```bash
mkdir -p /home/anhnx10/work/unitree_lerobot/data/replay_episode_0017/pick_and_put

ln -sfn \
  /home/anhnx10/work/unitree_lerobot/data/pick_and_put_p1/pick_and_put/episode_0017 \
  /home/anhnx10/work/unitree_lerobot/data/replay_episode_0017/pick_and_put/episode_0017

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir /home/anhnx10/work/unitree_lerobot/data/replay_episode_0017 \
  --repo-id anhnx10/pick_and_put_episode_0017 \
  --robot_type Unitree_G1_Dex3_2Cam_Lefthand \
  --mode image
```

Sau đó replay với:

```bash
python -u unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id=anhnx10/pick_and_put_episode_0017 \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex3" \
  --image_host=127.0.0.1 \
  --motion=true \
  --visualization=false
```

### 6. Convert Toàn Bộ Folder Raw Vào `data_converted`

Nếu muốn convert toàn bộ folder:

```text
/home/anhnx10/work/unitree_lerobot/data/pick_and_put_p1/pick_and_put
```

thì `--raw-dir` phải trỏ tới folder cha chứa task folder `pick_and_put`:

```text
/home/anhnx10/work/unitree_lerobot/data/pick_and_put_p1
```

Chạy:

```bash
conda activate unitree_lerobot_clean
cd /home/anhnx10/work/unitree_lerobot

export HF_LEROBOT_HOME=/home/anhnx10/work/unitree_lerobot/data_converted

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
  --raw-dir /home/anhnx10/work/unitree_lerobot/data/pick_and_put_p1 \
  --repo-id pick_and_put_all_2cam_lefthand \
  --robot_type Unitree_G1_Dex3_2Cam_Lefthand \
  --mode image
```

Dataset sẽ được lưu tại:

```text
/home/anhnx10/work/unitree_lerobot/data_converted/pick_and_put_all_2cam_lefthand
```

Replay episode đầu tiên:

```bash
export HF_LEROBOT_HOME=/home/anhnx10/work/unitree_lerobot/data_converted

python -u unitree_lerobot/eval_robot/replay_robot.py \
  --repo_id=pick_and_put_all_2cam_lefthand \
  --root="" \
  --episodes=0 \
  --frequency=30 \
  --arm="G1_29" \
  --ee="dex3" \
  --image_host=127.0.0.1 \
  --motion=true \
  --visualization=false
```

Replay episode khác bằng cách đổi `--episodes`, ví dụ:

```bash
--episodes=1
```

Lưu ý:

- `--repo-id` chỉ là tên dataset local trong LeRobot, không upload cloud nếu không truyền `--push_to_hub`.
- Nếu convert lại cùng `--repo-id`, converter sẽ xoá dataset local cũ cùng tên trước khi tạo lại.
- `--episodes` là index sau khi dataset được sort/convert, không nhất thiết trùng số trong tên raw folder như `episode_0028`.
