# Install Environment Without Conda Solver Hang

The original README command can hang at `Solving environment` when using conda's classic solver:

```bash
conda install -c conda-forge pinocchio ffmpeg=7.1.1
```

Use `libmamba` and force `conda-forge` only:

```bash
conda create -y -n unitree_lerobot_clean \
  -c conda-forge --override-channels --solver=libmamba \
  python=3.10 pinocchio ffmpeg=7.1.1
```

Activate the environment:

```bash
conda activate unitree_lerobot_clean
```

Install LeRobot:

```bash
cd /home/anhnx10/work/unitree_lerobot/unitree_lerobot/lerobot
python -m pip install -e .
```

Install this repo:

```bash
cd /home/anhnx10/work/unitree_lerobot
python -m pip install -e .
```

Install Unitree SDK2 Python:

```bash
cd /home/anhnx10/work/unitree_sdk2_python_fresh
python -m pip install -e .
```

Verify:

```bash
python - <<'PY'
import importlib.metadata
import pinocchio as pin
import lerobot
import unitree_lerobot
import unitree_sdk2py
import cyclonedds

print("pinocchio", pin.__version__)
print("lerobot", importlib.metadata.version("lerobot"))
print("unitree_lerobot", importlib.metadata.version("unitree_lerobot"))
print("unitree_sdk2py", importlib.metadata.version("unitree_sdk2py"))
print("cyclonedds import ok")
PY

ffmpeg -version | head -n 3
```

Expected verified versions from this machine:

```text
pinocchio 4.0.0
ffmpeg 7.1.1
lerobot 0.4.1
unitree_lerobot 0.3.0
unitree_sdk2py 1.0.1
cyclonedds import ok
```

Important notes:

- `--solver=libmamba` avoids the slow classic conda solver.
- `--override-channels` prevents mixing `defaults` with `conda-forge`.
- The LeRobot install can take a long time because it downloads large PyTorch CUDA packages.

## Part 2: Finish LeRobot README Environment

After reading `/home/anhnx10/work/unitree_lerobot/unitree_lerobot/lerobot/README.md`, the required environment steps are:

- Python 3.10 environment.
- `ffmpeg` installed inside the conda environment.
- `ffmpeg 7.X` with `libsvtav1`; this machine was verified with `ffmpeg 7.1.1`.
- LeRobot installed in editable mode with `pip install -e .`.
- Optional simulation extras can be installed with `aloha` and `pusht`.

The required base steps were already completed in Part 1. To finish the optional simulation environments from the LeRobot README:

```bash
conda activate unitree_lerobot_clean

cd /home/anhnx10/work/unitree_lerobot/unitree_lerobot/lerobot
python -m pip install -e ".[aloha,pusht]"
```

This installs the LeRobot simulation extras:

```text
gym-aloha 0.1.3
gym-pusht 0.1.6
mujoco 3.8.1
pygame 2.6.1
```

Verify the extras and `ffmpeg` encoder support:

```bash
python -c "import importlib.metadata, subprocess, gym_aloha, gym_pusht, mujoco, pygame; print('gym-aloha', importlib.metadata.version('gym-aloha')); print('gym-pusht', importlib.metadata.version('gym-pusht')); print('mujoco', importlib.metadata.version('mujoco')); print('pygame', importlib.metadata.version('pygame')); encoders=subprocess.check_output(['ffmpeg','-hide_banner','-encoders'], text=True, stderr=subprocess.STDOUT); print('ffmpeg libsvtav1', 'ok' if 'libsvtav1' in encoders else 'missing')"
```

Expected output from this machine:

```text
gym-aloha 0.1.3
gym-pusht 0.1.6
mujoco 3.8.1
pygame 2.6.1
ffmpeg libsvtav1 ok
```

Optional manual steps from the README:

- `wandb login` is only needed if you want Weights & Biases experiment tracking.
- System packages such as `cmake`, `build-essential`, `python3-dev`, `pkg-config`, and FFmpeg development libraries are only needed if PyAV or other native builds fail. They were not needed for the successful install above.
