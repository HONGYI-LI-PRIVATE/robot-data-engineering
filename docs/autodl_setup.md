# AutoDL Cloud Setup Guide

## 1. Register & Create Instance

1. Go to [autodl.com](https://www.autodl.com) and register
2. Create a new instance:
   - **GPU**: RTX 4090 (24GB VRAM, ~2 RMB/hour)
   - **Image**: PyTorch 2.7+ / CUDA 12.8 / Python 3.12 (pick the closest official image)
   - **Data disk**: Mount a data disk (50-100GB to start, expand as needed)
   - **Region**: Pick one with stock; price is the same

## 2. SSH Access

After instance starts, copy the SSH command from the console:

```bash
ssh -p <port> root@connect.<region>.autodl.com
```

Add to `~/.ssh/config` for convenience:

```
Host autodl
    HostName connect.<region>.autodl.com
    Port <port>
    User root
```

## 3. Environment Setup

### Option A: Docker (Recommended)

```bash
# Clone repo
cd /root
git clone https://github.com/<your-username>/robot-data-engineering.git
cd robot-data-engineering

# Build image
docker build -t robot-data-eng .

# Run with GPU access and data disk mount
docker run --gpus all \
    -v /root/autodl-tmp:/data \
    -v /root/robot-data-engineering:/workspace \
    -it robot-data-eng
```

### Option B: Direct Install (if Docker not available)

```bash
cd /root
git clone https://github.com/<your-username>/robot-data-engineering.git
cd robot-data-engineering

pip install -e ".[dev]"
```

## 4. Hugging Face Mirror (China)

Hugging Face Hub is blocked in China. Use the mirror:

```bash
# Add to ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com

# Test download
python -c "from lerobot.datasets import LeRobotDataset; ds = LeRobotDataset('lerobot/aloha_static_coffee'); print(ds.num_episodes)"
```

For AgiBot World, use ModelScope mirror:

```bash
pip install modelscope
# Download via ModelScope instead of HF Hub
```

## 5. W&B Setup

```bash
pip install wandb
wandb login  # Paste API key from wandb.ai/settings
python -m scripts.wandb_smoke_test
```

## 6. Cost Management

- **Always stop instance when done**: AutoDL charges by the hour even when idle
- **Data disk persists** across instance restarts — store datasets there (`/root/autodl-tmp/`)
- **Budget**: ~600 RMB for 300 hours of 4090 time (Weeks 4-7)
- This week (Week 1) should use < 5 hours of GPU time

## 7. Verify Setup

Run the full check:

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check LeRobot
python -c "import lerobot; print(lerobot.__version__)"

# Download and visualize
python -m scripts.visualize_episode --dataset lerobot/aloha_static_coffee --episode 0

# Check W&B
python -m scripts.wandb_smoke_test
```

If all four pass, Week 1 setup is complete.
