# Robot Data Engineering

Real-world robot data analysis, format conversion, evaluation, and training pipelines for embodied AI.

## What This Repo Does

- **Data analysis**: Automated health-check reports for LeRobot / HDF5 datasets — episode distributions, action statistics, timestamp stability, anomaly detection
- **Format conversion**: HDF5 (robomimic) → LeRobot, rosbag reader, MCAP reader
- **Evaluation harness**: Success rate with Wilson confidence intervals, cross-seed variance, paired significance tests, sample-size calculators
- **Training configs**: Reproducible ACT / Diffusion Policy training with LeRobot
- **Data scaling experiments**: How much data is enough? Quality vs. quantity ablation on ManiSkill3

## Project Structure

```
data/          Analysis tools and dataset health-check reports
convert/       Format converters (HDF5 → LeRobot, rosbag/MCAP readers)
eval/          Evaluation harness with statistical rigor
train/         Training configs (ACT, Diffusion Policy)
  configs/     YAML configs for reproducible runs
scripts/       Utility scripts (visualization, W&B smoke test)
reports/       Generated reports, figures, and analysis outputs
```

## Quick Start

### Local (Mac / Linux)

```bash
# Clone
git clone https://github.com/<your-username>/robot-data-engineering.git
cd robot-data-engineering

# Create environment (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Visualize a real robot episode
python -m scripts.visualize_episode --dataset lerobot/aloha_static_coffee --episode 0

# Or use LeRobot's built-in visualizer
lerobot-dataset-viz --repo-id lerobot/aloha_static_coffee --episode-index 0
```

### Cloud (AutoDL with GPU)

```bash
# Build Docker image
docker build -t robot-data-eng .

# Run with GPU and data disk
docker run --gpus all -v /root/autodl-tmp:/data -it robot-data-eng

# Inside container: set HF mirror for China
export HF_ENDPOINT=https://hf-mirror.com
```

See [docs/autodl_setup.md](docs/autodl_setup.md) for detailed cloud setup instructions.

### W&B Integration

```bash
wandb login
python -m scripts.wandb_smoke_test
```

## Datasets Used

All datasets are public and free:

| Dataset | Source | Use |
|---|---|---|
| `lerobot/aloha_static_*` | HF Hub | Analysis, visualization |
| robomimic (`lift`, `can`, `square`) | robomimic.github.io | HDF5 analysis & conversion |
| Open X-Embodiment subsets | HF Hub (LeRobot format) | Cross-embodiment comparison |
| AgiBot World | ModelScope / HF Hub | Large-scale real robot data |
| ManiSkill3 (`PickCube`) | pip install mani_skill | Data scaling ablation |

## Requirements

- Python >= 3.12
- PyTorch >= 2.7 (CUDA 12.8 for GPU)
- See `pyproject.toml` for full dependency list
