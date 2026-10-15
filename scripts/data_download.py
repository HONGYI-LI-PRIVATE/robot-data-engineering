"""Download datasets for Week 2 analysis.

我们需要两个数据集来学习机器人数据的结构和分析方法：

1. lerobot/aloha_static_coffee (LeRobot v2 格式)
   - 来自 ALOHA 双臂机器人的真机遥操作数据
   - 任务：冲咖啡
   - 格式：Parquet 表格(关节角度、动作) + MP4 视频(4个摄像头)
   - 50 个 episode，每个约 22 秒，50fps，共 ~1.57 GB

2. robomimic can/ph (HDF5 格式)
   - 来自 robomimic 的仿真环境人类示教数据
   - 任务：抓罐子 (can)，ph = proficient human (熟练操作员)
   - 格式：HDF5 文件，里面按 demo_0/demo_1/... 组织
   - 200 个 demo，low_dim 版 ~45 MB，raw 版 ~62 MB

两个数据集格式不同（Parquet+MP4 vs HDF5），正好用来练习读取不同格式的机器人数据。

Usage:
    python scripts/data_download.py              # download all
    python scripts/data_download.py --lerobot     # lerobot only
    python scripts/data_download.py --robomimic   # robomimic only

国内网络提示：下载前设置 HF 镜像
    export HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------- 路径配置 ----------
# __file__ 是当前脚本的路径，.parent.parent 往上两层到项目根目录，再进 data/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# LeRobot 数据集在 Hugging Face Hub 上的 ID，格式是 "组织名/数据集名"
LEROBOT_DATASET = "lerobot/aloha_static_coffee"
LEROBOT_DIR = DATA_DIR / "lerobot" / "aloha_static_coffee"

# robomimic 数据集本地存放路径
ROBOMIMIC_DIR = DATA_DIR / "robomimic" / "can_ph"


def download_lerobot():
    """下载 LeRobot 数据集。

    snapshot_download 会把整个数据集仓库拉下来，目录结构：
      data/lerobot/aloha_static_coffee/
      ├── meta/info.json              <- 元信息：帧率、动作维度、episode 数等
      ├── data/train-*.parquet        <- 所有帧的结构化数据（关节角、动作、时间戳）
      └── videos/                     <- 4 个摄像头的视频，每个摄像头一个 MP4
          ├── observation.images.cam_high/
          ├── observation.images.cam_low/
          ├── observation.images.cam_left_wrist/
          └── observation.images.cam_right_wrist/
    """
    print(f"\n{'=' * 60}")
    print(f"Downloading {LEROBOT_DATASET}")
    print(f"Destination: {LEROBOT_DIR}")
    print("Expected size: ~1.57 GB")
    print(f"{'=' * 60}\n")

    LEROBOT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # huggingface_hub 是 HF 官方的下载库
        # snapshot_download 下载整个数据集仓库到本地目录
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=LEROBOT_DATASET,  # Hub 上的数据集 ID
            repo_type="dataset",  # 告诉 HF 这是数据集而不是模型
            local_dir=str(LEROBOT_DIR),  # 下载到哪里
        )
        print(f"\nLeRobot dataset saved to {LEROBOT_DIR}")
    except Exception as e:
        # 如果 Python API 调用失败，改用命令行工具作为备选
        print(f"huggingface_hub download failed: {e}")
        print("Trying huggingface-cli fallback...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "huggingface_hub",
                "download",
                LEROBOT_DATASET,
                "--repo-type",
                "dataset",
                "--local-dir",
                str(LEROBOT_DIR),
            ],
            check=True,
        )
        print(f"\nLeRobot dataset saved to {LEROBOT_DIR}")


def download_robomimic():
    """下载 robomimic can/ph 数据集。

    我们下载两个版本：
    - low_dim_v15.hdf5: 只有数值数据（关节角度、末端位姿、动作），体积小，适合分析结构
    - demo_v15.hdf5:    原始录制数据，包含更多传感器信息

    HDF5 文件内部结构（用 h5py 打开后看到的）：
      data/
      ├── demo_0/
      │   ├── obs/                  <- 观测（机器人看到/感知到的）
      │   │   ├── robot0_eef_pos    <- 末端执行器位置 (T, 3)
      │   │   ├── robot0_joint_pos  <- 关节角度 (T, 7)
      │   │   └── object            <- 物体状态 (T, 14)
      │   ├── actions               <- 动作指令 (T, 7)
      │   └── dones                 <- 每帧是否结束 (T,) 布尔值
      ├── demo_1/
      │   └── ...
      └── ...共 200 个 demo

    注意：
    - 不走 robomimic 官方下载脚本，因为它依赖 egl_probe（Mac 装不了）。
      直接从 HuggingFace 仓库按文件下载，绕过这个问题。
    - image 版本（带相机图像）官方不提供直接下载，需要用仿真器从 demo 重放生成。
      HF 仓库里 can/ph 只有 low_dim 和 demo 两个文件，这里已经全部覆盖。
    """
    print(f"\n{'=' * 60}")
    print("Downloading robomimic can/ph")
    print(f"Destination: {ROBOMIMIC_DIR}")
    print("Expected size: low_dim ~45 MB, raw ~62 MB")
    print(f"{'=' * 60}\n")

    ROBOMIMIC_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # hf_hub_download 下载仓库中的单个文件（不是整个仓库）
        from huggingface_hub import hf_hub_download

        for filename, desc in [
            ("v1.5/can/ph/low_dim_v15.hdf5", "low_dim"),  # 纯数值，45 MB
            ("v1.5/can/ph/demo_v15.hdf5", "raw"),  # 原始录制，62 MB
        ]:
            print(f"Downloading {desc}: {filename}")
            hf_hub_download(
                repo_id="robomimic/robomimic_datasets",  # robomimic 官方 HF 仓库
                repo_type="dataset",
                filename=filename,  # 仓库内的文件路径
                local_dir=str(ROBOMIMIC_DIR),  # 下载到本地的目录
            )

        print(f"\nRobomimic dataset saved to {ROBOMIMIC_DIR}")
    except Exception as e:
        # 备选方案：用 robomimic 自带的下载脚本
        print(f"Download failed: {e}")
        print("Falling back to robomimic download script...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "robomimic.scripts.download_datasets",
                "--tasks",
                "can",  # 任务：抓罐子
                "--dataset_types",
                "ph",  # 数据类型：熟练人类示教
            ],
            check=True,
        )
        print("\nRobomimic downloaded to default location (~/.robomimic/datasets/)")
        print(f"You may want to move/symlink it to {ROBOMIMIC_DIR}")


def show_summary():
    """下载完成后统计每个数据集的实际大小和文件数。"""
    print(f"\n{'=' * 60}")
    print("Download summary")
    print(f"{'=' * 60}")
    for label, path in [("LeRobot", LEROBOT_DIR), ("Robomimic", ROBOMIMIC_DIR)]:
        if path.exists():
            # rglob("*") 递归遍历所有文件，累加大小
            size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024
            file_count = sum(1 for f in path.rglob("*") if f.is_file())
            print(f"  {label}: {size_mb:.1f} MB, {file_count} files")
        else:
            print(f"  {label}: not downloaded")


if __name__ == "__main__":
    # argparse 解析命令行参数，让你可以选择只下载其中一个
    parser = argparse.ArgumentParser(description="Download robot datasets")
    parser.add_argument("--lerobot", action="store_true", help="Download LeRobot only")
    parser.add_argument("--robomimic", action="store_true", help="Download Robomimic only")
    args = parser.parse_args()

    # 如果没指定任何 flag，就两个都下
    download_all = not args.lerobot and not args.robomimic

    if download_all or args.lerobot:
        download_lerobot()
    if download_all or args.robomimic:
        download_robomimic()

    show_summary()
