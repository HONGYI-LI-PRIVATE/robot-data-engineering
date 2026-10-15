# Scripts & Tools Usage Guide

每个脚本的功能说明和终端运行命令。所有命令都在项目根目录下执行。

---

## Week 1: 工程底座

### scripts/data_download.py

下载实验所需的数据集（LeRobot 格式 + robomimic HDF5）。

```bash
# 下载全部（LeRobot + robomimic）
python -m scripts.data_download

# 只下载 LeRobot 数据集
python -m scripts.data_download --lerobot

# 只下载 robomimic 数据集
python -m scripts.data_download --robomimic
```

国内网络需先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### scripts/visualize_episode.py

可视化 LeRobot 数据集中的单个 episode，输出 MP4 视频。

```bash
# 可视化第 0 个 episode
python -m scripts.visualize_episode --dataset lerobot/aloha_static_coffee --episode 0

# 指定输出路径
python -m scripts.visualize_episode \
    --dataset lerobot/aloha_static_coffee \
    --episode 0 \
    --output reports/ep0.mp4
```

### scripts/wandb_smoke_test.py

验证 Weights & Biases 集成是否可用。

```bash
python -m scripts.wandb_smoke_test
```

### scripts/swanlab_smoke_test.py

验证 SwanLab 集成是否可用。

```bash
python -m scripts.swanlab_smoke_test
```

---

## Week 2: 数据集分析工具

### data/report.py

对任意数据集（HDF5 或 LeRobot 格式）生成自包含 HTML 体检报告，包含：
- episode 长度分布
- 动作维度统计 + 夹爪二值化检测
- 关节速度/加速度异常（遥操作抖动）
- 时间戳稳定性 + 掉帧检测
- 废 episode 检测

```bash
# 分析 robomimic HDF5
python -m data.report --dataset data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5

# 分析 LeRobot 数据集
python -m data.report --dataset data/lerobot/aloha_static_coffee

# 指定输出目录
python -m data.report --dataset data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5 --output my_reports
```

### data/loader.py

统一数据加载器，自动识别格式（HDF5 / LeRobot），加载后返回相同的 DatasetInfo 结构。被 report.py 内部调用，一般不直接运行。

```python
from data.loader import load_dataset
info = load_dataset("data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5")
info = load_dataset("data/lerobot/aloha_static_coffee")
```

### data/analysis.py

五项分析函数，接收 DatasetInfo 返回结果字典。被 report.py 内部调用。

```python
from data.loader import load_dataset
from data import analysis

info = load_dataset("path/to/dataset")
basic = analysis.basic_stats(info)
actions = analysis.action_analysis(info)
vel_acc = analysis.velocity_acceleration(info)
ts = analysis.timestamp_check(info)       # HDF5 返回 None（无时间戳）
dead = analysis.dead_episodes(info)
```

---

## Week 3: 格式转换器

### convert/hdf5_to_lerobot.py

将 robomimic HDF5 转换为 LeRobot v2 格式（Parquet + MP4）。
使用 LeRobot 官方 `LeRobotDataset.create()` 接口。

```bash
# 查看 HDF5 结构（不转换，只打印映射关系）
python -m convert.hdf5_to_lerobot \
    --input data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5 \
    --repo-id test/can \
    --inspect-only

# 完整转换
python -m convert.hdf5_to_lerobot \
    --input data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5 \
    --repo-id user/robomimic_can_ph \
    --task "pick up the can and place it in the target bin"

# 只转换 train split
python -m convert.hdf5_to_lerobot \
    --input data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5 \
    --repo-id user/robomimic_can_ph \
    --task "pick up the can" \
    --split train

# 试跑前 5 个 episode
python -m convert.hdf5_to_lerobot \
    --input data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5 \
    --repo-id user/robomimic_can_ph \
    --task "pick up the can" \
    --max-episodes 5

# 指定输出目录和 FPS
python -m convert.hdf5_to_lerobot \
    --input data.hdf5 \
    --repo-id user/dataset \
    --task "manipulation" \
    --output-dir /path/to/output \
    --fps 30
```

转换器的字段映射关系：

| HDF5 obs key | LeRobot feature | 说明 |
|---|---|---|
| robot0_joint_pos | observation.state | 7 个关节角度 |
| robot0_eef_pos | observation.eef_pos | 末端位置 xyz |
| robot0_eef_quat | observation.eef_quat | 末端姿态四元数 |
| robot0_gripper_qpos | observation.gripper_pos | 夹爪开合 |
| robot0_joint_vel | observation.joint_vel | 关节角速度 |
| robot0_gripper_qvel | observation.gripper_vel | 夹爪速度 |
| *_image | observation.images.* | 图像自动发现 |
| object | (跳过) | 真机不可得 |
| *_cos / *_sin / *_site | (跳过) | 冗余 |
