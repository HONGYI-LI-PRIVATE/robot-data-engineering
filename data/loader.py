"""数据加载器：统一读取 LeRobot 和 HDF5 格式。

不管原始格式是什么，加载后都变成相同的 DatasetInfo 结构，
后面的分析代码不需要关心数据来源。

支持的格式：
  - LeRobot v2：传入数据集目录路径（含 meta/info.json）
  - HDF5 (robomimic)：传入 .hdf5 文件路径
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass
class EpisodeData:
    """一个 episode（一段完整操作）的数据。

    不管原始格式叫 demo 还是 episode，加载后统一叫 episode。
    """

    actions: np.ndarray  # (T, action_dim) 每帧的动作指令
    timestamps: np.ndarray | None = None  # (T,) 每帧的时间戳（秒），HDF5 通常没有


@dataclass
class DatasetInfo:
    """加载后的数据集，格式无关。

    所有分析函数都接收这个结构，不需要知道数据原来是 Parquet 还是 HDF5。
    """

    name: str
    format: str  # "lerobot" 或 "hdf5"
    fps: float | None  # 帧率，HDF5 可能没有
    num_episodes: int
    total_frames: int
    action_dim: int
    robot_type: str | None
    episodes: list[EpisodeData]
    camera_names: list[str] = field(default_factory=list)


def load_dataset(path: str) -> DatasetInfo:
    """自动检测格式并加载。

    传 .hdf5 文件 → HDF5 加载器
    传目录（含 meta/info.json）→ LeRobot 加载器
    """
    p = Path(path)
    if p.suffix == ".hdf5":
        return _load_hdf5(p)
    elif (p / "meta" / "info.json").exists():
        return _load_lerobot(p)
    else:
        raise ValueError(f"无法识别数据格式: {path}，需要 .hdf5 文件或含 meta/info.json 的目录")


def _load_lerobot(path: Path) -> DatasetInfo:
    """加载 LeRobot v2 格式（Parquet + MP4）。

    读取流程：
    1. meta/info.json → 元信息（帧率、机器人类型、摄像头列表）
    2. data/*.parquet → 所有帧的动作和时间戳
    3. 按 episode_index 分组 → 每个 episode 一个 EpisodeData
    """
    import pandas as pd

    with open(path / "meta" / "info.json") as f:
        info = json.load(f)

    fps = info.get("fps")
    robot_type = info.get("robot_type")

    # 从 features 里找出所有摄像头名称
    camera_names = []
    features = info.get("features", {})
    for k in features:
        if k.startswith("observation.images."):
            camera_names.append(k.replace("observation.images.", ""))

    # 读所有 Parquet 文件，拼成一张大表
    parquet_files = sorted((path / "data").rglob("*.parquet"))
    dfs = [pd.read_parquet(pf) for pf in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    # 按 episode_index 分组，每组就是一个 episode
    episodes = []
    for _, group in df.groupby("episode_index"):
        group = group.sort_values("frame_index")

        # action 列在 Parquet 里存的是 list，转成 numpy 数组
        actions = np.array(group["action"].tolist(), dtype=np.float32)

        timestamps = None
        if "timestamp" in group.columns:
            timestamps = group["timestamp"].values.astype(np.float64)

        episodes.append(EpisodeData(actions=actions, timestamps=timestamps))

    total_frames = sum(ep.actions.shape[0] for ep in episodes)
    action_dim = episodes[0].actions.shape[1] if episodes else 0

    return DatasetInfo(
        name=path.name,
        format="lerobot",
        fps=fps,
        num_episodes=len(episodes),
        total_frames=total_frames,
        action_dim=action_dim,
        robot_type=robot_type,
        episodes=episodes,
        camera_names=camera_names,
    )


def _load_hdf5(path: Path) -> DatasetInfo:
    """加载 HDF5 格式（如 robomimic）。

    HDF5 里的 demo_0, demo_1, ... 就是一个个 episode，
    每个 demo 里有 actions 数组。
    """
    import h5py

    f = h5py.File(path, "r")

    # 找所有 demo 并按编号排序
    demo_keys = sorted(
        [k for k in f["data"].keys() if k.startswith("demo")],
        key=lambda x: int(x.split("_")[1]),
    )

    # 尝试从 attrs 里读帧率
    fps = None
    env_args_str = f["data"].attrs.get("env_args")
    if env_args_str:
        try:
            env_args = json.loads(env_args_str)
            fps = env_args.get("env_kwargs", {}).get("control_freq")
        except (json.JSONDecodeError, TypeError):
            pass

    episodes = []
    for dk in demo_keys:
        actions = f[f"data/{dk}/actions"][:]
        episodes.append(EpisodeData(actions=actions, timestamps=None))

    f.close()

    total_frames = sum(ep.actions.shape[0] for ep in episodes)
    action_dim = episodes[0].actions.shape[1] if episodes else 0

    return DatasetInfo(
        name=path.stem,
        format="hdf5",
        fps=fps,
        num_episodes=len(episodes),
        total_frames=total_frames,
        action_dim=action_dim,
        robot_type=None,
        episodes=episodes,
        camera_names=[],
    )
