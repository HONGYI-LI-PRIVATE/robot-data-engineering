"""分析函数：对加载好的数据集跑各项统计检查。

每个函数接收 DatasetInfo，返回一个结果字典。
所有函数都是纯计算，不涉及数据格式——格式差异已经被 loader.py 抹平了。
"""

import numpy as np

from .loader import DatasetInfo


def basic_stats(info: DatasetInfo) -> dict:
    """基本统计：episode 数、总帧数、episode 长度分布。

    这是最基础的体检项，快速了解数据集的规模和形状。
    """
    lengths = np.array([ep.actions.shape[0] for ep in info.episodes])

    return {
        "num_episodes": info.num_episodes,
        "total_frames": info.total_frames,
        "fps": info.fps,
        "action_dim": info.action_dim,
        "robot_type": info.robot_type,
        "episode_lengths": lengths,
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "length_min": int(np.min(lengths)),
        "length_max": int(np.max(lengths)),
        "length_percentiles": {
            "P25": float(np.percentile(lengths, 25)),
            "P50": float(np.percentile(lengths, 50)),
            "P75": float(np.percentile(lengths, 75)),
            "P95": float(np.percentile(lengths, 95)),
        },
    }


def action_analysis(info: DatasetInfo) -> dict:
    """动作空间分析：每一维的分布、极值、夹爪二值化检测。

    对动作的每一维分别统计，重点看：
    - 某一维是不是几乎不变（可能没用到）
    - 最后一维（通常是夹爪）是连续还是只有开/合两个值
    """
    # 把所有 episode 的动作拼成一个大数组 (total_frames, action_dim)
    all_actions = np.concatenate([ep.actions for ep in info.episodes], axis=0)

    per_dim = []
    for d in range(info.action_dim):
        col = all_actions[:, d]
        unique_vals = np.unique(col)

        # 不同取值 ≤ 3 个就认为是二值化（留点余量，有时有 0, 0.5, 1 三档）
        is_binary = len(unique_vals) <= 3

        per_dim.append(
            {
                "dim": d,
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "unique_count": len(unique_vals),
                "is_binary": is_binary,
            }
        )

    gripper_dim = info.action_dim - 1
    gripper_is_binary = per_dim[gripper_dim]["is_binary"] if per_dim else False

    return {
        "all_actions": all_actions,
        "per_dim": per_dim,
        "gripper_dim": gripper_dim,
        "gripper_is_binary": gripper_is_binary,
    }


def velocity_acceleration(info: DatasetInfo) -> dict:
    """关节速度和加速度分析，标出遥操作抖动。

    速度 = 相邻帧动作的差分（一阶差分）
    加速度 = 速度的差分（二阶差分）

    人手遥操作时会抖，表现为加速度出现尖峰。
    超过 3 倍标准差的帧标记为异常。
    """
    all_velocities = []
    all_accelerations = []

    # 第一遍：收集所有速度和加速度，用来算全局统计量
    for ep in info.episodes:
        if ep.actions.shape[0] < 3:
            continue
        velocity = np.diff(ep.actions, axis=0)  # (T-1, D)
        acceleration = np.diff(velocity, axis=0)  # (T-2, D)
        all_velocities.append(velocity)
        all_accelerations.append(acceleration)

    if not all_accelerations:
        return {
            "all_velocities": np.array([]),
            "all_accelerations": np.array([]),
            "acc_mean": np.array([]),
            "acc_std": np.array([]),
            "anomaly_threshold": 3.0,
            "anomaly_episodes": [],
        }

    all_vel = np.concatenate(all_velocities, axis=0)
    all_acc = np.concatenate(all_accelerations, axis=0)

    # 全局的加速度均值和标准差，按维度算
    acc_mean = np.mean(all_acc, axis=0)
    acc_std = np.std(all_acc, axis=0)
    threshold = 3.0

    # 第二遍：用全局阈值标出每个 episode 的异常帧
    anomaly_episodes = []
    for ep_idx, ep in enumerate(info.episodes):
        if ep.actions.shape[0] < 3:
            anomaly_episodes.append(
                {
                    "episode": ep_idx,
                    "anomaly_frames": 0,
                    "total_frames": ep.actions.shape[0],
                    "anomaly_ratio": 0.0,
                }
            )
            continue

        velocity = np.diff(ep.actions, axis=0)
        acceleration = np.diff(velocity, axis=0)

        # 任意一维加速度偏离均值超过 3σ 就算异常帧
        is_anomaly = np.any(np.abs(acceleration - acc_mean) > threshold * acc_std, axis=1)
        anomaly_count = int(np.sum(is_anomaly))

        anomaly_episodes.append(
            {
                "episode": ep_idx,
                "anomaly_frames": anomaly_count,
                "total_frames": ep.actions.shape[0],
                "anomaly_ratio": anomaly_count / ep.actions.shape[0],
            }
        )

    return {
        "all_velocities": all_vel,
        "all_accelerations": all_acc,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "anomaly_threshold": threshold,
        "anomaly_episodes": anomaly_episodes,
    }


def timestamp_check(info: DatasetInfo) -> dict | None:
    """时间戳检查：帧率稳定性、掉帧检测。

    计算相邻帧的时间间隔，和预期的 1/fps 对比：
    - 间隔稳定 → 帧率正常
    - 间隔偶尔突然变大 → 掉帧
    - 间隔方差大 → 录制设备不稳定

    只有 LeRobot 格式有时间戳，HDF5 通常没有，返回 None。
    """
    has_timestamps = any(ep.timestamps is not None for ep in info.episodes)
    if not has_timestamps:
        return None

    all_intervals = []
    dropped_frames = []

    for ep_idx, ep in enumerate(info.episodes):
        if ep.timestamps is None or len(ep.timestamps) < 2:
            continue

        # 相邻帧的时间差
        intervals = np.diff(ep.timestamps)
        all_intervals.append(intervals)

        # 掉帧 = 实际间隔超过预期间隔的 1.5 倍
        if info.fps:
            expected_interval = 1.0 / info.fps
            drops = np.where(intervals > expected_interval * 1.5)[0]
            if len(drops) > 0:
                dropped_frames.append(
                    {
                        "episode": ep_idx,
                        "drop_count": len(drops),
                        "max_gap": float(np.max(intervals)),
                    }
                )

    if not all_intervals:
        return None

    all_intervals = np.concatenate(all_intervals)

    return {
        "all_intervals": all_intervals,
        "interval_mean": float(np.mean(all_intervals)),
        "interval_std": float(np.std(all_intervals)),
        "interval_min": float(np.min(all_intervals)),
        "interval_max": float(np.max(all_intervals)),
        "expected_interval": 1.0 / info.fps if info.fps else None,
        "dropped_frames": dropped_frames,
        "total_drops": sum(d["drop_count"] for d in dropped_frames),
    }


def dead_episodes(info: DatasetInfo) -> dict:
    """废 episode 检测：动作总方差接近 0 说明机器人几乎没动。

    动作总方差 = 每一维方差之和 = 协方差矩阵的迹 (trace)
    方差低于所有 episode 中位数的 1% → 判定为废 episode
    """
    variances = []
    for ep_idx, ep in enumerate(info.episodes):
        var_per_dim = np.var(ep.actions, axis=0)  # 每一维的方差
        total_var = float(np.sum(var_per_dim))  # 求和得到标量
        variances.append(
            {
                "episode": ep_idx,
                "total_variance": total_var,
                "length": ep.actions.shape[0],
            }
        )

    all_vars = np.array([v["total_variance"] for v in variances])
    median_var = float(np.median(all_vars))
    dead_threshold = median_var * 0.01

    # 按方差排序，最小的排前面，方便看哪些最可疑
    variances_sorted = sorted(variances, key=lambda x: x["total_variance"])
    dead_eps = [v for v in variances_sorted if v["total_variance"] < dead_threshold]

    return {
        "variances": variances_sorted,
        "all_vars": all_vars,
        "median_variance": median_var,
        "dead_threshold": dead_threshold,
        "dead_episodes": dead_eps,
        "dead_count": len(dead_eps),
    }
