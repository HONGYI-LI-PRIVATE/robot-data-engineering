"""将 robomimic HDF5 数据集转换为 LeRobot v2 格式。

使用 LeRobot 官方的 LeRobotDataset.create() 接口，不手动拼文件。

Usage:
    python -m convert.hdf5_to_lerobot \
        --input data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5 \
        --repo-id user/robomimic_can_ph \
        --task "pick up the can and place it in the target bin"

    # 带图像的版本（obs 里有 *_image 字段）
    python -m convert.hdf5_to_lerobot \
        --input data/robomimic/can_ph/v1.5/can/ph/image_v15.hdf5 \
        --repo-id user/robomimic_can_ph_image \
        --task "pick up the can and place it in the target bin"

    # 只转换 train split
    python -m convert.hdf5_to_lerobot \
        --input data.hdf5 --repo-id user/dataset --split train

    # 试跑前 5 个 episode
    python -m convert.hdf5_to_lerobot \
        --input data.hdf5 --repo-id user/dataset --max-episodes 5
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# ── robomimic obs → LeRobot feature 的映射 ──────────────────
#
# 左边是 HDF5 里 obs/ 下的 key，右边是 LeRobot 里的 feature name。
# 只映射对真机部署有意义的字段，跳过：
#   - object: 物体状态（真机上拿不到）
#   - *_cos / *_sin: joint_pos 的三角变换（冗余）
#   - *_site: 备用坐标系（冗余）
#
# 图像字段（key 以 _image 结尾）在运行时自动发现，不需要写死。

OBS_KEY_MAP = {
    "robot0_joint_pos": "observation.state",
    "robot0_eef_pos": "observation.eef_pos",
    "robot0_eef_quat": "observation.eef_quat",
    "robot0_gripper_qpos": "observation.gripper_pos",
    "robot0_joint_vel": "observation.joint_vel",
    "robot0_gripper_qvel": "observation.gripper_vel",
}

SKIP_OBS_KEYS = {
    "object",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_eef_quat_site",
}


def inspect_hdf5(path: str) -> dict:
    """读取 HDF5 元信息，不加载全部数据。"""
    with h5py.File(path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"] if k.startswith("demo")],
            key=lambda x: int(x.split("_")[1]),
        )

        demo0 = f[f"data/{demo_keys[0]}"]

        obs_keys = list(demo0["obs"].keys())
        image_keys = [k for k in obs_keys if k.endswith("_image")]
        state_keys = [k for k in obs_keys if not k.endswith("_image")]

        obs_shapes = {}
        for k in obs_keys:
            ds = demo0[f"obs/{k}"]
            obs_shapes[k] = {"shape": ds.shape[1:], "dtype": str(ds.dtype)}

        action_shape = demo0["actions"].shape[1:]
        action_dtype = str(demo0["actions"].dtype)

        fps = None
        env_name = None
        robot_type = None
        env_args_str = f["data"].attrs.get("env_args")
        if env_args_str:
            try:
                env_args = json.loads(env_args_str)
                env_name = env_args.get("env_name")
                kw = env_args.get("env_kwargs", {})
                fps = kw.get("control_freq")
                robots = kw.get("robots", [])
                if robots:
                    robot_type = robots[0] if isinstance(robots, list) else robots
            except (json.JSONDecodeError, TypeError):
                pass

        masks = {}
        if "mask" in f:
            for mk in f["mask"]:
                masks[mk] = [x.decode() if isinstance(x, bytes) else x for x in f["mask"][mk][:]]

        return {
            "demo_keys": demo_keys,
            "num_demos": len(demo_keys),
            "obs_keys": obs_keys,
            "image_keys": image_keys,
            "state_keys": state_keys,
            "obs_shapes": obs_shapes,
            "action_shape": action_shape,
            "action_dtype": action_dtype,
            "fps": fps,
            "env_name": env_name,
            "robot_type": robot_type,
            "masks": masks,
        }


def build_features(meta: dict) -> dict:
    """根据 HDF5 元信息构建 LeRobot features 字典。"""
    features = {}

    features["action"] = {
        "dtype": "float32",
        "shape": tuple(meta["action_shape"]),
        "names": None,
    }

    for obs_key, obs_info in meta["obs_shapes"].items():
        if obs_key in SKIP_OBS_KEYS:
            continue

        if obs_key in OBS_KEY_MAP:
            lr_name = OBS_KEY_MAP[obs_key]
        elif obs_key.endswith("_image"):
            camera_name = obs_key.replace("_image", "")
            lr_name = f"observation.images.{camera_name}"
        else:
            lr_name = f"observation.{obs_key}"

        shape = tuple(obs_info["shape"])

        if obs_key.endswith("_image"):
            features[lr_name] = {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
        else:
            features[lr_name] = {
                "dtype": "float32",
                "shape": shape,
                "names": None,
            }

    return features


def convert(
    input_path: str,
    repo_id: str,
    task: str,
    output_dir: str | None = None,
    fps: int | None = None,
    split: str | None = None,
    max_episodes: int | None = None,
    robot_type: str | None = None,
):
    """执行转换。"""
    from lerobot.datasets import LeRobotDataset

    input_path = str(Path(input_path).resolve())
    meta = inspect_hdf5(input_path)

    if fps is None:
        fps = meta["fps"] or 20
        print(f"Using fps={fps} from HDF5 metadata")

    if robot_type is None:
        robot_type = meta["robot_type"]

    demo_keys = meta["demo_keys"]
    if split and split in meta["masks"]:
        allowed = set(meta["masks"][split])
        demo_keys = [k for k in demo_keys if k in allowed]
        print(f"Using split '{split}': {len(demo_keys)} episodes")

    if max_episodes:
        demo_keys = demo_keys[:max_episodes]
        print(f"Limiting to {max_episodes} episodes")

    features = build_features(meta)
    has_images = len(meta["image_keys"]) > 0

    print(f"Converting {len(demo_keys)} episodes from {Path(input_path).name}")
    print(f"  Features: {list(features.keys())}")
    print(f"  Images: {meta['image_keys'] if has_images else 'none'}")
    print(f"  Robot: {robot_type}, FPS: {fps}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        use_videos=has_images,
        root=output_dir,
    )

    with h5py.File(input_path, "r") as f:
        for ep_idx, demo_key in enumerate(demo_keys):
            demo = f[f"data/{demo_key}"]
            actions = demo["actions"][:].astype(np.float32)
            n_frames = actions.shape[0]

            obs_arrays = {}
            for obs_key in meta["obs_keys"]:
                if obs_key in SKIP_OBS_KEYS:
                    continue
                if obs_key in OBS_KEY_MAP:
                    lr_name = OBS_KEY_MAP[obs_key]
                elif obs_key.endswith("_image"):
                    camera_name = obs_key.replace("_image", "")
                    lr_name = f"observation.images.{camera_name}"
                else:
                    lr_name = f"observation.{obs_key}"

                data = demo[f"obs/{obs_key}"][:]
                if not obs_key.endswith("_image"):
                    data = data.astype(np.float32)
                obs_arrays[lr_name] = data

            for t in range(n_frames):
                frame = {"action": actions[t], "task": task}
                for lr_name, arr in obs_arrays.items():
                    frame[lr_name] = arr[t]
                dataset.add_frame(frame)

            dataset.save_episode()

            if (ep_idx + 1) % 20 == 0 or ep_idx == len(demo_keys) - 1:
                print(f"  [{ep_idx + 1}/{len(demo_keys)}] episodes converted")

    dataset.finalize()
    print(f"Done. Dataset saved to: {dataset.root}")
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert robomimic HDF5 to LeRobot v2 format",
    )
    parser.add_argument("--input", required=True, help="Path to .hdf5 file")
    parser.add_argument("--repo-id", required=True, help="Dataset name (e.g. user/robomimic_can)")
    parser.add_argument("--task", default="manipulation", help="Task description string")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS")
    parser.add_argument("--split", default=None, help="Only convert this split (e.g. train, valid)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to convert")
    parser.add_argument("--robot-type", default=None, help="Override robot type")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print HDF5 structure, don't convert",
    )
    args = parser.parse_args()

    if args.inspect_only:
        meta = inspect_hdf5(args.input)
        print(f"File: {args.input}")
        print(f"Episodes: {meta['num_demos']}")
        print(f"FPS: {meta['fps']}")
        print(f"Env: {meta['env_name']}")
        print(f"Robot: {meta['robot_type']}")
        print(f"Action shape: {meta['action_shape']}")
        print("\nObs keys:")
        for k, v in meta["obs_shapes"].items():
            skip = " [SKIP]" if k in SKIP_OBS_KEYS else ""
            mapped = f" → {OBS_KEY_MAP[k]}" if k in OBS_KEY_MAP else ""
            print(f"  {k}: shape={v['shape']} dtype={v['dtype']}{mapped}{skip}")
        print(f"\nSplits: {list(meta['masks'].keys())}")
        print("\nLeRobot features that would be created:")
        features = build_features(meta)
        for k, v in features.items():
            print(f"  {k}: dtype={v['dtype']} shape={v['shape']}")
        return

    convert(
        input_path=args.input,
        repo_id=args.repo_id,
        task=args.task,
        output_dir=args.output_dir,
        fps=args.fps,
        split=args.split,
        max_episodes=args.max_episodes,
        robot_type=args.robot_type,
    )


if __name__ == "__main__":
    main()
