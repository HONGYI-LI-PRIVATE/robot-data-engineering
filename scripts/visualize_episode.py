"""Visualize a single episode from a LeRobot dataset.

Usage:
    python -m scripts.visualize_episode --dataset lerobot/aloha_static_coffee --episode 0
    python -m scripts.visualize_episode \
        --dataset lerobot/aloha_static_coffee --episode 0 --output reports/ep0.mp4

LeRobot also provides a built-in CLI:
    lerobot-dataset-viz --repo-id lerobot/aloha_static_coffee --episode-index 0
"""

import argparse
from pathlib import Path

import imageio
import torch
from lerobot.datasets import LeRobotDataset


def print_dataset_info(dataset: LeRobotDataset, dataset_id: str) -> None:
    print(f"Dataset: {dataset_id}")
    print(f"  Total episodes: {dataset.num_episodes}")
    print(f"  Total frames: {dataset.num_frames}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Features: {list(dataset.features.keys())}")


def get_camera_keys(dataset: LeRobotDataset) -> list[str]:
    return [k for k in dataset.features if "image" in k]


def render_episode_video(
    dataset: LeRobotDataset,
    episode_index: int,
    camera_key: str,
    output_path: Path,
    fps: int,
) -> None:
    from_idx = 0
    to_idx = 0
    count = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["episode_index"].item() == episode_index:
            if count == 0:
                from_idx = i
            to_idx = i
            count += 1
        elif count > 0:
            break

    if count == 0:
        print(f"Episode {episode_index} not found")
        return

    print(f"Episode {episode_index}: {count} frames (indices {from_idx}..{to_idx})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")

    for i in range(from_idx, to_idx + 1):
        frame_data = dataset[i]
        img = frame_data[camera_key]
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype("uint8")
        writer.append_data(img)

    writer.close()
    print(f"Saved video to {output_path}")


def visualize_episode(
    dataset_id: str,
    episode_index: int = 0,
    output_path: str | None = None,
    fps: int | None = None,
) -> Path | None:
    dataset = LeRobotDataset(dataset_id)
    print_dataset_info(dataset, dataset_id)

    camera_keys = get_camera_keys(dataset)
    if not camera_keys:
        print("No camera/image keys found in this dataset.")
        print("Available features:")
        for k, v in dataset.features.items():
            print(f"  {k}: {v}")
        return None

    print(f"  Camera keys: {camera_keys}")

    if fps is None:
        fps = dataset.fps

    if output_path is None:
        output_path = f"reports/episode_{episode_index}.mp4"

    out = Path(output_path)
    render_episode_video(dataset, episode_index, camera_keys[0], out, fps)
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize a LeRobot episode")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID on HF Hub")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: dataset FPS)")
    args = parser.parse_args()

    visualize_episode(args.dataset, args.episode, args.output, args.fps)


if __name__ == "__main__":
    main()
