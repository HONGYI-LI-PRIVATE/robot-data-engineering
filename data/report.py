"""报告生成器：一键生成数据集体检报告。

使用方式：

    # Python 调用
    from data.report import robdata_analyse
    robdata_analyse("data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5")
    robdata_analyse("data/lerobot/aloha_static_coffee")

    # 命令行
    python -m data.report --dataset data/robomimic/can_ph/v1.5/can/ph/low_dim_v15.hdf5
"""

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接写文件
import matplotlib.pyplot as plt
import numpy as np

from . import analysis
from .loader import DatasetInfo, load_dataset


def robdata_analyse(path: str, output_dir: str = "reports") -> str:
    """一键生成数据集体检报告。

    自动检测格式（HDF5 或 LeRobot），跑全部分析项，
    输出一个自包含的 HTML 文件（图表直接嵌在里面，不依赖外部文件）。

    Args:
        path: 数据集路径（.hdf5 文件或 LeRobot 目录）
        output_dir: 报告输出目录

    Returns:
        报告文件路径
    """
    print(f"Loading dataset: {path}")
    info = load_dataset(path)
    print(
        f"  {info.format} | {info.num_episodes} episodes | "
        f"{info.total_frames} frames | {info.action_dim}D actions"
    )

    print("Running analysis...")
    results = {
        "basic": analysis.basic_stats(info),
        "actions": analysis.action_analysis(info),
        "vel_acc": analysis.velocity_acceleration(info),
        "timestamps": analysis.timestamp_check(info),
        "dead": analysis.dead_episodes(info),
    }

    print("Generating report...")
    html = _render_html(info, results)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"report_{info.name}_{datetime.now():%Y%m%d_%H%M%S}.html"
    report_file.write_text(html, encoding="utf-8")

    print(f"Report saved to: {report_file}")
    return str(report_file)


# ─── 绘图函数 ───────────────────────────────────────────────


def _fig_to_base64(fig) -> str:
    """把 matplotlib 图表转成 base64 字符串，嵌入 HTML 时不需要额外图片文件。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _plot_episode_lengths(basic: dict) -> str:
    """Episode 长度分布直方图，红色虚线标注分位数。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    lengths = basic["episode_lengths"]
    ax.hist(lengths, bins=min(30, len(lengths)), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Episode Length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution")

    for label, val in basic["length_percentiles"].items():
        ax.axvline(val, color="red", linestyle="--", alpha=0.5)
        ax.text(
            val,
            ax.get_ylim()[1] * 0.9,
            label,
            rotation=90,
            va="top",
            fontsize=8,
            color="red",
        )

    return _fig_to_base64(fig)


def _plot_action_dims(actions: dict) -> str:
    """每个动作维度的箱线图，二值化维度标红。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    all_actions = actions["all_actions"]
    n_dims = all_actions.shape[1]

    ax.boxplot(
        [all_actions[:, d] for d in range(n_dims)],
        tick_labels=[f"dim_{d}" for d in range(n_dims)],
    )
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Value")
    ax.set_title("Action Distribution per Dimension")

    for dim_info in actions["per_dim"]:
        if dim_info["is_binary"]:
            ax.text(
                dim_info["dim"] + 1,
                ax.get_ylim()[1],
                "BIN",
                ha="center",
                fontsize=8,
                color="red",
                fontweight="bold",
            )

    return _fig_to_base64(fig)


def _plot_velocity_acceleration(vel_acc: dict) -> str:
    """速度和加速度分布，红色虚线标注异常阈值。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_vel = vel_acc["all_velocities"]
    if all_vel.size > 0:
        vel_norms = np.linalg.norm(all_vel, axis=1)
        axes[0].hist(vel_norms, bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Velocity Norm")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Joint Velocity Distribution")

    all_acc = vel_acc["all_accelerations"]
    if all_acc.size > 0:
        acc_norms = np.linalg.norm(all_acc, axis=1)
        axes[1].hist(acc_norms, bins=50, edgecolor="black", alpha=0.7, color="orange")

        threshold_val = float(np.mean(acc_norms) + 3 * np.std(acc_norms))
        axes[1].axvline(
            threshold_val,
            color="red",
            linestyle="--",
            label=f"3σ = {threshold_val:.4f}",
        )
        axes[1].legend()
        axes[1].set_xlabel("Acceleration Norm")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Joint Acceleration Distribution")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_timestamps(ts: dict) -> str:
    """帧间隔分布，标出预期间隔和掉帧阈值。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    intervals = ts["all_intervals"]
    ax.hist(intervals, bins=50, edgecolor="black", alpha=0.7, color="green")

    expected = ts["expected_interval"]
    if expected:
        ax.axvline(expected, color="red", linestyle="--", label=f"Expected: {expected:.4f}s")
        ax.axvline(
            expected * 1.5,
            color="orange",
            linestyle="--",
            label=f"Drop threshold: {expected * 1.5:.4f}s",
        )
        ax.legend()

    ax.set_xlabel("Frame Interval (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Frame Interval Distribution")
    return _fig_to_base64(fig)


def _plot_dead_episodes(dead: dict) -> str:
    """每个 episode 的动作总方差，废 episode 标红。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    variances_list = dead["variances"]

    total_vars = [v["total_variance"] for v in variances_list]
    colors = [
        "red" if v["total_variance"] < dead["dead_threshold"] else "steelblue"
        for v in variances_list
    ]

    ax.bar(range(len(total_vars)), total_vars, color=colors, edgecolor="none")
    ax.axhline(
        dead["dead_threshold"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Dead threshold: {dead['dead_threshold']:.6f}",
    )
    ax.legend()
    ax.set_xlabel("Episode (sorted by variance, smallest first)")
    ax.set_ylabel("Total Action Variance")
    ax.set_title(f"Episode Action Variance ({dead['dead_count']} dead episodes in red)")
    return _fig_to_base64(fig)


# ─── HTML 渲染 ───────────────────────────────────────────────


def _render_html(info: DatasetInfo, results: dict) -> str:
    basic = results["basic"]
    actions = results["actions"]
    vel_acc = results["vel_acc"]
    ts = results["timestamps"]
    dead = results["dead"]

    # 生成所有图表
    img_lengths = _plot_episode_lengths(basic)
    img_actions = _plot_action_dims(actions)
    img_acc = _plot_velocity_acceleration(vel_acc)
    img_dead = _plot_dead_episodes(dead)
    img_ts = _plot_timestamps(ts) if ts else None

    # ── 动作维度表格 ──
    action_rows = ""
    for d in actions["per_dim"]:
        binary_tag = (
            " <span style='color:red;font-weight:bold'>BINARY</span>" if d["is_binary"] else ""
        )
        action_rows += (
            f"<tr><td>{d['dim']}</td><td>{d['mean']:.4f}</td><td>{d['std']:.4f}</td>"
            f"<td>{d['min']:.4f}</td><td>{d['max']:.4f}</td>"
            f"<td>{d['unique_count']}{binary_tag}</td></tr>\n"
        )

    # ── 异常帧 Top 10 表格 ──
    anomaly_rows = ""
    top_anomalies = sorted(
        vel_acc["anomaly_episodes"],
        key=lambda x: x.get("anomaly_ratio", 0),
        reverse=True,
    )[:10]
    for a in top_anomalies:
        anomaly_rows += (
            f"<tr><td>Episode {a['episode']}</td><td>{a['anomaly_frames']}</td>"
            f"<td>{a['total_frames']}</td><td>{a['anomaly_ratio']:.1%}</td></tr>\n"
        )

    # ── 废 episode 表格 ──
    dead_rows = ""
    for d_ep in dead["dead_episodes"]:
        dead_rows += (
            f"<tr><td>Episode {d_ep['episode']}</td>"
            f"<td>{d_ep['total_variance']:.6f}</td>"
            f"<td>{d_ep['length']}</td></tr>\n"
        )
    if not dead_rows:
        dead_rows = "<tr><td colspan='3'>No dead episodes found</td></tr>"

    # ── 时间戳部分 ──
    if ts:
        ts_section = f"""
        <h2>4. Timestamp Check</h2>
        <table>
            <tr><td>Expected Interval</td>
                <td>{ts["expected_interval"]:.4f}s ({info.fps} fps)</td></tr>
            <tr><td>Actual Mean Interval</td>
                <td>{ts["interval_mean"]:.4f}s</td></tr>
            <tr><td>Interval Std</td>
                <td>{ts["interval_std"]:.6f}s</td></tr>
            <tr><td>Min / Max Interval</td>
                <td>{ts["interval_min"]:.4f}s / {ts["interval_max"]:.4f}s</td></tr>
            <tr><td>Total Dropped Frames</td><td>{ts["total_drops"]}</td></tr>
        </table>
        <img src="data:image/png;base64,{img_ts}">
        """
    else:
        ts_section = """
        <h2>4. Timestamp Check</h2>
        <p>No timestamps available (HDF5 format typically does not include timestamps).</p>
        """

    # ── 组装完整 HTML ──
    gripper_label = "Binary (open/close)" if actions["gripper_is_binary"] else "Continuous"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Dataset Report: {info.name}</title>
<style>
    body {{
        font-family: -apple-system, "Segoe UI", sans-serif;
        max-width: 1000px; margin: 40px auto; padding: 0 20px;
        line-height: 1.6; color: #333;
    }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
    h2 {{ color: #555; margin-top: 40px; }}
    h3 {{ color: #666; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    img {{ max-width: 100%; margin: 15px 0; }}
    .summary {{
        background: #f0f7ff; padding: 15px; border-radius: 8px; margin: 20px 0;
        display: flex; gap: 20px; flex-wrap: wrap;
    }}
    .summary span {{ white-space: nowrap; }}
</style>
</head>
<body>

<h1>Dataset Report: {info.name}</h1>
<div class="summary">
    <span><strong>Format:</strong> {info.format}</span>
    <span><strong>Robot:</strong> {info.robot_type or "N/A"}</span>
    <span><strong>FPS:</strong> {info.fps or "N/A"}</span>
    <span><strong>Cameras:</strong> {
        ", ".join(info.camera_names) if info.camera_names else "N/A"
    }</span>
    <span><strong>Generated:</strong> {datetime.now():%Y-%m-%d %H:%M}</span>
</div>

<h2>1. Basic Statistics</h2>
<table>
    <tr><td>Episodes</td><td>{basic["num_episodes"]}</td></tr>
    <tr><td>Total Frames</td><td>{basic["total_frames"]}</td></tr>
    <tr><td>Action Dimensions</td><td>{basic["action_dim"]}</td></tr>
    <tr><td>Episode Length (mean &plusmn; std)</td>
        <td>{basic["length_mean"]:.1f} &plusmn; {basic["length_std"]:.1f} frames</td></tr>
    <tr><td>Episode Length (min / max)</td>
        <td>{basic["length_min"]} / {basic["length_max"]} frames</td></tr>
    <tr><td>Percentiles (P25 / P50 / P75 / P95)</td>
        <td>{basic["length_percentiles"]["P25"]:.0f} / \
{basic["length_percentiles"]["P50"]:.0f} / \
{basic["length_percentiles"]["P75"]:.0f} / \
{basic["length_percentiles"]["P95"]:.0f}</td></tr>
</table>
<img src="data:image/png;base64,{img_lengths}">

<h2>2. Action Space Analysis</h2>
<table>
    <tr><th>Dim</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Unique Values</th></tr>
    {action_rows}
</table>
<p><strong>Gripper (dim {actions["gripper_dim"]}):</strong> {gripper_label}</p>
<img src="data:image/png;base64,{img_actions}">

<h2>3. Velocity &amp; Acceleration Analysis</h2>
<p>Anomaly threshold: acceleration deviation &gt; \
{vel_acc["anomaly_threshold"]}&sigma; from mean</p>
<img src="data:image/png;base64,{img_acc}">

<h3>Top 10 Episodes by Anomaly Ratio (jitter)</h3>
<table>
    <tr><th>Episode</th><th>Anomaly Frames</th><th>Total Frames</th><th>Ratio</th></tr>
    {anomaly_rows}
</table>

{ts_section}

<h2>5. Dead Episode Detection</h2>
<p>Dead threshold: total variance &lt; {dead["dead_threshold"]:.6f}
   (1%% of median variance {dead["median_variance"]:.4f})</p>
<img src="data:image/png;base64,{img_dead}">

<h3>Dead Episodes ({dead["dead_count"]} found)</h3>
<table>
    <tr><th>Episode</th><th>Total Variance</th><th>Length</th></tr>
    {dead_rows}
</table>

</body>
</html>"""

    return html


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset analysis report")
    parser.add_argument("--dataset", required=True, help="Path to dataset (.hdf5 or LeRobot dir)")
    parser.add_argument("--output", default="reports", help="Output directory")
    args = parser.parse_args()

    robdata_analyse(args.dataset, args.output)
