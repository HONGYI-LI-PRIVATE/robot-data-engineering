"""Smoke test for SwanLab integration.

Usage:
    python -m scripts.swanlab_smoke_test
"""

import swanlab


def main():
    run = swanlab.init(
        project="robot-data-engineering",
        experiment_name="smoke-test",
    )

    for step in range(20):
        swanlab.log(
            {
                "train/loss": 1.0 / (step + 1),
                "train/lr": 1e-4 * (0.95**step),
                "eval/success_rate": min(step * 0.05, 1.0),
            },
            step=step,
        )

    run.finish()
    print("SwanLab smoke test complete!")


if __name__ == "__main__":
    main()
