"""Smoke test for Weights & Biases integration.

Usage:
    python -m scripts.wandb_smoke_test

Requires WANDB_API_KEY to be set or `wandb login` to have been run.
"""

import wandb


def main():
    run = wandb.init(
        project="robot-data-engineering",
        name="smoke-test",
        tags=["week1", "setup"],
    )

    for step in range(20):
        wandb.log(
            {
                "train/loss": 1.0 / (step + 1),
                "train/lr": 1e-4 * (0.95**step),
                "eval/success_rate": min(step * 0.05, 1.0),
            },
            step=step,
        )

    wandb.log({"summary/total_episodes": 50, "summary/dataset": "aloha_static_coffee"})

    run.finish()
    print(f"W&B smoke test complete. Run URL: {run.url}")


if __name__ == "__main__":
    main()
