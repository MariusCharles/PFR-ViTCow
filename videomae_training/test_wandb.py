#!/usr/bin/env python3
"""Quick test to verify W&B is working"""

import sys

print("Testing W&B installation and logging...")

try:
    import wandb

    print(f"✅ wandb imported successfully (version: {wandb.__version__})")
except ImportError as e:
    print(f"❌ Failed to import wandb: {e}")
    print("Install with: pip install wandb")
    sys.exit(1)

# Test initialization
print("\nTesting wandb.init()...")
try:
    run = wandb.init(
        project="test-project",
        name="test-run",
        mode="online",  # or "offline" for testing without internet
    )
    print("✅ W&B initialized successfully")
    print(f"   Run URL: {run.url}")
    print(f"   Run ID: {run.id}")
except Exception as e:
    print(f"❌ Failed to initialize wandb: {e}")
    sys.exit(1)

# Test logging
print("\nTesting wandb.log()...")
try:
    test_metrics = {
        "train/loss": 1.234,
        "train/lr": 0.0001,
        "val/loss": 0.987,
        "epoch": 0,
    }
    wandb.log(test_metrics, step=0)
    print(f"✅ Logged {len(test_metrics)} metrics successfully")
    print(f"   Metrics: {list(test_metrics.keys())}")
except Exception as e:
    print(f"❌ Failed to log to wandb: {e}")
    wandb.finish()
    sys.exit(1)

# Finish
print("\nClosing W&B run...")
wandb.finish()
print("✅ Test completed successfully!")
print("\nCheck your metrics at: https://wandb.ai")
