from pathlib import Path
import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_latest_event_file(log_dir="results/tensorboard_logs"):
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    event_files = list(log_path.rglob("events.out.tfevents.*"))
    if not event_files:
        return None

    event_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return event_files[0]


def load_tensorboard_scalars(log_dir="results/tensorboard_logs"):
    event_file = find_latest_event_file(log_dir)
    if event_file is None:
        return None, None

    ea = EventAccumulator(str(event_file))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    return ea, tags


def extract_scalar_dataframe(log_dir="results/tensorboard_logs", preferred_tags=None):
    if preferred_tags is None:
        preferred_tags = [
            "rollout/ep_rew_mean",
            "train/value_loss",
            "train/policy_gradient_loss",
            "train/entropy_loss",
        ]

    ea, tags = load_tensorboard_scalars(log_dir)
    if ea is None:
        return None

    found = [tag for tag in preferred_tags if tag in tags]
    if not found:
        return None

    dfs = []
    for tag in found:
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events],
            "tag": tag
        })
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)