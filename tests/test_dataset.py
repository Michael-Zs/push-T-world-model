import torch

from push_t_jepa.config import EnvConfig
from push_t_jepa.dataset import PushTJEPADataset, collect_trajectories, collect_trajectories_with_stats


def test_collected_trajectories_are_seed_deterministic():
    first = collect_trajectories(trajectories=2, steps=7, seed=11)
    second = collect_trajectories(trajectories=2, steps=7, seed=11)
    assert torch.equal(first[0].actions, second[0].actions)


def test_dataset_returns_normalized_action_conditioned_sample():
    trajectories = collect_trajectories(env_config=EnvConfig(image_size=64), trajectories=1, steps=8, seed=2)
    sample = PushTJEPADataset(trajectories, horizon=4)[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["actions"].shape == (4, 2)
    assert sample["future_image"].shape == (3, 64, 64)
    assert sample["state"].shape == (6,)
    assert sample["future_state"].shape == (6,)
    assert sample["future_image"].max() <= 1.0


def test_balanced_collector_generates_many_effective_pushes():
    _, stats = collect_trajectories_with_stats(trajectories=30, steps=32, seed=5)
    assert stats.effective_step_rate >= 0.35
    assert stats.mean_translation > 0.01
    assert stats.mean_rotation > 0.01
