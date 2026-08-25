import torch

from push_t_jepa.dataset import PushTJEPADataset, collect_trajectories


def test_collected_trajectories_are_seed_deterministic():
    first = collect_trajectories(trajectories=2, steps=7, seed=11)
    second = collect_trajectories(trajectories=2, steps=7, seed=11)
    assert torch.equal(first[0].actions, second[0].actions)


def test_dataset_returns_normalized_action_conditioned_sample():
    trajectories = collect_trajectories(trajectories=1, steps=8, seed=2)
    sample = PushTJEPADataset(trajectories, horizon=4)[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["actions"].shape == (4, 2)
    assert sample["future_image"].shape == (3, 64, 64)
    assert sample["future_image"].max() <= 1.0
