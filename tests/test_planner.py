import numpy as np
import torch
from torch import nn

from push_t_jepa.env import PushTEnv
from push_t_jepa.model import JEPAModel
from push_t_jepa.planner import CEMPlanner
from push_t_jepa.config import CEMConfig, EnvConfig, ModelConfig


class _DeterministicPredictor(nn.Module):
    """用于确认 CEM 优化方向的可解析 embedding 世界模型。"""

    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(embedding_dim=1, action_horizon=4)
        self.weight = nn.Parameter(torch.zeros(1))

    def encode_context(self, image):
        return torch.zeros((image.shape[0], 1), device=image.device)

    def encode_target(self, image):
        return torch.ones((image.shape[0], 1), device=image.device)

    def predict_from_context(self, context, actions):
        return context + actions[:, :, 0].mean(dim=1, keepdim=True)


def test_cem_plan_has_bounded_action_sequence():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    actions = CEMPlanner(JEPAModel(), seed=5).plan(image, image)
    assert actions.shape == (8, 2)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)


def test_rollout_replanning_returns_one_observation_per_execution_step():
    env = PushTEnv(config=EnvConfig(image_size=64), seed=9)
    observations = CEMPlanner(JEPAModel(), seed=9).rollout_replan(env, env.reset(), steps=3)
    assert len(observations) == 4


def test_cem_favors_actions_that_reduce_deterministic_embedding_distance():
    planner = CEMPlanner(
        _DeterministicPredictor(),
        CEMConfig(horizon=8, population=256, elite_count=24, iterations=5),
        seed=3,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    actions = planner.plan(image, image)
    assert actions[:, 0].mean() > 0.2
