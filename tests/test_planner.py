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
        return torch.zeros((image.shape[0], 1, 8, 8), device=image.device)

    def encode_target(self, image):
        return torch.ones((image.shape[0], 1, 8, 8), device=image.device)

    def encode_goal(self, image):
        return self.encode_target(image)

    def encode_goal_from_latent(self, latent):
        return latent

    def predict_pose(self, latent):
        pose = torch.zeros((latent.shape[0], 6), device=latent.device)
        pose[:, 2] = latent[:, 0].mean(dim=(1, 2))
        return pose

    def predict_from_context(self, context, actions):
        return context + actions[:, :, 0].mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)


def test_cem_plan_has_bounded_action_sequence():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    actions = CEMPlanner(JEPAModel(), seed=5).plan(image, image)
    assert actions.shape == (8, 2)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)


def test_rollout_replanning_returns_one_observation_per_execution_step():
    env = PushTEnv(config=EnvConfig(image_size=64), seed=9)
    observations = CEMPlanner(JEPAModel(), seed=9).rollout_replan(env, env.reset(), steps=3)
    assert len(observations) == 4


def test_geometry_approach_action_moves_pusher_toward_visible_t_object():
    env = PushTEnv(config=EnvConfig(image_size=64), seed=9)
    env.set_state(
        pusher=np.array([0.20, 0.50]),
        object_position=np.array([0.50, 0.50]),
        object_angle=0.0,
    )
    action = CEMPlanner(JEPAModel(), seed=9).approach_action(env.render())
    assert action is not None
    assert action[0] > 0.9
    assert np.isclose(np.linalg.norm(action), 1.0)


def test_cem_favors_actions_that_reduce_deterministic_embedding_distance():
    planner = CEMPlanner(
        _DeterministicPredictor(),
        CEMConfig(horizon=8, population=256, elite_count=24, iterations=5),
        seed=3,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    actions = planner.plan(image, image)
    assert actions[:, 0].mean() > 0.2


def test_goal_cost_uses_only_t_pose_coordinates():
    planner = CEMPlanner(_DeterministicPredictor(), seed=3)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert planner.goal_cost(image, image) == 1.0


def test_oracle_cem_plan_has_bounded_action_sequence():
    env = PushTEnv(config=EnvConfig(image_size=64), seed=4)
    env.reset()
    actions = CEMPlanner(JEPAModel(), CEMConfig(horizon=8, population=4, elite_count=2, iterations=1), seed=4).plan_oracle(
        env, np.array([0.72, 0.62], dtype=np.float32), target_angle=0.45
    )
    assert actions.shape == (8, 2)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)


def test_oracle_cem_can_select_no_op_at_current_goal():
    env = PushTEnv(config=EnvConfig(image_size=64), seed=6)
    env.reset()
    state = env.state
    actions = CEMPlanner(JEPAModel(), CEMConfig(horizon=8, population=4, elite_count=2, iterations=1), seed=6).plan_oracle(
        env, state.object_position, target_angle=state.object_angle
    )
    assert np.allclose(actions, 0.0)
