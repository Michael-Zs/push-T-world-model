import numpy as np

from push_t_jepa.env import PushTEnv
from push_t_jepa.model import JEPAModel
from push_t_jepa.planner import CEMPlanner


def test_cem_plan_has_bounded_action_sequence():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    actions = CEMPlanner(JEPAModel(), seed=5).plan(image, image)
    assert actions.shape == (8, 2)
    assert np.all(actions >= -1.0) and np.all(actions <= 1.0)


def test_rollout_replanning_returns_one_observation_per_execution_step():
    env = PushTEnv(seed=9)
    observations = CEMPlanner(JEPAModel(), seed=9).rollout_replan(env, env.reset(), steps=3)
    assert len(observations) == 4
