import numpy as np
import pytest

from push_t_jepa.env import PushTEnv


def test_reset_is_seed_deterministic_and_returns_rgb_image():
    first = PushTEnv(seed=3).reset()
    second = PushTEnv(seed=3).reset()
    assert first.shape == (64, 64, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_contact_push_changes_t_object_pose():
    env = PushTEnv(seed=0)
    env.reset()
    env.set_state(
        pusher=np.array([0.42, 0.50]),
        object_position=np.array([0.50, 0.50]),
        object_angle=0.0,
    )
    before = env.state.object_position.copy()
    env.step(np.array([1.0, 0.0]))
    assert env.state.object_position[0] > before[0]


def test_step_clamps_pusher_inside_arena():
    env = PushTEnv(seed=0)
    env.reset()
    env.set_state(
        pusher=np.array([0.99, 0.99]),
        object_position=np.array([0.50, 0.50]),
        object_angle=0.0,
    )
    env.step(np.array([1.0, 1.0]))
    assert np.all(env.state.pusher <= 1.0)


def test_step_rejects_wrong_action_shape():
    with pytest.raises(ValueError, match="动作"):
        PushTEnv(seed=0).step(np.array([0.0]))
