import numpy as np
import pytest

from push_t_jepa.config import EnvConfig
from push_t_jepa.env import PushTEnv


def test_reset_is_seed_deterministic_and_returns_rgb_image():
    first = PushTEnv(seed=3).reset()
    second = PushTEnv(seed=3).reset()
    assert first.shape == (EnvConfig().image_size, EnvConfig().image_size, 3)
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


def test_pusher_never_penetrates_t_while_pushing():
    env = PushTEnv(seed=0)
    env.reset()
    env.set_state(
        pusher=np.array([0.36, 0.50]),
        object_position=np.array([0.50, 0.50]),
        object_angle=0.0,
    )
    for _ in range(5):
        env.step(np.array([1.0, 0.0]))
    assert env.pusher_clearance() >= -0.002
    assert env.state.object_position[0] > 0.50


def test_moving_away_from_contact_does_not_drag_t_object():
    env = PushTEnv(seed=0)
    env.reset()
    env.set_state(
        pusher=np.array([0.415, 0.50]),
        object_position=np.array([0.50, 0.50]),
        object_angle=0.0,
    )
    env.step(np.array([-1.0, 0.0]))
    assert np.linalg.norm(env.state.object_position - np.array([0.50, 0.50])) < 0.002


def test_render_uses_pymunk_world_coordinates_for_t_shape():
    env = PushTEnv(EnvConfig(image_size=64), seed=0)
    env.set_state(np.array([0.2, 0.2]), np.array([0.5, 0.5]), 0.0)
    image = env.render()
    assert np.all(image[32, 32] < np.array([100, 100, 100]))


def test_quasistatic_contact_has_no_motion_after_zero_action():
    env = PushTEnv(EnvConfig(image_size=64), seed=0)
    env.set_state(np.array([0.36, 0.55]), np.array([0.50, 0.50]), 0.0)
    env.step(np.array([1.0, 0.0]))
    after_push = env.state
    env.step(np.array([0.0, 0.0]))
    after_release = env.state
    assert np.allclose(after_release.object_position, after_push.object_position)
    assert after_release.object_angle == after_push.object_angle
