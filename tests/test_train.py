import math

import torch

from push_t_jepa.model import JEPAModel, VAEJEPAModel
from push_t_jepa.train import (
    _foreground_reconstruction_loss,
    jepa_loss,
    load_checkpoint,
    run_training,
    save_checkpoint,
    train_epoch,
    train_vae_epoch,
    validate_pose,
)


def test_one_training_epoch_returns_finite_loss_and_checkpoint_round_trips(tmp_path):
    batch = {
        "image": torch.rand(4, 3, 64, 64),
        "actions": torch.rand(4, 4, 2),
        "future_image": torch.rand(4, 3, 64, 64),
    }
    model = JEPAModel()
    loss = train_epoch(model, [batch], torch.optim.Adam(model.parameters(), lr=1e-3), 0.99)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {"seed": 1}, {"loss": loss})
    loaded = load_checkpoint(path, JEPAModel())
    assert math.isfinite(loss)
    assert loaded["metrics"]["loss"] == loss


def test_variance_regularization_penalizes_collapsed_embeddings():
    collapsed_loss, collapsed_std = jepa_loss(torch.zeros(8, 64), torch.zeros(8, 64), variance_weight=1.0)
    diverse = torch.randn(8, 64)
    diverse_loss, diverse_std = jepa_loss(diverse, diverse, variance_weight=1.0)
    assert collapsed_std < 0.02
    assert collapsed_loss > diverse_loss


def test_foreground_reconstruction_gives_objects_more_weight_than_background():
    target = torch.full((1, 3, 8, 8), 245 / 255)
    target[:, :, 3:5, 3:5] = 0.1
    background_error = target.clone()
    background_error[:, :, 0:2, 0:2] = 0.0
    foreground_error = target.clone()
    foreground_error[:, :, 3:5, 3:5] = 0.9
    assert _foreground_reconstruction_loss(foreground_error, target) > _foreground_reconstruction_loss(background_error, target)


def test_vae_training_reports_finite_kl_and_reconstruction_losses():
    batch = {
        "image": torch.rand(2, 3, 64, 64),
        "actions": torch.rand(2, 4, 2),
        "future_image": torch.rand(2, 3, 64, 64),
        "state": torch.rand(2, 6),
        "future_state": torch.rand(2, 6),
    }
    model = VAEJEPAModel()
    metrics = train_vae_epoch(model, [batch], torch.optim.Adam(model.parameters(), lr=1e-3), ema_momentum=0.99, kl_weight=1e-3)
    assert metrics["kl_loss"] >= 0.0
    assert math.isfinite(metrics["reconstruction_loss"])


def test_full_vae_training_writes_a_vae_checkpoint(tmp_path):
    checkpoint = run_training(
        output=tmp_path,
        trajectories=2,
        steps=8,
        epochs=1,
        batch_size=2,
        seed=4,
        vae=True,
    )
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["config"]["model_type"] == "vae_jepa"


def test_pose_validation_returns_finite_error_for_labeled_batch():
    batch = {"image": torch.rand(2, 3, 64, 64), "actions": torch.rand(2, 4, 2), "future_image": torch.rand(2, 3, 64, 64), "future_state": torch.rand(2, 6)}
    assert math.isfinite(validate_pose(JEPAModel(), [batch]))


def test_full_training_writes_best_checkpoint_and_history(tmp_path):
    checkpoint = run_training(
        output=tmp_path,
        trajectories=4,
        steps=8,
        epochs=2,
        batch_size=4,
        seed=1,
    )
    assert checkpoint.is_file()
    assert (tmp_path / "history.json").is_file()


def test_loading_pre_decoder_checkpoint_keeps_planning_weights(tmp_path):
    legacy_model = JEPAModel()
    state = {name: value for name, value in legacy_model.state_dict().items() if not name.startswith("decoder.")}
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": state, "config": {}, "metrics": {}}, path)
    info = load_checkpoint(path, JEPAModel())
    assert info["decoder_available"] is False


def test_training_reports_collection_and_epoch_progress(tmp_path):
    messages: list[str] = []
    run_training(tmp_path, trajectories=2, steps=8, epochs=1, batch_size=2, seed=2, progress=messages.append)
    assert any("采集轨迹" in message for message in messages)
    assert any("Epoch 1/1" in message for message in messages)


def test_training_writes_tensorboard_event_file(tmp_path):
    run_training(tmp_path, trajectories=2, steps=8, epochs=1, batch_size=2, seed=3)
    assert any((tmp_path / "tensorboard").glob("events.out.tfevents.*"))
