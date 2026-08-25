import math

import torch

from push_t_jepa.model import JEPAModel
from push_t_jepa.train import jepa_loss, load_checkpoint, run_training, save_checkpoint, train_epoch


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
