import math

import torch

from push_t_jepa.model import JEPAModel
from push_t_jepa.train import load_checkpoint, save_checkpoint, train_epoch


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
