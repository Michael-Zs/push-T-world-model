import pytest
import torch

from push_t_jepa.model import JEPAModel


def test_jepa_predicts_embedding_and_stops_target_gradient():
    model = JEPAModel()
    prediction, target = model(
        torch.rand(2, 3, 64, 64),
        torch.rand(2, 4, 2),
        torch.rand(2, 3, 64, 64),
    )
    assert prediction.shape == (2, 64)
    assert target.shape == (2, 64)
    assert target.requires_grad is False


def test_ema_moves_target_parameters_after_context_change():
    model = JEPAModel()
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    model.update_target_encoder(momentum=0.5)
    assert not torch.equal(before, next(model.target_encoder.parameters()))


def test_jepa_rejects_wrong_action_horizon():
    with pytest.raises(ValueError, match="动作"):
        JEPAModel()(
            torch.rand(1, 3, 64, 64),
            torch.rand(1, 3, 2),
            torch.rand(1, 3, 64, 64),
        )
