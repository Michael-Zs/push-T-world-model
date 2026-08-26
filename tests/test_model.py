import pytest
import torch

from push_t_jepa.config import ModelConfig
from push_t_jepa.model import JEPAModel, VAEJEPAModel


def test_jepa_predicts_spatial_latent_and_stops_target_gradient():
    model = JEPAModel()
    prediction, target = model(
        torch.rand(2, 3, 64, 64),
        torch.rand(2, 4, 2),
        torch.rand(2, 3, 64, 64),
    )
    assert prediction.shape == (2, 64, 8, 8)
    assert target.shape == (2, 64, 8, 8)
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


def test_jepa_supports_eight_step_action_prediction():
    model = JEPAModel(ModelConfig(action_horizon=8))
    prediction, _ = model(torch.rand(1, 3, 64, 64), torch.rand(1, 8, 2), torch.rand(1, 3, 64, 64))
    assert prediction.shape == (1, 64, 8, 8)


def test_decoder_reconstructs_rgb_image_shape_from_spatial_latent():
    model = JEPAModel()
    reconstruction = model.decode(torch.rand(2, 64, 8, 8))
    assert reconstruction.shape == (2, 3, 64, 64)
    assert torch.all(reconstruction >= 0.0)
    assert torch.all(reconstruction <= 1.0)


def test_decoder_supports_configured_256_pixel_images():
    model = JEPAModel(ModelConfig(image_size=256))
    reconstruction = model.decode(torch.rand(1, 64, 8, 8))
    assert reconstruction.shape == (1, 3, 256, 256)


def test_decoder_rejects_global_embedding_without_spatial_layout():
    with pytest.raises(ValueError, match="embedding"):
        JEPAModel().decode(torch.rand(1, 64))


def test_goal_encoder_ignores_blue_pusher_pixels():
    model = JEPAModel()
    image = torch.tensor((245 / 255, 247 / 255, 250 / 255)).view(1, 3, 1, 1).expand(1, 3, 64, 64).clone()
    image[:, :, 30:35, 30:35] = torch.tensor([40, 120, 220]).view(1, 3, 1, 1) / 255
    background = torch.tensor((245 / 255, 247 / 255, 250 / 255)).view(1, 3, 1, 1).expand_as(image)
    assert torch.allclose(model.encode_goal(image), model.encode_goal(background))


def test_pose_head_predicts_push_t_state_shape_from_spatial_latent():
    model = JEPAModel()
    assert model.predict_pose(torch.rand(2, 64, 8, 8)).shape == (2, 6)


def test_pose_head_preserves_spatial_position_information():
    model = JEPAModel()
    first = torch.zeros(1, 64, 8, 8); first[:, 0, 1, 1] = 1
    second = torch.zeros(1, 64, 8, 8); second[:, 0, 6, 6] = 1
    assert not torch.allclose(model.predict_pose(first), model.predict_pose(second))


def test_vae_encoder_returns_spatial_distribution_and_finite_kl():
    model = VAEJEPAModel(ModelConfig(action_horizon=4))
    mean, logvar = model.encode_distribution(torch.rand(2, 3, 64, 64))
    assert mean.shape == logvar.shape == (2, 64, 8, 8)
    assert torch.isfinite(model.kl_loss(mean, logvar))
