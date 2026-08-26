import pytest

from push_t_jepa.config import CEMConfig, EnvConfig, ModelConfig


def test_default_cpu_configuration_has_design_values():
    assert EnvConfig(image_size=256).image_size == 256
    assert ModelConfig().embedding_dim == 64
    assert CEMConfig().population == 128


def test_cem_rejects_more_elites_than_candidates():
    with pytest.raises(ValueError, match="精英"):
        CEMConfig(population=4, elite_count=5)


def test_physics_configuration_has_fixed_substeps():
    config = EnvConfig()
    assert config.physics_substeps == 8
    assert config.physics_dt == 1 / 240
