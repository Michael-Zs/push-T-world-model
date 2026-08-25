from push_t_jepa.demo import run_demo
from push_t_jepa.train import run_smoke_training
import json


def test_smoke_training_and_demo_write_reusable_artifacts(tmp_path):
    checkpoint = run_smoke_training(tmp_path / "train", seed=4)
    output = run_demo(checkpoint, tmp_path / "demo", seed=4, steps=3)
    assert checkpoint.is_file()
    assert (output / "rollout.gif").is_file()
    assert (output / "metrics.json").is_file()
    from PIL import Image
    assert Image.open(output / "rollout.gif").size == (192, 64)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "final_embedding_distance" in metrics
