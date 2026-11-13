"""Smoke tests on the real weights (CPU). Skipped if the weights are missing."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "nutriscan-yolo11s.pt"
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="model weights not present")


@pytest.fixture(scope="module")
def model():
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))


def test_classes_match_calorie_table_and_card(model):
    from nutrition import KCAL_PER_SERVING
    names = set(model.names.values())
    assert len(names) == 17
    assert names <= set(KCAL_PER_SERVING), f"missing calories for {names - set(KCAL_PER_SERVING)}"
    card = json.loads((ROOT / "models" / "model_card.json").read_text())
    assert set(card["classes"]) == names


def test_predict_runs_on_cpu(model):
    import numpy as np
    from PIL import Image
    img = Image.fromarray(np.full((480, 640, 3), 255, dtype=np.uint8))  # blank plate
    res = model.predict(img, device="cpu", verbose=False)[0]
    assert res.boxes is not None and res.orig_shape == (480, 640)
