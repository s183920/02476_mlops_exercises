import os

import pytest
import torch

from tests import _PATH_DATA, _PROJECT_ROOT

MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "trained_model.pt")

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Data files not found")
def test_model():

    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    x = torch.randn(5000, 1, 28, 28)

    pred = model(x)
    assert pred.shape == (5000, 10), "Prediction should have shape (5000, 10) but has {}".format(pred.shape)
