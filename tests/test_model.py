from tests import _PATH_DATA, _PROJECT_ROOT
import torch
import os
from mlops_exercises.data.data import mnist


def test_model():

    model = torch.load(os.path.join(_PROJECT_ROOT, "models", "trained_model.pt"), map_location=torch.device('cpu'))
    train_loader, test_loader = mnist()
    
    pred = model(next(iter(train_loader))[0])
    assert pred.shape == (5000, 10), "Prediction should have shape (5000, 10) but has {}".format(pred.shape)
    
    pred = model(next(iter(test_loader))[0])
    assert pred.shape == (5000, 10), "Prediction should have shape (5000, 10) but has {}".format(pred.shape)
    