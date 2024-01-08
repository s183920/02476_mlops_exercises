import os

import pytest

from mlops_exercises.data.data import mnist
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed", "corruptmnist")), reason="Data files not found")
@pytest.mark.parametrize("batch_size", [1, 1000, 5000])
def test_data(batch_size):
    train_loader, test_loader = mnist(batch_size=batch_size)
    
    assert len(train_loader.dataset) == 50000 , "Train dataset should have 50k images but has {}".format(len(train_loader.dataset))
    assert len(test_loader.dataset) == 5000, "Test dataset should have 5k images but has {}".format(len(test_loader.dataset))
    
    img, lbl = next(iter(train_loader))
    assert img.shape == (batch_size, 28, 28), "Train images should have shape ({}, 28, 28) but have {}".format(batch_size, img.shape)
    assert lbl.shape == (batch_size,), "Train labels should have shape ({},) but have {}".format(batch_size, lbl.shape)
    
    img, lbl = next(iter(test_loader))
    assert img.shape == (batch_size, 28, 28), "Test images should have shape ({}, 28, 28) but have {}".format(batch_size, img.shape)
    assert lbl.shape == (batch_size,), "Test labels should have shape ({},) but have {}".format(batch_size, lbl.shape)
    # dataset = MNIST(...)
    # assert len(dataset) == N_train for training and N_test for test
    # assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    # assert that all labels are represented
    
# @pytest.mark.skipif(not os.path.exists(os.path.join(_PATH_DATA, "processed", "corruptmnist")), reason="Data files not found")
# @pytest.mark.parametrize("batch_size", [1, 1000, 10000])
# def test_batch_size(batch_size):
#     train_loader, test_loader = mnist(batch_size=batch_size)
    
#     assert 