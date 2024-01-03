# When this file runs, it should take the raw data e.g. the corrupted MNIST files from yesterday
# which now should be located in a data/raw folder and process them into a single tensor, normalize
# the tensor and save this intermediate representation to the data/processed folder. By normalization here
# we refer to making sure the images have mean 0 and standard deviation 1.

import os

import torch
from torchvision import datasets, transforms


def process_data(train=True):
    """
    Process the data into a single tensor, normalize the tensor and save this intermediate representation to the data/processed folder.

    Parameters
    ----------
    train : bool
        Whether to process the training data or the test data.

    Returns
    -------
    None
    """

    # setup folders
    folder = "data/raw/corruptmnist"
    os.makedirs("data/processed/corruptmnist", exist_ok=True)

    if train:
        images_list = [f"train_images_{i}.pt" for i in range(10)]
        labels_list = [f"train_target_{i}.pt" for i in range(10)]
    else:
        images_list = [f"test_images.pt"]
        labels_list = [f"test_target.pt"]

    # handle images
    images = torch.empty(0, 28, 28)
    for i, image_path in enumerate(images_list):
        # load the tensor
        if ((i>=6) and train):        
            images = torch.cat((images, torch.load(os.path.join(folder+"_v2", image_path))), dim=0)
        else:
            images = torch.cat((images, torch.load(os.path.join(folder, image_path))), dim=0)

    # normalize the tensor
    normalise = transforms.Normalize(mean=images.mean(dim=(1, 2)), std=images.std(dim=(1, 2)))
    images = normalise(images)

    # save the tensor to the data/processed folder
    torch.save(images, os.path.join("data/processed/corruptmnist", "train_images.pt" if train else "test_images.pt"))

    # handle labels
    labels = torch.empty(0)
    for i, label_path in enumerate(labels_list):
        # load the tensor
        if ((i>=6) and train):        
            labels = torch.cat((labels, torch.load(os.path.join(folder+"_v2", label_path))))
        else:
            labels = torch.cat((labels, torch.load(os.path.join(folder, label_path))))

    # save the tensor to the data/processed folder
    torch.save(labels, os.path.join("data/processed/corruptmnist", "train_target.pt" if train else "test_target.pt"))


if __name__ == "__main__":
    # Get the data and process it

    process_data(train=True)
    process_data(train=False)
