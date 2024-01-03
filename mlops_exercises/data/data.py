import glob
import os

import torch
from torchvision import transforms


class CorruptedMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.folder = "data/processed/corruptmnist"
        # self.images_list = glob.glob(f'{folder}/{"train" if train else "test"}_images**.pt', recursive = True)
        # self.labels_list = glob.glob(os.path.join(folder, f'{"train" if train else "test"}_target*.pt'), recursive = True)
        # if train:
        #     self.images_list = [os.path.join(folder, f'train_images_{i}.pt') for i in range(6)]
        #     self.labels_list = [os.path.join(folder, f'train_target_{i}.pt') for i in range(6)]
        # else:
        #     self.images_list = [os.path.join(folder, f'test_images.pt')]
        #     self.labels_list = [os.path.join(folder, f'test_target.pt')]
        # print("Found {} images and {} labels".format(len(self.images_list), len(self.labels_list)))

        if train:
            self.images = torch.load(os.path.join(self.folder, "train_images.pt"))
            self.labels = torch.load(os.path.join(self.folder, "train_target.pt"))
        else:
            self.images = torch.load(os.path.join(self.folder, "test_images.pt"))
            self.labels = torch.load(os.path.join(self.folder, "test_target.pt"))

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(dtype = torch.),
        # ])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # images = torch.load(self.images_list[idx])
        # labels = torch.load(self.labels_list[idx])
        # return images, labels
        return self.images[idx], self.labels[idx].long()


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)
    # return train, test

    # mnist_path = os.path.join('data', 'corruptmnist')
    mnist_path = os.path.join("data", "processed", "corruptmnist")

    train_set = CorruptedMNIST(train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=5000)

    test_set = CorruptedMNIST(train=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=5000)

    return train_loader, test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train, test = mnist()
    for images, labels in train:
        print(images.shape)
        print(labels.shape)

        # display some examples
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i in range(25):
            ax = axes.flatten()[i]
            ax.imshow(images[i].view(28, 28), cmap="gray")
            ax.set_axis_off()
            ax.set_title(labels[i])

        plt.show()

        break
    for images, labels in test:
        print(images.shape)
        print(labels.shape)
        break
