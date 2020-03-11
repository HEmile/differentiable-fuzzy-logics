import numpy as np
import torch
import dfl.config as config
from torchvision import datasets, transforms

kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}


class GenHelper(torch.utils.data.Dataset):
    def __init__(self, mother, length, mapping):
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    if random_seed != None:
        np.random.seed(random_seed)

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = dslen // split_fold
    print(valid_size)
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid


def get_loaders(seed):
    mnist_dataset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    mnist_unsup, mnist_sup = train_valid_split(
        mnist_dataset, config.split_unsup_sup, seed
    )

    unsup_loader = torch.utils.data.DataLoader(
        mnist_unsup, batch_size=config.batch_size, shuffle=True, **kwargs
    )
    sup_loader = torch.utils.data.DataLoader(
        mnist_sup, batch_size=config.batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=config.test_batch_size,
        shuffle=True,
        **kwargs
    )
    return sup_loader, unsup_loader, test_loader
