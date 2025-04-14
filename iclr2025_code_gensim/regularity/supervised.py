#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We *need* to access private methods to change last layer of NNs:
# pylint: disable=W0212

"""
Usage:
  retrain.py (-h | --help)
  retrain.py [options]

Options:
  -h, --help              Show this screen.
  --all                   Train on all shapes or nameable only
  --epochs=<e>            Number of epochs [default: 10].
  --start-from=<f>        Model to load/start from
                          [default: ./cornet/cornet_s_epoch43.pth.tar].
  --split=<s>             Test/Val ratio [default: 0.8].
  --seed=<e>              Seed [default: 1]
"""

import time
import copy
import docopt
import torch
import torchvision
from tqdm import tqdm 
import math
from collections import OrderedDict
from torch import nn


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here for no good reason
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model




def load_cornet(restore_path, n_lab):
    """
    Restores the model and the optimizer from a file, and updates them to add
    extra labels whose number is dictated by `n_lab`. This has to be written on
    a neural network basis.
    """
    model = CORnet_S()
    model = torch.nn.DataParallel(model)
    ckpt_data = None
    if torch.cuda.is_available():
        model = model.cuda()
        ckpt_data = torch.load(restore_path)
    else:
        ckpt_data = torch.load(restore_path, map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"])
    model = model.module

    optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)

    # This hacks into cornet and updates the last layer with n_lab new neurons
    last_layer = model._modules["decoder"]._modules["linear"]

    BACKUP_WEIGHT = last_layer.weight.data
    BACKUP_BIAS = last_layer.bias.data

    new_last_layer = torch.nn.Linear(512, 1000 + n_lab)

    new_last_layer.weight.data[:1000, :] = BACKUP_WEIGHT
    new_last_layer.bias.data[:1000] = BACKUP_BIAS

    model._modules["decoder"]._modules["linear"] = new_last_layer

    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    save_model(model, "./steps/retrained_-1.pth.tar")

    return model, optimizer


def load_data(data_folder, split):
    """
    For a given folder two lengths, loads the images in the folder and split
    them in train dataset and val dataset of given length. Should be updated to
    compute mixture alone
    """

    img_dataset = torchvision.datasets.ImageFolder(
        data_folder,
        torchvision.transforms.ToTensor(),
        target_transform=(lambda x: x + 1000),
    )

    split_train = round(split * len(img_dataset))
    split_val = len(img_dataset) - split_train

    train_set, val_set = torch.utils.data.random_split(
        img_dataset, [split_train, split_val]
    )

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_set, batch_size=4, shuffle=True, num_workers=8
        ),
        "val": torch.utils.data.DataLoader(
            val_set, batch_size=4, shuffle=True, num_workers=8
        ),
    }
    dataset_sizes = {"train": len(train_set), "val": len(val_set)}

    return dataloaders, dataset_sizes


def save_model(model, fname):
    """Saves a given model at a given filename"""
    ckpt_data = {}

    ckpt_data["state_dict"] = model.state_dict()
    torch.save(ckpt_data, fname)


def train_model(model, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    """
    Trains a given model on some data with some parameters for some epochs.
    """

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    since = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                for children in model.modules():
                    # See
                    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
                    # and what the training flag does to batchnorm
                    if isinstance(children, torch.nn.BatchNorm2d):
                        children.training = False
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).double()
            if phase == "train":
                scheduler.step()

            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{epoch},{phase},{epoch_acc:.4f}")

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        save_model(model, f"steps/retrained_{epoch}.pth.tar")

    print(f"\nTraining complete in {time.time() - since}s")
    print(f"Best val Acc: {best_acc:4f}")

    model.load_state_dict(best_model_wts)
    return model


def main():
    """Main function that loads the arguments and calls the right functions"""

    arguments = docopt.docopt(__doc__)

    category_str = "all" if bool(arguments["--all"]) else "nameable"
    n_lab = 11 if bool(arguments["--all"]) else 5

    # Load model and data
    model, optimizer = load_cornet(arguments["--start-from"], n_lab)
    dataloaders, dataset_sizes = load_data(
        f"train_{category_str}_shapes", float(arguments["--split"])
    )
    seed=int(arguments['--seed'])

    final_model = train_model(
        model,
        optimizer,
        dataloaders,
        dataset_sizes,
        int(arguments["--epochs"]),
    )

    save_name = f"cornet_s_retrained_{category_str}_shapes_{seed}.pth.tar"
    save_model(final_model, save_name)


if __name__ == "__main__":
    main()
