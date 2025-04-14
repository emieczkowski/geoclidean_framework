#!/usr/bin/env python
# -*- coding: utf-8 -*-

# We *need* to access private methods to change last layer of NNs:
# pylint: disable=W0212

"""
Tests a net on imagenet-val over many epochs
"""

import logging
import torch
import numpy as np
import torchvision

import cornet.cornet_s


def load_imagenet(img_path="imagenet", num_items=50):
    """
    loads imagenet
    """

    img_dataset = torchvision.datasets.ImageFolder(
        img_path,
        torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    # Only loads the first num_items images per label:
    mask = [
        i
        for j in range(0, num_items * 1000, num_items)
        for i in range(j, j + num_items)
    ]
    # Just to test: load less stuff!
    # mask = [mask[i] for i in range(100)]
    img_dataset.samples = [img_dataset.samples[idx] for idx in mask]
    img_dataset.targets = [img_dataset.targets[idx] for idx in mask]

    return torch.utils.data.DataLoader(
        img_dataset,
        batch_size=num_items,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def test_on_dataset(net, validation_gen, this_epoch):
    """
    tests models on imagenet
    """

    ckpt_data = None
    restore_path = f"./steps/retrained_{this_epoch}.pth.tar"

    if torch.cuda.is_available():
        net = net.cuda()
        ckpt_data = torch.load(restore_path)
    else:
        ckpt_data = torch.load(restore_path, map_location="cpu")

    new_last_layer = torch.nn.Linear(512, 1000 + 5)
    net.module._modules["decoder"]._modules["linear"] = new_last_layer
    net.load_state_dict(ckpt_data["state_dict"])
    net = net.module

    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net = net.cuda()

    net.eval()

    lab = 0
    scores = np.zeros(1000)
    with torch.set_grad_enabled(False):
        for local_batch_val, local_labels_val in validation_gen:

            if torch.cuda.is_available():
                local_labels_val = local_labels_val.cuda(non_blocking=True)

            pred_val = net(local_batch_val)
            top1 = pred_val.topk(1, dim=1, largest=True, sorted=True)[1]

            score = (top1 == local_labels_val).sum().to(dtype=torch.float)
            score = score / len(local_labels_val)
            scores[lab] = score
            print(f"{this_epoch},{lab},{score}")
            lab = lab + 1

    np.save(f"./score_save/save_scores_epoch_{this_epoch}_p1", scores)
    logging.info("Avg p1 score for epoch %d: %f", this_epoch, scores.mean())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(message)s"
    )

    MODEL = torch.nn.DataParallel(cornet.cornet_s.CORnet_S())

    IMAGENET = load_imagenet("imagenet", num_items=50)

    for epoch in range(-1, 10):
        test_on_dataset(MODEL, IMAGENET, epoch)
