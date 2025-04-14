import typer
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


def generate_gaussian(mu: np.array, sigma: np.array):
    return np.random.normal(loc=mu, scale=sigma)


def generate_gaussian_triplet(
    mus: np.array, sigmas: np.array, probs: float = None, n_samples: int = 10000
):
    M, D = mus.shape
    assert M == 2, "Number of Gaussians must equal 2."
    if probs == None:
        probs = np.full(M, 1 / M)
    samples = np.zeros((n_samples, 3, D))
    labels = np.zeros(
        (
            n_samples,
            3,
        )
    )
    for i in range(n_samples):
        t1 = np.random.randint(0, M)
        t2 = np.random.randint(0, M)
        anchor = generate_gaussian(mus[t1], sigmas[t1])
        positive = generate_gaussian(mus[t1], sigmas[t1])
        negative = generate_gaussian(mus[t2], sigmas[t2])
        samples[i] = np.concatenate([anchor, positive, negative], 0)
        labels[i] = [t1, t1, t2]
    return samples, labels


class GaussianTripletsDataset(Dataset):
    def __init__(self, mus, sigmas, n_samples=10000):
        self.samples, self.labels = generate_gaussian_triplet(mus, sigmas, n_samples)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.samples[idx]).float()
        return sample


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def plot_histogram(samples, n_bins=50):
    samples, labels = samples
    samples, labels = samples.flatten(), labels.flatten()
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 6))

    plt.hist(samples, bins=n_bins, alpha=0.4, color="blue", density=True)
    for i, label in enumerate(unique_labels):
        labeled_samples = samples[labels == label]
        plt.scatter(
            labeled_samples,
            np.zeros_like(labeled_samples),
            label=f"Label {label}",
            s=100,
            marker="|",
            alpha=0.4,
        )
    plt.title("Histogram of All Gaussian Triplets")
    plt.xlabel("Value")
    plt.ylabel("p(x)")
    plt.grid(True)


def main(seed: int = 0, plot: bool = False):
    np.random.seed(seed=seed)
    torch.random.manual_seed(seed)

    mus = np.array([[1.0], [5.0]])
    sigmas = np.array([[[1.0]], [[1.0]]])
    samples = generate_gaussian_triplet(mus, sigmas)

    dataset = GaussianTripletsDataset(mus, sigmas, n_samples=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for data in dataloader:
        X1 = data[:, 0, :]
        X2 = data[:, 1, :]
        X3 = data[:, 2, :]
        # loss = triplet_loss(X1, X2, X3)

    if plot:
        plot_histogram(samples)
        plt.show()


if __name__ == "__main__":
    typer.run(main)
