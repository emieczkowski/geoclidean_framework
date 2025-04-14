import typer
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from model import MLP
from generate import GaussianTripletsDataset, numpy_collate, plot_histogram
from loss import *
from scipy.stats import pearsonr   

def train(model, trainloader, criterion, optimizer, epochs, device):
    model.train()
    losses = []
    with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
        for epoch in range(epochs):
            total_loss = 0
            for X in trainloader:
                X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
                X1, X2, X3 = X1.to(device), X2.to(device), X3.to(device)
                optimizer.zero_grad()
                phi1 = model(X1)
                phi2 = model(X2)
                phi3 = model(X3)
                loss = criterion(phi1, phi2, phi3)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(trainloader)
            pbar.set_postfix({"Average Epoch Loss": avg_loss})
            pbar.update()
            losses.append(avg_loss)
    return losses


def evaluate(model, trainset, testset, device):
    classifier = nn.Sequential(
        nn.Linear(1, 2)
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    batch_size = 256
    model.eval()
    
    for epoch in range(1):
        classifier.train()
        num_batches = len(trainset.samples) // batch_size
        for i in range(0, len(trainset.samples), batch_size):
            end_idx = min(i+batch_size, len(trainset.samples))
            data = torch.tensor(trainset.samples[i:end_idx,0,:], dtype=torch.float32).to(device)
            label = torch.tensor(trainset.labels[i:end_idx,0], dtype=torch.int64).to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                embedding = model(data)
            output = classifier(embedding)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        pred_results = []
        true_targets = []
        classifier.eval()
        num_batches = len(testset.samples) // batch_size
        for i in range(0, len(testset.samples), batch_size):
            end_idx = min(i+batch_size, len(testset.samples))
            data = torch.tensor(testset.samples[i:end_idx,0,:], dtype=torch.float32).to(device)
            label = torch.tensor(testset.labels[i:end_idx,0], dtype=torch.int64)
            embedding = model(data)
            output = classifier(embedding).detach().cpu()
            pred_results.extend(list(torch.squeeze(torch.argmax(output, 1)).numpy()))
            true_targets.extend(list(torch.squeeze(label).detach().cpu().numpy()))

        pred_results = np.array(pred_results)
        true_targets = np.array(true_targets)
        acc = np.mean(pred_results == true_targets)
        print(acc)

    return acc



def main(seed: int = 0, batch_size: int = 256, lr: float = 1e-5, epochs: int = 3):
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")

    # set random seeds
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set up data
    mus = np.array([[1.0, 1], [5.0, 5]])
    sigmas = np.array([[[1.0, 1]], [[1.0, 1]]])
    trainset = GaussianTripletsDataset(mus, sigmas, n_samples=10000)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = GaussianTripletsDataset(mus, sigmas, n_samples=10000)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    # print(testset.samples.shape)

    # init model
    model = MLP(input_size=2, output_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.TripletMarginLoss()
    criterion = triplet_loss_l2

    losses = train(model, trainloader, criterion, optimizer, epochs, device)
    embeddings = model(torch.from_numpy(testset.samples).float().to(device)).cpu().detach().numpy() # (10000,3,1)
    embeddings = np.squeeze(embeddings)
    with open(f'data.pickle', 'wb') as handle:
        pickle.dump([testset.samples, testset.labels, embeddings], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(losses[-1])
    accuracy = evaluate(model, trainset, testset, device)
    # print(f"Test accuracy: {accuracy:.2f}%")

    X = torch.from_numpy(testset.samples.reshape(-1, 2)).float().to(device)
    y = testset.labels.reshape(-1, 1)
    with torch.no_grad():
        Xpred = model(X).cpu().detach().numpy()
    plot_histogram((Xpred, y))
    plt.show()
    plt.close()

    # ---- Compare with probabilistic generative similarity ----+
    mu1 = mus[0,0]
    mu2 = mus[1,0]
    sig = sigmas[0,0,0]
    print(sig)

    embeddings = model(torch.from_numpy(testset.samples).float().to(device)).cpu().detach().numpy() # (10000,3,1)
    embedding_similarity = np.zeros((30000,1))
    for i, triplet in enumerate(embeddings):
        x1 = triplet[0,:]
        x2 = triplet[1,:]
        x3 = triplet[2,:]
        # embedding_similarity[3*i,0] = np.exp(np.sum(x1 * x2))
        # embedding_similarity[3*i+1,0] = np.exp(np.sum(x1 * x3))
        # embedding_similarity[3*i+2,0] = np.exp(np.sum(x2 * x3))
        embedding_similarity[3*i,0] = -np.sum((x1 - x2) ** 2)
        embedding_similarity[3*i+1,0] = -np.sum((x1 - x3) ** 2)
        embedding_similarity[3*i+2,0] = -np.sum((x2 - x3) ** 2)

    prob_similarity = np.zeros((30000,1))
    for i, triplet in enumerate(testset.samples):
        x1 = triplet[0,:]
        x2 = triplet[1,:]
        x3 = triplet[2,:]
        prob_similarity[3*i,0] = prob_similarity_pair(x1, x2, mu1, mu2, sig)
        prob_similarity[3*i+1,0] = prob_similarity_pair(x1, x3, mu1, mu2, sig)
        prob_similarity[3*i+2,0] = prob_similarity_pair(x2, x3, mu1, mu2, sig)

    all_similarities = np.concatenate([embedding_similarity, prob_similarity], 1)
    plt.scatter(all_similarities[:,0],all_similarities[:,1], s=8)
    plt.title('Probabilistic similarity (y) vs. Trained embedding similarity (x)')
    plt.show()
    plt.close()

    plt.scatter(-((-all_similarities[:,0]) ** 0.5) ,all_similarities[:,1], s=8)
    plt.title('Probabilistic similarity (y) vs. Trained embedding similarity (x)')
    plt.show()
    plt.close()

    print('Correlations')
    print(pearsonr(all_similarities[:,0],all_similarities[:,1])) 
    print(pearsonr(-((-all_similarities[:,0]) ** 0.5),all_similarities[:,1]))
    # ---------------------+

    # plot_histogram(dataset.samples)



if __name__ == "__main__":
    typer.run(main)
