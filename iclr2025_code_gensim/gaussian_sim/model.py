import typer

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers: int = 1, n_hidden: int = 32):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, n_hidden))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


def main(seed: int = 0):
    torch.random.manual_seed(seed)
    model = MLP(10, 10)
    x = torch.randn(1, 10)
    output = model(x)


if __name__ == "__main__":
    typer.run(main)
