import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import json
import argparse
from sklearn.model_selection import train_test_split

# --- Dataset Class (Loads Triplets) ---
class BNNOOODataset(Dataset):
    """Dataset loading image triplets (anchor, positive, negative)."""
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_dir)
    
    def get_images(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img_folder = self.data_dir[idx]

        img_paths = [
            os.path.join(img_folder, "inlier_0.png"),
            os.path.join(img_folder, "inlier_1.png"),
            os.path.join(img_folder, "inlier_2.png"),
            os.path.join(img_folder, "outlier.png")
        ]

        inlier0 = Image.open(img_paths[0]).convert('RGB')
        inlier1 = Image.open(img_paths[1]).convert('RGB')
        inlier2 = Image.open(img_paths[2]).convert('RGB')
        outlier = Image.open(img_paths[3]).convert('RGB')

        if self.transform:
            inlier0 = self.transform(inlier0)
            inlier1 = self.transform(inlier1)
            inlier2 = self.transform(inlier2)
            outlier = self.transform(outlier)

        return inlier0, inlier1, inlier2, outlier
    
    def __getitem__(self, idx):
        return self.get_images(idx)

# --- BNN Encoder ---
class BNNEncoder(nn.Module):
    """CNN encoder outputting parameters (mu, logvar) for a Gaussian distribution."""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # CNN backbone (similar to SimpleEncoder)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Output heads for mu and logvar
        self.fc_mu = nn.Linear(128, feature_dim)
        self.fc_logvar = nn.Linear(128, feature_dim)
        
    def forward(self, x):
        features = self.backbone(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

# --- KL Divergence Function ---
def kl_divergence_gaussian(mu1, logvar1, mu2, logvar2, epsilon=1e-8):
    """Calculates KL divergence KL(N1 || N2) for diagonal Gaussians."""
    # Ensure variance is positive
    var1 = torch.exp(logvar1) + epsilon
    var2 = torch.exp(logvar2) + epsilon
    log_var_ratio = logvar2 - logvar1
    
    kl_components = 0.5 * (
        var1 / var2 + 
        (mu1 - mu2)**2 / var2 - 
        1.0 + 
        log_var_ratio
    )
    
    # Sum over feature dimension
    return torch.sum(kl_components, dim=1)

# --- Plotting Function (same as before) ---
def plot_metrics(log_dir, metrics=['train_loss', 'val_loss', 'val_acc']):
    metrics_path = os.path.join(log_dir, 'version_0', 'metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"Warning: metrics.csv not found in {metrics_path}. Skipping plotting.")
        return
    try:
        metrics_df = pd.read_csv(metrics_path)
    except pd.errors.EmptyDataError:
        print(f"Warning: metrics.csv found but is empty in {metrics_path}. Skipping plotting.")
        return
    except Exception as e:
        print(f"Warning: Error reading metrics.csv from {metrics_path}: {e}. Skipping plotting.")
        return

    metrics_df = metrics_df.dropna(subset=['epoch'])
    metrics_df['epoch'] = metrics_df['epoch'].astype(int)
    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 2, 1)
    if 'train_loss' in metrics and 'train_loss' in metrics_df.columns:
        train_loss_df = metrics_df.dropna(subset=['train_loss'])
        if not train_loss_df.empty:
             ax1.plot(train_loss_df['epoch'], train_loss_df['train_loss'], label='Train Loss')
    if 'val_loss' in metrics and 'val_loss' in metrics_df.columns:
        val_loss_df = metrics_df.dropna(subset=['val_loss'])
        if not val_loss_df.empty:
            ax1.plot(val_loss_df['epoch'], val_loss_df['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax2 = plt.subplot(1, 2, 2)
    if 'val_acc' in metrics and 'val_acc' in metrics_df.columns:
        val_acc_df = metrics_df.dropna(subset=['val_acc'])
        if not val_acc_df.empty:
            ax2.plot(val_acc_df['epoch'], val_acc_df['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    plt.tight_layout()
    plot_filename = os.path.join(log_dir, 'bnn_kl_training_plots.png') # Changed plot name
    try:
        plt.savefig(plot_filename)
        print(f"Saved training plots to {plot_filename}")
    except Exception as e:
         print(f"Error saving plot {plot_filename}: {e}")
    finally:
        plt.close()

# --- Lightning Trainer Module ---
class BNNKLTripetTrainer(pl.LightningModule):
    def __init__(self, feature_dim: int = 128, learning_rate: float = 1e-3, margin: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.bnn_encoder = BNNEncoder(feature_dim=feature_dim)
        
    def forward(self, x):
        # Returns distribution parameters
        return self.bnn_encoder(x)
    
    def training_step(self, batch, batch_idx):
        inlier0, inlier1, inlier2, outlier = batch
        
        mu_i0, lv_i0 = self(inlier0)
        mu_i1, lv_i1 = self(inlier1)
        mu_i2, lv_i2 = self(inlier2)
        mu_o, lv_o = self(outlier)
        
        kl_pos = (kl_divergence_gaussian(mu_i0, lv_i0, mu_i1, lv_i1)+
                         kl_divergence_gaussian(mu_i0, lv_i0, mu_i2, lv_i2)+
                         kl_divergence_gaussian(mu_i1, lv_i1, mu_i2, lv_i2))/3
        kl_neg = (kl_divergence_gaussian(mu_i0, lv_i0, mu_o, lv_o)+
                         kl_divergence_gaussian(mu_i1, lv_i1, mu_o, lv_o)+
                         kl_divergence_gaussian(mu_i2, lv_i2, mu_o, lv_o))/3
        
        # Target is 1 (we want kl_pos + margin < kl_neg)
        target = torch.ones_like(kl_neg) 
        loss = F.margin_ranking_loss(
            kl_neg, # input 1 (should be larger)
            kl_pos, # input 2 (should be smaller)
            target, 
            margin=self.hparams.margin
        )
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inlier0, inlier1, inlier2, outlier = batch
        
        mu_i0, lv_i0 = self(inlier0)
        mu_i1, lv_i1 = self(inlier1)
        mu_i2, lv_i2 = self(inlier2)
        mu_o, lv_o = self(outlier)
        
        kl_pos = (kl_divergence_gaussian(mu_i0, lv_i0, mu_i1, lv_i1)+
                         kl_divergence_gaussian(mu_i0, lv_i0, mu_i2, lv_i2)+
                         kl_divergence_gaussian(mu_i1, lv_i1, mu_i2, lv_i2))/3
        kl_neg = (kl_divergence_gaussian(mu_i0, lv_i0, mu_o, lv_o)+
                         kl_divergence_gaussian(mu_i1, lv_i1, mu_o, lv_o)+
                         kl_divergence_gaussian(mu_i2, lv_i2, mu_o, lv_o))/3
        
        target = torch.ones_like(kl_neg)
        loss = F.margin_ranking_loss(kl_neg, kl_pos, target, margin=self.hparams.margin)
        
        # Accuracy: KL distance to positive should be less than KL distance to negative
        acc = (kl_pos < kl_neg).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Train a BNN encoder with KL divergence triplet loss")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--feature_dim", type=int, default=128, help="Feature dimension for mu and logvar")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet loss")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    args = parser.parse_args()
    
    log_dir = "./bnn_kl_logs" # Log directory
    results_file = "results_bnn_kl.json"

    # --- Data Setup ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    X = []  
    y = []  
    trial_metadata = []
    root_dir = "data"
    
    for rule_type in os.listdir(root_dir):
        rule_path = os.path.join(root_dir, rule_type)
        if not os.path.isdir(rule_path): continue
    
        for rule_folder in os.listdir(rule_path):
            img_folder = os.path.join(rule_path, rule_folder)
            if not os.path.isdir(img_folder): continue
    
            X.append(img_folder)
    
    indices = np.arange(len(X))
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]

    train_dataset = BNNOOODataset(
        X_train,
        transform=transform
    )
    val_dataset = BNNOOODataset(
        X_test,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    
    # --- Training Setup ---
    print("--- Training BNN Encoder with KL Divergence Loss ---")
    logger = CSVLogger(save_dir=log_dir, name="") # Use CSVLogger
    model = BNNKLTripetTrainer(
        feature_dim=args.feature_dim,
        learning_rate=args.learning_rate,
        margin=args.margin
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=10,
        accelerator='auto',
        devices='auto'
    )
    
    # --- Train --- 
    trainer.fit(model, train_loader, val_loader)

    # --- Plot and Save Results ---
    plot_metrics(log_dir)

    best_val_acc = trainer.callback_metrics.get('val_acc')
    final_val_loss = trainer.callback_metrics.get('val_loss')

    results = {
        'best_val_accuracy': float(best_val_acc.cpu().numpy()) if best_val_acc is not None else None,
        'final_val_loss': float(final_val_loss.cpu().numpy()) if final_val_loss is not None else None,
        # Add other relevant hyperparameters if needed
        'margin': args.margin,
        'feature_dim': args.feature_dim
    }
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_file}")

if __name__ == "__main__":
    main() 
