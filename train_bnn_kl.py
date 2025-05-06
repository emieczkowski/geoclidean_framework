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
import torch.utils.data
import random
import glob 

'''
trains a BNN encoder with KL divergence triplet loss on the relational match task.
'''

# --- Function to set seed --- # Added
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Dataset Class (Loads Triplets) ---
class BNNRelationalMatchDataset(Dataset):
    """Dataset loading image triplets (anchor, positive, negative)."""
    def __init__(self, data_dir: str, labels_file: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels_df = pd.read_csv(labels_file)
        
        # Group images by label for triplet sampling
        self.pos_images = self.labels_df[self.labels_df['label'] == 1]['filename'].tolist()
        self.neg_images = self.labels_df[self.labels_df['label'] == 0]['filename'].tolist()
        
    def __len__(self):
        return len(self.labels_df)
    
    def get_triplet(self, idx: int):
        """Loads a triplet (anchor, positive, negative). Returns None if loading fails irrecoverably."""
        try:
            anchor_name = self.labels_df.iloc[idx]['filename']
            anchor_label = self.labels_df.iloc[idx]['label']
        except IndexError:
            print(f"Warning: Index {idx} out of bounds for labels_df.")
            return None # Index out of bounds

        # --- Load Anchor --- 
        try:
            anchor_img = Image.open(os.path.join(self.data_dir, anchor_name)).convert('RGB')
        except (FileNotFoundError, Image.UnidentifiedImageError) as e:
            print(f"Error loading ANCHOR image {anchor_name} for index {idx}: {e}. Skipping triplet.")
            return None # Cannot proceed without anchor
        except Exception as e:
            print(f"Unexpected error loading ANCHOR image {anchor_name} for index {idx}: {e}. Skipping triplet.")
            return None

        # --- Load Positive --- 
        pos_pool = self.pos_images if anchor_label == 1 else self.neg_images
        valid_pos_pool = [x for x in pos_pool if x != anchor_name]
        if not valid_pos_pool: valid_pos_pool = pos_pool # Fallback if anchor is the only one
        if not valid_pos_pool: 
            print(f"Warning: No valid positive examples found for anchor {anchor_name}. Skipping triplet.")
            return None

        pos_img = None
        for attempt in range(2): # Try twice to find a loadable positive image
            pos_name = np.random.choice(valid_pos_pool)
            try:
                pos_img = Image.open(os.path.join(self.data_dir, pos_name)).convert('RGB')
                break # Success
            except (FileNotFoundError, Image.UnidentifiedImageError) as e:
                print(f"Error loading POSITIVE image {pos_name} for anchor {anchor_name} (Attempt {attempt+1}): {e}")
                # Remove problematic image from pool for this triplet and retry if possible
                valid_pos_pool = [x for x in valid_pos_pool if x != pos_name]
                if not valid_pos_pool: break # No more options
            except Exception as e:
                 print(f"Unexpected error loading POSITIVE image {pos_name} for anchor {anchor_name} (Attempt {attempt+1}): {e}")
                 break # Don't retry on unexpected errors
        
        if pos_img is None:
            print(f"Failed to load a valid POSITIVE image for anchor {anchor_name}. Skipping triplet.")
            return None

        # --- Load Negative --- 
        neg_pool = self.neg_images if anchor_label == 1 else self.pos_images
        if not neg_pool:
            print(f"Warning: No negative examples found for anchor {anchor_name}. Skipping triplet.")
            return None
        
        neg_img = None
        original_neg_pool = list(neg_pool) # Copy to allow modification
        for attempt in range(2): # Try twice
            if not original_neg_pool:
                break # No more options
            neg_name = np.random.choice(original_neg_pool)
            try:
                neg_img = Image.open(os.path.join(self.data_dir, neg_name)).convert('RGB')
                break # Success
            except (FileNotFoundError, Image.UnidentifiedImageError) as e:
                print(f"Error loading NEGATIVE image {neg_name} for anchor {anchor_name} (Attempt {attempt+1}): {e}")
                original_neg_pool.remove(neg_name) # Remove problematic image from pool for this triplet
            except Exception as e:
                print(f"Unexpected error loading NEGATIVE image {neg_name} for anchor {anchor_name} (Attempt {attempt+1}): {e}")
                break # Don't retry on unexpected errors

        if neg_img is None:
            print(f"Failed to load a valid NEGATIVE image for anchor {anchor_name}. Skipping triplet.")
            return None

        # --- Apply Transform --- 
        if self.transform:
            try:
                anchor_img = self.transform(anchor_img)
                pos_img = self.transform(pos_img)
                neg_img = self.transform(neg_img)
            except Exception as e:
                print(f"Error applying transform to triplet for anchor {anchor_name}: {e}. Skipping triplet.")
                return None

        return anchor_img, pos_img, neg_img
    
    def __getitem__(self, idx):
        # get_triplet now handles errors and returns None if it fails
        return self.get_triplet(idx)

# --- Custom Collate Function ---
def collate_fn_skip_none(batch):
    """Collate function that filters out None values.
    Args:
        batch: List of samples from the dataset, potentially containing None.
    Returns:
        A collated batch if valid samples exist, otherwise None (or handle as needed).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None 
    return torch.utils.data.default_collate(batch)

# --- BNN Encoder (with Dropout) ---
class BNNEncoder(nn.Module):
    """CNN encoder outputting parameters (mu, logvar) for a Gaussian distribution."""
    def __init__(self, feature_dim=128, dropout_prob=0.2): # Added dropout_prob
        super().__init__()
        self.feature_dim = feature_dim
        self.dropout_prob = dropout_prob
        
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
        
        # Dropout layer before final FC layers
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        # Output heads for mu and logvar
        self.fc_mu = nn.Linear(128, feature_dim)
        self.fc_logvar = nn.Linear(128, feature_dim)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features) # Apply dropout
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

# --- Plotting Function (Modified to only plot train loss if available) ---
def plot_metrics(log_dir, metrics=['train_loss'], seed=None):
    # Find the specific version directory within the log_dir
    version_dirs = glob.glob(os.path.join(log_dir, 'version_*'))
    if not version_dirs: 
        print(f"Warning: No version directory found in {log_dir}. Skipping plotting.")
        return
    # Assume latest version if multiple exist
    metrics_path = os.path.join(sorted(version_dirs)[-1], 'metrics.csv')
    
    if not os.path.exists(metrics_path):
        print(f"Warning: metrics.csv not found in {metrics_path}. Skipping plotting.")
        return
    try:
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
            print(f"Warning: metrics.csv found but is empty in {metrics_path}. Skipping plotting.")
            return
    except Exception as e:
        print(f"Warning: Error reading metrics.csv from {metrics_path}: {e}. Skipping plotting.")
        return

    metrics_df = metrics_df.dropna(subset=['epoch'])
    metrics_df['epoch'] = metrics_df['epoch'].astype(int)
    metrics_df = metrics_df.drop_duplicates(subset=['epoch'], keep='last') 
    metrics_df = metrics_df.sort_values(by='epoch')
    
    plt.figure(figsize=(8, 5))
    plotted = False
    if 'train_loss' in metrics and 'train_loss' in metrics_df.columns:
        train_loss_df = metrics_df.dropna(subset=['train_loss'])
        if not train_loss_df.empty:
             plt.plot(train_loss_df['epoch'], train_loss_df['train_loss'], label='Train Loss')
             plotted = True
             
    if not plotted:
        print("No metrics to plot.")
        plt.close()
        return
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot in the base log directory (parent of version_X)
    plot_suffix = f"_seed_{seed}" if seed is not None else ""
    plot_filename = os.path.join(log_dir, f'bnn_kl_training_plots{plot_suffix}.png') 
    try:
        plt.savefig(plot_filename)
        print(f"Saved training plots to {plot_filename}")
    except Exception as e:
         print(f"Error saving plot {plot_filename}: {e}")
    finally:
        plt.close()

# --- Lightning Trainer Module (Add test_step) ---
class BNNKLTripetTrainer(pl.LightningModule):
    def __init__(self, feature_dim: int = 128, learning_rate: float = 1e-3, 
                 margin: float = 1.0, dropout_prob: float = 0.2):
        super().__init__()
        self.save_hyperparameters("feature_dim", "learning_rate", "margin", "dropout_prob") 
        self.bnn_encoder = BNNEncoder(feature_dim=feature_dim, dropout_prob=self.hparams.dropout_prob)
        
    def forward(self, x):
        return self.bnn_encoder(x)
    
    def training_step(self, batch, batch_idx):
        if batch is None:
            return None 
            
        anchor_img, pos_img, neg_img = batch
        
        mu_a, lv_a = self(anchor_img)
        mu_p, lv_p = self(pos_img)
        mu_n, lv_n = self(neg_img)
        
        kl_pos = kl_divergence_gaussian(mu_a, lv_a, mu_p, lv_p)
        kl_neg = kl_divergence_gaussian(mu_a, lv_a, mu_n, lv_n)
        
        target = torch.ones_like(kl_neg) 
        loss = F.margin_ranking_loss(kl_neg, kl_pos, target, margin=self.hparams.margin)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        if batch is None:
            return None 
            
        anchor_img, pos_img, neg_img = batch
        
        # No gradient needed for testing
        with torch.no_grad():
            mu_a, lv_a = self(anchor_img)
            mu_p, lv_p = self(pos_img)
            mu_n, lv_n = self(neg_img)
            
            kl_pos = kl_divergence_gaussian(mu_a, lv_a, mu_p, lv_p)
            kl_neg = kl_divergence_gaussian(mu_a, lv_a, mu_n, lv_n)
            
            target = torch.ones_like(kl_neg) 
            loss = F.margin_ranking_loss(kl_neg, kl_pos, target, margin=self.hparams.margin)
            
            # Accuracy: KL distance to positive should be less than KL distance to negative
            acc = (kl_pos < kl_neg).float().mean()
            
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_acc': acc} 
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Train a BNN encoder with KL divergence triplet loss")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory (e.g., data_concepts_relmatch_loaded)")
    parser.add_argument("--test_image_subdir", type=str, default="test", help="Subdirectory containing test images.")
    parser.add_argument("--test_labels_filename", type=str, default="test_labels.csv", help="Filename for test labels CSV.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--feature_dim", type=int, default=128, help="Feature dimension for mu and logvar")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for triplet loss")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility") 
    parser.add_argument("--dropout_prob", type=float, default=0.2, help="Dropout probability in BNN encoder.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator (auto, cpu, gpu, mps).")
    parser.add_argument("--devices", type=str, default=None, help="Devices to use (e.g., '1', '0,1', None for auto).")
    args = parser.parse_args()
    
    set_seed(args.seed)
    print(f"Running with seed: {args.seed}")

    base_log_dir = "./bnn_kl_logs" 
    log_dir = os.path.join(base_log_dir, f"seed_{args.seed}") 
    results_file = os.path.join(log_dir, f"results_bnn_kl_seed_{args.seed}.json") 
    os.makedirs(log_dir, exist_ok=True)

    # --- Data Setup (Add test_dataset) --- 
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BNNRelationalMatchDataset(
        os.path.join(args.data_dir, 'train'),
        os.path.join(args.data_dir, 'train_labels.csv'),
        transform=train_transform
    )
    test_dataset = BNNRelationalMatchDataset(
        os.path.join(args.data_dir, args.test_image_subdir),
        os.path.join(args.data_dir, args.test_labels_filename),
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn_skip_none,
        pin_memory=True, persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_skip_none,
        pin_memory=True, persistent_workers=args.num_workers > 0
    )
    
    # --- Training Setup --- 
    print("--- Training BNN Encoder with KL Divergence Loss ---")
    run_name = "bnn_kl_run" 
    logger = CSVLogger(save_dir=log_dir, name=run_name) 
    
    model = BNNKLTripetTrainer(
        feature_dim=args.feature_dim,
        learning_rate=args.learning_rate,
        margin=args.margin,
        dropout_prob=args.dropout_prob
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=10,
        accelerator=args.accelerator,
        devices=args.devices if args.devices else 'auto',
        enable_checkpointing=False
    )
    
    # --- Train --- 
    trainer.fit(model, train_loader) 
    print("Training finished.")

    # --- Test --- 
    print("Starting Testing...")
    test_results = trainer.test(model=model, dataloaders=test_loader)
    print("Testing finished.")
    print("Test Results:", test_results)

    # --- Plot and Save Results (Adjusted for test accuracy) --- 
    # Extract test accuracy from results
    final_test_acc = test_results[0].get('test_acc', None) if test_results else None
    stopped_epoch = trainer.current_epoch
    
    # Plotting remains the same (plots train loss)
    plot_metrics(log_dir, seed=args.seed) 

    # Save final_test_acc, keep val metrics as None
    results = {
        "seed": args.seed, 
        'final_test_acc': final_test_acc,
        'best_val_accuracy': None,
        'final_val_loss': None,
        'stopped_epoch': stopped_epoch,
        'margin': args.margin,
        'feature_dim': args.feature_dim,
        'dropout_prob': args.dropout_prob
    }
    with open(results_file, 'w') as f: 
        json.dump(results, f, indent=4)
    print(f"Saved results to {results_file}")

if __name__ == "__main__":
    main() 
