# train_contrastive.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt 
import json
import argparse
from pytorch_lightning.callbacks import EarlyStopping 
import random 
import torch.utils.data 

# --- Function to set seed --- # 
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


# --- Dataset Class (Revised for Composite Images) ---
class AnalogyTripletDataset(Dataset):
    """
    Dataset loading composite analogy images (S top, C1 bottom-left, C2 bottom-right)
    and cropping them into (stimulus, correct_comp, foil_comp) triplets.
    """
    def __init__(self, image_dir: str, labels_file: str, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform
        print(f"Loaded {len(self.labels_df)} labels from {labels_file}")
        if 'filename' not in self.labels_df.columns or 'label' not in self.labels_df.columns:
             raise ValueError(f"Labels file {labels_file} must contain 'filename' and 'label' columns.")

    def __len__(self):
        return len(self.labels_df)

    def _crop_image(self, img: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Crops the composite image into S, C_bl, C_br."""
        width, height = img.size
        h_mid = height // 2
        w_mid = width // 2

        img_s = img.crop((0, 0, width, h_mid))
        img_c_bl = img.crop((0, h_mid, w_mid, height))
        img_c_br = img.crop((w_mid, h_mid, width, height))

        return img_s, img_c_bl, img_c_br

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]
        img_filename = row['filename']
        correct_label = row['label'] 
        img_path = os.path.join(self.image_dir, img_filename)
        
        try:
            # Load the single composite image
            composite_img = Image.open(img_path).convert('RGB')

            # Crop into components
            img_s, img_c_bl, img_c_br = self._crop_image(composite_img)

            # Assign correct and foil based on label
            if correct_label == 0:
                img_c_correct = img_c_bl
                img_c_foil = img_c_br
            elif correct_label == 1:
                img_c_correct = img_c_br
                img_c_foil = img_c_bl
            else:
                 print(f"Warning: Invalid label {correct_label} for image {img_filename}. Skipping.")
                 return None 

            # Apply transforms individually to the cropped images
            if self.transform:
                transformed_s = self.transform(img_s)
                transformed_c_correct = self.transform(img_c_correct)
                transformed_c_foil = self.transform(img_c_foil)
            else:
                to_tensor = transforms.ToTensor()
                transformed_s = to_tensor(img_s)
                transformed_c_correct = to_tensor(img_c_correct)
                transformed_c_foil = to_tensor(img_c_foil)

            return transformed_s, transformed_c_correct, transformed_c_foil

        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}. Skipping item {idx}.")
            return None # Return None on FileNotFoundError
        except Image.UnidentifiedImageError as e:
            print(f"Warning: Cannot identify image file {img_path} at index {idx}: {e}. Skipping.")
            return None # Return None on PIL error
        except Exception as e:
            print(f"Error processing image {img_path} at index {idx}: {e}. Skipping.")
            return None # Return None on other exceptions


# --- Standard CNN Encoder (Using ResNet18) ---
class ImageEncoder(nn.Module):
    """CNN encoder using ResNet18 backbone, outputting a feature vector."""
    def __init__(self, feature_dim=128, pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim
        print(f"Initializing ImageEncoder with pretrained={pretrained}")

        # Load ResNet18 based on the pretrained flag
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.flatten = nn.Flatten()

        # Output layer (maps 512 ResNet features to our desired feature_dim)
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = self.flatten(features) # Flatten the output of the backbone
        embedding = self.fc(features)
        return embedding

# --- Projection Head ---
class ProjectionHead(nn.Module):
    """MLP head for contrastive learning."""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.head(x)

# --- InfoNCE Loss Helper Function ---
def info_nce_loss(features_s, features_c_plus, features_c_minus, temperature=0.1):
    """
    Calculates InfoNCE loss for the S, C+, C- setup.
    Assumes features are L2 normalized.
    """
    sim_pos = F.cosine_similarity(features_s, features_c_plus, dim=-1)
    sim_neg = F.cosine_similarity(features_s, features_c_minus, dim=-1)

    exp_sim_pos = torch.exp(sim_pos / temperature)
    exp_sim_neg = torch.exp(sim_neg / temperature)
    denominator = exp_sim_pos + exp_sim_neg

    loss = -torch.log(exp_sim_pos / (denominator + 1e-8)) # Add epsilon for stability
    return loss.mean()


# --- Custom Collate Function ---
def collate_fn_skip_none(batch):
    """Collate function that filters out None values from a batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if batch is empty after filtering
    # Use default collate if batch has valid samples
    return torch.utils.data.default_collate(batch)


# --- Lightning Module for Contrastive Learning ---
class ContrastiveLearner(pl.LightningModule):
    def __init__(self, feature_dim: int = 128, projection_dim: int = 128,
                 learning_rate: float = 1e-4, temperature: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters("feature_dim", "projection_dim", "learning_rate", "temperature", "pretrained")
        self.encoder = ImageEncoder(feature_dim=feature_dim, pretrained=pretrained)
        self.projector = ProjectionHead(input_dim=feature_dim, output_dim=projection_dim)

    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projector(embedding)
        return projection # Use projection for loss calculation

    def training_step(self, batch, batch_idx):
        # Add check for None batch returned by collate_fn
        if batch is None:
            return None # Skip step if batch is empty
        
        stimulus_img, correct_img, foil_img = batch

        # Get projected features
        proj_s = self(stimulus_img)
        proj_c_plus = self(correct_img)
        proj_c_minus = self(foil_img)

        # L2 Normalize features (important for cosine similarity stability)
        proj_s_norm = F.normalize(proj_s, p=2, dim=-1)
        proj_c_plus_norm = F.normalize(proj_c_plus, p=2, dim=-1)
        proj_c_minus_norm = F.normalize(proj_c_minus, p=2, dim=-1)

        loss = info_nce_loss(proj_s_norm, proj_c_plus_norm, proj_c_minus_norm,
                             temperature=self.hparams.temperature)

        # Calculate training accuracy
        with torch.no_grad():
            sim_pos = F.cosine_similarity(proj_s.detach(), proj_c_plus.detach(), dim=-1)
            sim_neg = F.cosine_similarity(proj_s.detach(), proj_c_minus.detach(), dim=-1)
            acc = (sim_pos > sim_neg).float().mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True) # Log train accuracy
        return loss


    def test_step(self, batch, batch_idx):
        if batch is None:
            print(f"Warning: Skipping test step for batch {batch_idx} due to empty batch after filtering.")
            print(f"Warning: Skipping test step for batch {batch_idx} due to empty batch after filtering.")
            return None
             
        stimulus_img, correct_img, foil_img = batch

        # Get encoder features (embeddings BEFORE projection head)
        with torch.no_grad(): # Ensure no gradients computed during testing for embeddings
            embed_s = self.encoder(stimulus_img)
            embed_c_plus = self.encoder(correct_img)
            embed_c_minus = self.encoder(foil_img)

        # --- Optional: Calculate loss based on *projected* features (as before) ---
        # This requires passing through projector head as well
        # If only evaluating accuracy, this part can be skipped
        proj_s = self.projector(embed_s) # Pass embeddings through projector
        proj_c_plus = self.projector(embed_c_plus)
        proj_c_minus = self.projector(embed_c_minus)

        proj_s_norm = F.normalize(proj_s, p=2, dim=-1)
        proj_c_plus_norm = F.normalize(proj_c_plus, p=2, dim=-1)
        proj_c_minus_norm = F.normalize(proj_c_minus, p=2, dim=-1)

        loss = info_nce_loss(proj_s_norm, proj_c_plus_norm, proj_c_minus_norm,
                             temperature=self.hparams.temperature)
        # --- End Optional Loss Calculation ---

        # Calculate accuracy using ENCODER embeddings:
        # Similarity(embed(S), embed(C+)) > Similarity(embed(S), embed(C-))
        sim_pos = F.cosine_similarity(embed_s, embed_c_plus, dim=-1)
        sim_neg = F.cosine_similarity(embed_s, embed_c_minus, dim=-1)
        acc = (sim_pos > sim_neg).float().mean()

        # Log test metrics (prefix with test_)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True) # Loss still based on projections
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)  # Acc based on encoder embeddings
        return loss


    def configure_optimizers(self):
        # Consider weight decay
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-6)

# --- Plotting Function  ---
def plot_metrics(log_dir, final_test_acc=None, metrics=['train_loss', 'train_acc'], plot_filename=None):
    metrics_path = os.path.join(log_dir, 'version_0', 'metrics.csv') 
    if plot_filename is None:
        plot_filename = os.path.join(log_dir, 'contrastive_training_plots.png')

    if not os.path.exists(metrics_path):
        print(f"Warning: metrics.csv not found in {metrics_path}. Skipping plotting.")
        return
    try:
        metrics_df = pd.read_csv(metrics_path)
        if metrics_df.empty:
             print(f"Warning: metrics.csv found but is empty in {metrics_path}. Skipping plotting.")
             return

        # Clean up potential NaNs and ensure epoch is integer for plotting
        metrics_df = metrics_df.dropna(subset=['epoch'])
        metrics_df = metrics_df.drop_duplicates(subset=['epoch'], keep='last') # Keep last entry per epoch
        metrics_df['epoch'] = metrics_df['epoch'].astype(int)
        metrics_df = metrics_df.sort_values(by='epoch')

        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
        plt.figure(figsize=(14, 6)) # Adjusted figure size
        # Determine number of subplots based on available metrics
        num_plots = 0
        plot_indices = {}
        if 'train_loss' in metrics and 'train_loss' in metrics_df.columns: num_plots+=1; plot_indices['loss'] = num_plots
        if 'train_acc' in metrics and 'train_acc' in metrics_df.columns: num_plots+=1; plot_indices['acc'] = num_plots

        if num_plots == 0:
            print("No metrics to plot based on available data.")
            return

        # Loss Plot
        if 'loss' in plot_indices:
            ax1 = plt.subplot(1, num_plots, plot_indices['loss'])
            if 'train_loss' in metrics and 'train_loss' in metrics_df.columns:
                train_loss_df = metrics_df[['epoch', 'train_loss']].dropna() # Select columns and drop NaNs
                if not train_loss_df.empty:
                     ax1.plot(train_loss_df['epoch'], train_loss_df['train_loss'], label='Train Loss', marker='.', markersize=8, linestyle='-')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training Loss', fontsize=14)
            ax1.legend(fontsize=10)
            ax1.grid(True, linestyle=':')
            ax1.tick_params(axis='both', which='major', labelsize=10)

        # Accuracy Plot
        if 'acc' in plot_indices:
            ax2 = plt.subplot(1, num_plots, plot_indices['acc'])
            plotted_acc = False
            if 'train_acc' in metrics and 'train_acc' in metrics_df.columns:
                 train_acc_df = metrics_df[['epoch', 'train_acc']].dropna() # Select columns and drop NaNs
                 if not train_acc_df.empty:
                    ax2.plot(train_acc_df['epoch'], train_acc_df['train_acc'], label='Train Accuracy', marker='.', markersize=8, linestyle='-')
                    plotted_acc = True

            # Plot final test accuracy if available
            if final_test_acc is not None:
                ax2.axhline(final_test_acc, color='r', linestyle='--', linewidth=2, label=f'Final Test Acc: {final_test_acc*100:.2f}%')
                plotted_acc = True

            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Accuracy', fontsize=12)
            ax2.set_title('Training Accuracy vs Final Test Accuracy', fontsize=14)
            if plotted_acc:
                 ax2.legend(fontsize=10)
            ax2.grid(True, linestyle=':')
            ax2.tick_params(axis='both', which='major', labelsize=10)
            ax2.set_ylim(bottom=max(0, ax2.get_ylim()[0]), top=min(1.05, ax2.get_ylim()[1])) # Adjust y-axis limits slightly


        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"Saved training plots to {plot_filename}")
        plt.close()

    except pd.errors.EmptyDataError:
        print(f"Warning: metrics.csv found but is empty in {metrics_path}. Skipping plotting.")
    except Exception as e:
        print(f"Warning: Error processing metrics.csv or plotting from {metrics_path}: {e}. Skipping plotting.")
        # import traceback # Uncomment for detailed debugging
        # traceback.print_exc()


# --- Visualization Helper ---
def visualize_sample(dataset, index):
    """Loads and displays a single sample (composite, S, C+, C-) from the dataset."""
    print(f"Visualizing dataset sample at index: {index}")
    row = dataset.labels_df.iloc[index]
    img_filename = row['filename']
    correct_label = row['label']
    img_path = os.path.join(dataset.image_dir, img_filename)

    try:
        composite_img = Image.open(img_path).convert('RGB')
        img_s, img_c_bl, img_c_br = dataset._crop_image(composite_img)

        if correct_label == 0:
            img_c_correct = img_c_bl
            img_c_foil = img_c_br
            label_str = "0 (Bottom-Left Correct)"
        elif correct_label == 1:
            img_c_correct = img_c_br
            img_c_foil = img_c_bl
            label_str = "1 (Bottom-Right Correct)"
        else:
            print(f"  Skipping visualization due to invalid label: {correct_label}")
            return

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"Sample {index}: {img_filename} | Label: {label_str}", y=1.02)
        
        axes[0].imshow(composite_img)
        axes[0].set_title("Original Composite")
        axes[0].axis('off')

        axes[1].imshow(img_s)
        axes[1].set_title("Stimulus (S)")
        axes[1].axis('off')

        axes[2].imshow(img_c_correct)
        axes[2].set_title("Correct (C+)")
        axes[2].axis('off')

        axes[3].imshow(img_c_foil)
        axes[3].set_title("Foil (C-)")
        axes[3].axis('off')

        plt.tight_layout()
        plt.show() # Show the plot - this will pause execution

    except FileNotFoundError:
        print(f"  Error visualizing: Image file not found at {img_path}")
    except Exception as e:
        print(f"  Error visualizing sample {index}: {e}")


# --- Main Execution (Adjust Data Loading Paths and Trainer Calls) ---
def main():
    parser = argparse.ArgumentParser(description="Train a contrastive encoder for relational matching")
    # Data dir points to base containing train/test subdirs with IMAGES
    parser.add_argument("--data_dir", type=str, default="data_concepts_relmatch_loaded", help="Base data directory containing train/test subdirectories with composite images")
    parser.add_argument("--labels_dir", type=str, default="data_concepts_relmatch_loaded", help="Directory containing train_labels.csv and test_labels.csv")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--feature_dim", type=int, default=128, help="Feature dimension from encoder")
    parser.add_argument("--projection_dim", type=int, default=128, help="Output dimension of projection head")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for InfoNCE loss")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.995, help="Training accuracy threshold for early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs with train_acc above threshold to wait before stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Disable using ImageNet pre-trained weights for ResNet")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--visualize_samples", action="store_true", help="Show first few samples using matplotlib before training.")

    args = parser.parse_args()

    # --- Set Seed --- # 
    set_seed(args.seed)
    print(f"Running with seed: {args.seed}")

    base_log_dir = "./contrastive_logs"
    log_dir = os.path.join(base_log_dir, f"seed_{args.seed}") # Seed-specific log directory
    # Define suffix based on pretrained status
    status_suffix = "_pre" if args.pretrained else "_nopre"
    results_file = os.path.join(log_dir, f"results_contrastive_seed_{args.seed}{status_suffix}.json") # Seed-specific and status-specific results file
    plot_file = os.path.join(log_dir, f"contrastive_training_plots_seed_{args.seed}{status_suffix}.png") # Seed-specific and status-specific plot file

    # Ensure the seed-specific log directory exists
    os.makedirs(log_dir, exist_ok=True)

    if args.visualize_samples:
        print("--- Visualizing Training Samples ---")
        # Assume train subdir and labels file naming convention
        vis_train_img_dir = os.path.join(args.data_dir, "train") 
        vis_train_labels = os.path.join(args.data_dir, "train_labels.csv") 
        if os.path.exists(vis_train_img_dir) and os.path.exists(vis_train_labels):
            vis_dataset = AnalogyTripletDataset(
                image_dir=vis_train_img_dir,
                labels_file=vis_train_labels,
                transform=None 
            )
            print(f"Attempting to visualize up to 3 samples from {vis_train_img_dir}...")
            for i in range(min(3, len(vis_dataset))):
                visualize_sample(vis_dataset, i) 
            print("------------------------------------")
        else:
            print(f"Warning: Cannot visualize training samples. Check paths:")
            print(f"  Image Dir: {vis_train_img_dir} (Exists: {os.path.exists(vis_train_img_dir)})")
            print(f"  Labels File: {vis_train_labels} (Exists: {os.path.exists(vis_train_labels)})")
        print("Visualization check complete. Continuing with training setup...")
    # --- End Visualization --- 

    # --- Data Setup ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define paths based on args
    train_image_dir = os.path.join(args.data_dir, 'train')
    test_image_dir = os.path.join(args.data_dir, 'test') 
    train_labels_csv = os.path.join(args.labels_dir, 'train_labels.csv')
    test_labels_csv = os.path.join(args.labels_dir, 'test_labels.csv') 

    # Check if files/dirs exist
    if not os.path.isdir(train_image_dir): print(f"ERROR: Train image directory not found: {train_image_dir}"); return
    if not os.path.isdir(test_image_dir): print(f"ERROR: Test image directory not found: {test_image_dir}"); return
    if not os.path.exists(train_labels_csv): print(f"ERROR: Train labels CSV not found: {train_labels_csv}"); return
    if not os.path.exists(test_labels_csv): print(f"ERROR: Test labels CSV not found: {test_labels_csv}"); return

    train_dataset = AnalogyTripletDataset(train_image_dir, train_labels_csv, transform=transform)
    test_dataset = AnalogyTripletDataset(test_image_dir, test_labels_csv, transform=transform) 

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn_skip_none 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True if torch.cuda.is_available() else False,
        collate_fn=collate_fn_skip_none 
    )

    # --- Model & Trainer ---
    model = ContrastiveLearner(
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        pretrained=args.pretrained
    )
    # --- Logger Setup --- # 
    # Logs will go into ./contrastive_logs/seed_X/contrastive_run_SUFFIX/version_0/
    logger = CSVLogger(log_dir, name=f"contrastive_run{status_suffix}") 

    # --- Configure Early Stopping --- # 
    early_stop_callback = EarlyStopping(
        monitor='train_acc',       
        stopping_threshold=args.early_stopping_threshold, 
        patience=args.early_stopping_patience,    
        mode='max',             
        verbose=True            
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=10, 
        accelerator="auto", 
        callbacks=[early_stop_callback] 
    )

    # --- Training ---
    print("Starting Contrastive Learning Training...")
    trainer.fit(model, train_dataloaders=train_loader) 
    print("Training finished.")

    # --- Testing ---
    print("Starting Testing...")
    test_results = trainer.test(model, dataloaders=test_loader) 
    print("Testing finished.")
    print("Test Results:", test_results)


    # --- Save Results ---
    final_test_acc = test_results[0].get('test_acc', None) if test_results else None
    # Also capture epoch when stopped if early stopping occurred
    # Ensure early_stopping_callback exists before accessing attributes
    stopped_epoch = -1 # Default value
    if trainer.early_stopping_callback:
         stopped_epoch = trainer.early_stopping_callback.stopped_epoch if hasattr(trainer.early_stopping_callback, 'stopped_epoch') and trainer.early_stopping_callback.stopped_epoch > 0 else trainer.current_epoch

    results = {
        "seed": args.seed, 
        "final_test_acc": final_test_acc,
        "stopped_epoch": stopped_epoch
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved test results to {results_file}")

    # --- Plot Metrics ---
    plot_metrics(logger.log_dir, final_test_acc=final_test_acc, metrics=['train_loss', 'train_acc'], plot_filename=plot_file) 

if __name__ == '__main__':
    main()
