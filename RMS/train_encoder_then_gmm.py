import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import argparse

'''
trains GMM, MLP, and Logistic Regression classifiers on VGG16 features extracted from the relational match images.
tests one by one, and saves results to a json file.
'''


def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
        # Verify required columns exist
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        # Returns S, C_correct, C_foil, label
        row = self.labels_df.iloc[idx]
        img_filename = row['filename']
        correct_label = row['label'] # 0 for bottom-left, 1 for bottom-right

        img_path = os.path.join(self.image_dir, img_filename)

        try:
            composite_img = Image.open(img_path).convert('RGB')
            img_s, img_c_bl, img_c_br = self._crop_image(composite_img)

            if correct_label == 0:
                img_c_correct = img_c_bl
                img_c_foil = img_c_br
            elif correct_label == 1:
                img_c_correct = img_c_br
                img_c_foil = img_c_bl
            else:
                 print(f"Warning: Invalid label {correct_label} for image {img_filename}. Returning None.")
                 return None # Need collate_fn to handle this if used

            # Apply transforms individually
            if self.transform:
                transformed_s = self.transform(img_s)
                transformed_c_correct = self.transform(img_c_correct)
                transformed_c_foil = self.transform(img_c_foil)
            else:
                to_tensor = transforms.ToTensor()
                transformed_s = to_tensor(img_s)
                transformed_c_correct = to_tensor(img_c_correct)
                transformed_c_foil = to_tensor(img_c_foil)

            # Return the original label as well
            return transformed_s, transformed_c_correct, transformed_c_foil, correct_label

        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}. Returning None.")
            return None
        except Exception as e:
            print(f"Error processing image {img_path} at index {idx}: {e}. Returning None.")
            return None

def extract_vgg16_features(dataloader, device):
    """
    Extracts features from the final conv block of a pretrained VGG16 model.

    Args:
        dataloader: DataLoader yielding batches of (S_img, C_plus_img, C_minus_img, label).
        device: The torch device ('cuda' or 'cpu').

    Returns:
        Tuple: (S_features, C_plus_features, C_minus_features, labels)
               Each feature array is shape (N, 25088), labels is shape (N,).
    """
    print("Loading pretrained VGG-16...")
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Remove classifier layers and adaptive pooling
    # The output of vgg16.features is the output of the last Conv layer (before pooling/flattening)
    feature_extractor = vgg16.features
    feature_extractor.eval() # Set to evaluation mode
    feature_extractor.to(device)

    s_features_list = []
    c_plus_features_list = []
    c_minus_features_list = []
    labels_list = []

    print("Extracting VGG-16 features...")
    with torch.no_grad():
        for batch in dataloader:
            # Handle potential Nones returned by dataset __getitem__
            if batch is None: continue
            s_img, c_plus_img, c_minus_img, labels = batch

            # Move images to device
            s_img = s_img.to(device)
            c_plus_img = c_plus_img.to(device)
            c_minus_img = c_minus_img.to(device)

            # Extract features
            s_feat = feature_extractor(s_img)       # Shape: [batch, 512, 7, 7]
            c_plus_feat = feature_extractor(c_plus_img) # Shape: [batch, 512, 7, 7]
            c_minus_feat = feature_extractor(c_minus_img)# Shape: [batch, 512, 7, 7]

            # Flatten features
            s_feat_flat = torch.flatten(s_feat, start_dim=1) # Shape: [batch, 512*7*7=25088]
            c_plus_feat_flat = torch.flatten(c_plus_feat, start_dim=1)
            c_minus_feat_flat = torch.flatten(c_minus_feat, start_dim=1)

            # Append to lists (move to CPU, convert to numpy)
            s_features_list.append(s_feat_flat.cpu().numpy())
            c_plus_features_list.append(c_plus_feat_flat.cpu().numpy())
            c_minus_features_list.append(c_minus_feat_flat.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    # Concatenate lists into final numpy arrays
    s_features_all = np.concatenate(s_features_list, axis=0)
    c_plus_features_all = np.concatenate(c_plus_features_list, axis=0)
    c_minus_features_all = np.concatenate(c_minus_features_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)

    print(f"Finished extraction. Feature shape: {s_features_all.shape}, Labels shape: {labels_all.shape}")
    return s_features_all, c_plus_features_all, c_minus_features_all, labels_all

def main():
    parser = argparse.ArgumentParser(description="Extract VGG features and train GMM/MLP/Logistic classifiers.")
    parser.add_argument("--data_dir", type=str, default="data_concepts_relmatch_loaded", help="Base data directory containing train/test subdirectories with composite images")
    parser.add_argument("--labels_dir", type=str, default="data_concepts_relmatch_loaded", help="Directory containing train_labels.csv and test_labels.csv")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--results_file", type=str, default=None, help="File to save results (defaults to results_vgg_classifiers_seed_SEED.json)")
    args = parser.parse_args()

    # Set default results file name if not provided
    if args.results_file is None:
        args.results_file = f"results_vgg_classifiers_seed_{args.seed}.json"

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms (standard ImageNet normalization)
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

    if not os.path.isdir(train_image_dir): print(f"ERROR: Train image directory not found: {train_image_dir}"); return
    if not os.path.isdir(test_image_dir): print(f"ERROR: Test image directory not found: {test_image_dir}"); return
    if not os.path.exists(train_labels_csv): print(f"ERROR: Train labels CSV not found: {train_labels_csv}"); return
    if not os.path.exists(test_labels_csv): print(f"ERROR: Test labels CSV not found: {test_labels_csv}"); return

    train_dataset = AnalogyTripletDataset(train_image_dir, train_labels_csv, transform=transform)
    test_dataset = AnalogyTripletDataset(test_image_dir, test_labels_csv, transform=transform)

    # Create dataloaders for feature extraction
    # Handle potential None items returned by dataset if files are missing/invalid
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
             return None # Return None if batch becomes empty after filtering
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader_extract = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader_extract = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # --- Define Feature File Paths --- # New
    train_features_file = 'vgg16_features_train.npz'
    test_features_file = 'vgg16_features_test.npz'

    # --- Load or Extract VGG16 Features --- # Modified
    if os.path.exists(train_features_file) and os.path.exists(test_features_file):
        print(f"Loading pre-extracted features from {train_features_file} and {test_features_file}...")
        train_data = np.load(train_features_file)
        S_train_feat = train_data['s_features']
        C_plus_train_feat = train_data['c_plus_features']
        C_minus_train_feat = train_data['c_minus_features']
        y_train = train_data['labels']

        test_data = np.load(test_features_file)
        S_test_feat = test_data['s_features']
        C_plus_test_feat = test_data['c_plus_features']
        C_minus_test_feat = test_data['c_minus_features']
        y_test = test_data['labels']
        print("Features loaded successfully.")
    else:
        print("Pre-extracted features not found. Extracting features...")
        print("--- Extracting Features for Training Set ---")
        S_train_feat, C_plus_train_feat, C_minus_train_feat, y_train = extract_vgg16_features(train_loader_extract, device)
        print("--- Extracting Features for Test Set ---")
        S_test_feat, C_plus_test_feat, C_minus_test_feat, y_test = extract_vgg16_features(test_loader_extract, device)

        # Save extracted features
        print(f"Saving extracted features to {train_features_file} and {test_features_file}...")
        try:
            np.savez_compressed(train_features_file,
                                s_features=S_train_feat,
                                c_plus_features=C_plus_train_feat,
                                c_minus_features=C_minus_train_feat,
                                labels=y_train)
            np.savez_compressed(test_features_file,
                                s_features=S_test_feat,
                                c_plus_features=C_plus_test_feat,
                                c_minus_features=C_minus_test_feat,
                                labels=y_test)
            print("Features saved successfully.")
        except Exception as e:
            print(f"ERROR saving features: {e}")

    # --- Prepare Concatenated Features for Classifiers --- #
    # Input for classifiers: [S_feat, C_plus_feat, C_minus_feat]
    # Target: y (original label, 0 or 1)
    X_train_concat = np.concatenate((S_train_feat, C_plus_train_feat, C_minus_train_feat), axis=1)
    X_test_concat = np.concatenate((S_test_feat, C_plus_test_feat, C_minus_test_feat), axis=1)
    print(f"Concatenated train features shape: {X_train_concat.shape}")
    print(f"Concatenated test features shape: {X_test_concat.shape}")

    results = {'seed': args.seed}

    # --- Train and Evaluate GMM --- #
    print("\n--- Training and Evaluating GMM --- ")
    try:
        # GMM needs n_components. Using 2 as per original script (binary labels).
        # It might struggle with high-dimensional concatenated features.
        n_components = 2
        # Add covariance_type='diag' to simplify the model
        gmm = GaussianMixture(n_components=n_components, random_state=args.seed, max_iter=200, n_init=5, covariance_type='diag')
        print(f"Fitting GMM with {n_components} components (covariance_type='diag')...")
        gmm.fit(X_train_concat) # Fit on concatenated features

        # Predict based on which component has higher probability for the *correct* class origin? Tricky.
        # A common GMM classification approach: Assign cluster labels, then map clusters to classes.
        train_cluster_labels = gmm.predict(X_train_concat)
        test_cluster_labels = gmm.predict(X_test_concat)

        # Map clusters to majority class label
        # Be careful: This assumes clusters align well with classes, which might not be true.
        map_cluster_to_class = {}
        for cluster_id in range(n_components):
            cluster_mask_train = (train_cluster_labels == cluster_id)
            if np.any(cluster_mask_train):
                 # Find the majority true label within this cluster
                 majority_class = np.bincount(y_train[cluster_mask_train]).argmax()
                 map_cluster_to_class[cluster_id] = majority_class
            else:
                 map_cluster_to_class[cluster_id] = 0 # Default if cluster is empty

        # Apply mapping
        gmm_train_preds = np.array([map_cluster_to_class.get(c, 0) for c in train_cluster_labels])
        gmm_test_preds = np.array([map_cluster_to_class.get(c, 0) for c in test_cluster_labels])

        gmm_train_acc = accuracy_score(y_train, gmm_train_preds)
        gmm_test_acc = accuracy_score(y_test, gmm_test_preds)
        print(f"GMM Training accuracy: {gmm_train_acc:.4f}")
        print(f"GMM Test accuracy: {gmm_test_acc:.4f}")
        results['gmm_test_accuracy'] = gmm_test_acc
        results['gmm_train_accuracy'] = gmm_train_acc
    except Exception as e:
        print(f"ERROR during GMM training/evaluation: {e}")
        results['gmm_test_accuracy'] = None
        results['gmm_train_accuracy'] = None

    # --- Train and Evaluate MLP --- #
    print("\n--- Training and Evaluating MLP --- ")
    try:
        mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation='relu',
            solver='adam',
            max_iter=1000, # Match screenshot
            random_state=args.seed,
            early_stopping=True, # Good practice
            n_iter_no_change=10,
            verbose=False
        )
        print("Fitting MLP...")
        mlp.fit(X_train_concat, y_train)

        mlp_train_preds = mlp.predict(X_train_concat)
        mlp_test_preds = mlp.predict(X_test_concat)

        mlp_train_acc = accuracy_score(y_train, mlp_train_preds)
        mlp_test_acc = accuracy_score(y_test, mlp_test_preds)
        print(f"MLP Training accuracy: {mlp_train_acc:.4f}")
        print(f"MLP Test accuracy: {mlp_test_acc:.4f}")
        results['mlp_test_accuracy'] = mlp_test_acc
        results['mlp_train_accuracy'] = mlp_train_acc
    except Exception as e:
        print(f"ERROR during MLP training/evaluation: {e}")
        results['mlp_test_accuracy'] = None
        results['mlp_train_accuracy'] = None

    # --- Train and Evaluate Logistic Regression --- #
    print("\n--- Training and Evaluating Logistic Regression --- ")
    try:
        # Use large C for less regularization, max_iter for convergence
        logreg = LogisticRegression(random_state=args.seed, max_iter=1000, C=1.0, solver='liblinear')
        print("Fitting Logistic Regression...")
        logreg.fit(X_train_concat, y_train)

        logreg_train_preds = logreg.predict(X_train_concat)
        logreg_test_preds = logreg.predict(X_test_concat)

        logreg_train_acc = accuracy_score(y_train, logreg_train_preds)
        logreg_test_acc = accuracy_score(y_test, logreg_test_preds)
        print(f"Logistic Regression Training accuracy: {logreg_train_acc:.4f}")
        print(f"Logistic Regression Test accuracy: {logreg_test_acc:.4f}")
        results['logreg_test_accuracy'] = logreg_test_acc
        results['logreg_train_accuracy'] = logreg_train_acc
    except Exception as e:
        print(f"ERROR during Logistic Regression training/evaluation: {e}")
        results['logreg_test_accuracy'] = None
        results['logreg_train_accuracy'] = None

    # --- Save Final Results --- #
    print(f"\nSaving final results to {args.results_file}...")
    try:
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print("Results saved successfully.")
    except Exception as e:
        print(f"ERROR saving results: {e}")


if __name__ == "__main__":
    main() 