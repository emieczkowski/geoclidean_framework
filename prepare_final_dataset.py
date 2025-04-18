import os
import pandas as pd
import random
import shutil
import numpy as np

# --- Configuration ---
DATA_DIR = "data_relational_match"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABEL_FILE = os.path.join(DATA_DIR, "train_labels.csv") 

FINAL_TRAIN_LABEL_FILE = os.path.join(DATA_DIR, "train_labels_final.csv") # Output for final training set
TEST_LABELS_FILE = os.path.join(DATA_DIR, "test_labels_final.csv") 

NUM_TEST_SPLIT = 200
EXPECTED_TOTAL_IMAGES = 1000

print("Starting dataset final preparation...")

# --- 1. Load Original Labels ---
print(f"Loading labels from {LABEL_FILE}...")
# --- Restore original logic to read the CSV --- 
if not os.path.exists(LABEL_FILE):
    print(f"Error: Input label file not found at {LABEL_FILE}")
    exit()

try:
    full_labels_df = pd.read_csv(LABEL_FILE)
    print(f"Loaded {len(full_labels_df)} labels from {LABEL_FILE}")
except Exception as e:
    print(f"Error reading {LABEL_FILE}: {e}")
    exit()

# --- 2. Verify Total Count --- (Check against actual labels found)
if len(full_labels_df) < EXPECTED_TOTAL_IMAGES:
    print(f"Warning: Expected {EXPECTED_TOTAL_IMAGES} images based on config, but only found {len(full_labels_df)} labels.")


# --- 3. Random Selection for Test Set ---
if len(full_labels_df) < NUM_TEST_SPLIT:
    print(f"Error: Cannot select {NUM_TEST_SPLIT} test images, only {len(full_labels_df)} total images available.")
    exit()

random.seed(42) # for reproducibility
test_set_indices = random.sample(range(len(full_labels_df)), NUM_TEST_SPLIT)
test_set_df = full_labels_df.iloc[test_set_indices]
test_filenames = test_set_df['filename'].tolist()

print(f"Randomly selected {len(test_filenames)} images for the test set.")

# --- 4. Move Files ---
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
    print(f"Created test directory: {TEST_DIR}")

moved_count = 0
move_errors = 0
print(f"Moving selected images from {TRAIN_DIR} to {TEST_DIR}...")
for filename in test_filenames:
    src_path = os.path.join(TRAIN_DIR, filename)
    dst_path = os.path.join(TEST_DIR, filename)
    try:
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            moved_count += 1
        else:
            print(f"  Warning: Source file not found, cannot move: {src_path}")
            move_errors += 1
    except Exception as e:
        print(f"  Error moving {filename}: {e}")
        move_errors += 1

print(f"Moved {moved_count} files. Encountered {move_errors} errors.")
if move_errors > 0:
    print("Warning: Some files selected for the test set could not be moved.")

# --- 5. Create Final Train Labels ---
final_train_df = full_labels_df.drop(test_set_indices)

try:
    final_train_df.to_csv(FINAL_TRAIN_LABEL_FILE, index=False)
    print(f"Saved final training labels ({len(final_train_df)} entries) to {FINAL_TRAIN_LABEL_FILE}")
except Exception as e:
    print(f"Error saving final train labels: {e}")

# --- 6. Create Test Labels File ---
# test_filenames_df = pd.DataFrame(test_filenames, columns=['filename'])
try:
    # Save the test_set_df which contains filenames and labels
    test_set_df.to_csv(TEST_LABELS_FILE, index=False)
    print(f"Saved test labels ({len(test_set_df)} entries) to {TEST_LABELS_FILE}")
except Exception as e:
    print(f"Error saving test labels list: {e}")

print("\nDataset preparation finished.") 