import os
import random
import itertools

# --- Configuration ---
# Directory containing the generated scene images (adjust if you generated more elsewhere)
# It should contain subdirectories for different categories/types if generated that way.
DATASET_DIR = "data_composite_example"
NUM_PAIRS_TO_SAMPLE = 20  # How many pairs to sample for the similarity test


def find_image_files(directory):
    """Recursively finds all .png image files in a directory."""
    image_paths = []
    if not os.path.isdir(directory):
        print(f"Error: Dataset directory '{directory}' not found.")
        return []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

# --- Main Execution ---
if __name__ == "__main__":
    all_image_files = find_image_files(DATASET_DIR)

    if len(all_image_files) < 2:
        print(f"Error: Found {len(all_image_files)} images in '{DATASET_DIR}'. Need at least 2 to sample pairs.")
        exit()

    print(f"Found {len(all_image_files)} images in '{DATASET_DIR}'.")
    print(f"Sampling {NUM_PAIRS_TO_SAMPLE} pairs...")

    sampled_pairs = []

    # Generate all possible unique pairs
    possible_pairs = list(itertools.combinations(all_image_files, 2))

    if len(possible_pairs) < NUM_PAIRS_TO_SAMPLE:
        print(f"Warning: Requested {NUM_PAIRS_TO_SAMPLE} pairs, but only {len(possible_pairs)} unique pairs are possible. Sampling all possible pairs.")
        sampled_pairs = possible_pairs
    else:
        # Sample unique pairs randomly
        sampled_indices = random.sample(range(len(possible_pairs)), NUM_PAIRS_TO_SAMPLE)
        sampled_pairs = [possible_pairs[i] for i in sampled_indices]

    print("\n--- Sampled Image Pairs for Similarity Test ---")
    for i, (img1_path, img2_path) in enumerate(sampled_pairs):
        print(f"Pair {i+1}:")
        print(f"  Image 1: {img1_path}")
        print(f"  Image 2: {img2_path}")
        print("--")

    # --- Placeholder for CNN Similarity Steps ---
    print("\n--- Next Steps (CNN Similarity - Placeholder) ---")
    print("1. Define/Load a CNN model (e.g., pre-trained on ImageNet like ResNet, VGG, or a custom model).")
    print("2. Adapt the CNN: Often remove the final classification layer and use the output of a preceding layer as features.")
    print("3. For each pair in sampled_pairs:")
    print("    a. Load and preprocess both images (resize, normalize) to match CNN input requirements.")
    print("    b. Pass each image through the CNN to get its feature vector.")
    print("    c. Calculate the similarity between the two feature vectors (e.g., using cosine similarity).")
    print("    d. Store the similarity score along with the image pair paths.")
    print("4. Analyze or use the calculated similarity scores.")
    print("5. Consider fine-tuning the CNN on a relevant dataset or using contrastive learning if pre-trained features are insufficient.") 