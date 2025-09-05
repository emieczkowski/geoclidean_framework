import os
import matplotlib.pyplot as plt
from geoclidean_env_euclid import CANVAS_SIZE, generate_objects_from_concept
from plot_utils import initial_plot, plot_obj
import random
import copy
import re # Import regex module
import csv # Add csv import for labels
import pandas as pd
from shapely.geometry import Point
import numpy as np # Add numpy for distance calculation
import shapely # Add shapely for isinstance check
import glob # For finding concept directories
import traceback # For detailed error printing

'''
generates all RMS task data from scratch using Geoclidean
'''


# --- Configuration ---
OUTPUT_DIR = "data_concepts_relmatch_loaded" # New output dir to avoid overwriting old
GEOCLIDEAN_ELEMENTS_DIR = "../geoclidean/elements" # Base dir for author concepts
NUM_TRAIN_IMAGES = 800 # Generate fewer examples for testing
NUM_TEST_IMAGES = 200   # Generate fewer examples for testing
MAX_GENERATION_ATTEMPTS = 50 # Increase attempts slightly maybe?
VISIBILITY_THRESHOLD = 150
OUTPUT_FILENAME_PREFIX = "crelmatch_loaded_example_" # New prefix
MARGIN = 2.0

# --- New Function to Load Concepts ---
def load_concepts_from_disk(base_dir):
    """Loads concept rules from .txt files in subdirectories."""
    loaded_concepts = []
    print(f"Scanning for concepts in: {base_dir}")
    concept_dirs = glob.glob(os.path.join(base_dir, '*/'))
    
    if not concept_dirs:
        print(f"Warning: No concept subdirectories found in {base_dir}")
        return []

    for concept_dir in concept_dirs:
        concept_name = os.path.basename(os.path.dirname(concept_dir)) # Get dir name
        print(f"  Loading concept: {concept_name}")
        pos_path = os.path.join(concept_dir, 'concept.txt')
        close_path = os.path.join(concept_dir, 'close_concept.txt')
        far_path = os.path.join(concept_dir, 'far_concept.txt')
        
        try:
            with open(pos_path, 'r') as f:
                pos_rules = [line.strip() for line in f if line.strip()] # Read non-empty lines
            with open(close_path, 'r') as f:
                close_rules = [line.strip() for line in f if line.strip()]
            # Far rules are optional for current task, handle if missing
            far_rules = []
            if os.path.exists(far_path):
                 with open(far_path, 'r') as f:
                    far_rules = [line.strip() for line in f if line.strip()]
            else:
                print(f"    Note: far_concept.txt not found for {concept_name}, skipping far rules.")

            if pos_rules and close_rules:
                loaded_concepts.append((concept_name, pos_rules, close_rules, far_rules))
                print(f"    Successfully loaded {concept_name} (Pos: {len(pos_rules)}, Close: {len(close_rules)}, Far: {len(far_rules)} rules)")
            else:
                 print(f"    Warning: Missing positive or close rules for {concept_name}. Skipping concept.")

        except FileNotFoundError as e:
            print(f"    Warning: Could not find expected file for {concept_name}: {e}. Skipping concept.")
        except Exception as e:
            print(f"    Error reading files for {concept_name}: {e}. Skipping concept.")
            
    print(f"Finished loading concepts. Found {len(loaded_concepts)} valid concepts.")
    return loaded_concepts

# --- Function to Generate One Scene --- 
# Added concept_name parameter for conditional checks
def generate_single_scene(concept_name, concept_rules, max_attempts, vis_threshold):
    """Generates objects for a single scene using specific rules, ensuring objects are within bounds."""
    attempt = 0
    # closure_tolerance = 1e-4 # No longer needed

    while attempt < max_attempts:
        attempt += 1
        plt.close('all') 
        generated_objs = None # Initialize
        named_objs = None
        try:
            # Get both viewable objects and the dictionary of named objects
            generated_objs, named_objs = generate_objects_from_concept(concept_rules, visibility_threshold=vis_threshold)
            
            # Check if generation succeeded (generate_objects_from_concept returns None, None on failure)
            if generated_objs is not None:
                # --- Visibility Check (Bounds Check) ---
                all_within_bounds = True
                for obj in generated_objs:
                    if obj is None or not hasattr(obj, 'bounds'): 
                        all_within_bounds = False 
                        # print(f"    Generation attempt {attempt}: Invalid object {obj} found. Retrying...") # Verbose
                        break 
                    minx, miny, maxx, maxy = obj.bounds
                    if minx < MARGIN or miny < MARGIN or maxx > CANVAS_SIZE - MARGIN or maxy > CANVAS_SIZE - MARGIN:
                        all_within_bounds = False
                        # print(f"    Generation attempt {attempt}: Object out of bounds or margin {obj.bounds}. Retrying...") # Verbose
                        break
                
                # --- Return or Retry --- 
                if all_within_bounds: # Passed generation and bounds checks
                    return generated_objs # Return the list of viewable objects
                else:
                    # Failed bounds check, retry
                    # print(f"    Attempt {attempt}: Failed bounds check. Retrying...") # Verbose
                    continue 
            else:
                 # generate_objects_from_concept returned None, None (likely internal generation error)
                 # print(f"    Generation attempt {attempt} yielded no objects (returned None). Retrying...") # Verbose
                 continue 
        except Exception as e:
            # Catch any other unexpected error during generation or checks
            print(f"    Unexpected Error generating concept ({concept_name}) attempt {attempt}: {e}. Retrying...") # Updated concept name in log
            traceback.print_exc() # Optional: print full traceback
            plt.close()
            continue # Retry on any exception
            
    # If loop finishes without returning, all attempts failed
    print(f"Failed to generate valid scene for concept {concept_name} after {max_attempts} attempts.") # Updated log message
    return None # Return None if all attempts failed

# --- Function to Generate Dataset --- 
def generate_dataset(loaded_concepts, num_images, target_dir, label_file_path):
    # Takes loaded_concepts as input now
    print(f"--- Generating {os.path.basename(target_dir)} Set ({num_images} images) ---")
    
    if not loaded_concepts:
        print("Error: No concepts were loaded. Cannot generate dataset.")
        return
        
    labels_list = []
    start_img_idx = 0

    # Resume Logic (Simplified: assumes starting fresh or continuing sequentially)
    existing_nums = []
    if os.path.isdir(target_dir):
        try:
            for filename in os.listdir(target_dir):
                match = re.match(rf"{OUTPUT_FILENAME_PREFIX}(\d+)\.png$", filename)
                if match:
                    existing_nums.append(int(match.group(1)))
        except Exception as e:
            print(f"Warning: Error listing existing files in {target_dir}: {e}")
    if existing_nums:
        start_img_idx = max(existing_nums) + 1
        print(f"Resuming from index {start_img_idx}")
        # Basic load of existing labels
        if os.path.exists(label_file_path):
             try:
                 labels_df = pd.read_csv(label_file_path)
                 labels_list = labels_df.to_dict('records')
                 print(f"Loaded {len(labels_list)} labels from {label_file_path}")
             except Exception as e:
                 print(f"Warning: Could not load existing labels: {e}")
                 labels_list = []
        else:
            labels_list = []
    else:
        print(f"Starting fresh generation in {target_dir}")
        labels_list = []

    # --- Generation Loop ---
    img_saved_count = len([n for n in existing_nums]) # Count existing valid files
    target_total_images = img_saved_count + num_images # Total to reach
    current_img_idx = start_img_idx

    while img_saved_count < target_total_images:
        actual_img_num_to_generate = current_img_idx
        print(f"\nGenerating image index {actual_img_num_to_generate} (Target: {target_total_images})...")

        # --- Randomly select a concept from the loaded list --- 
        concept_name, standard_rules, close_foil_rules, far_foil_rules = random.choice(loaded_concepts)
        relational_rules = standard_rules # Standard and Match use the positive rules
        
        # *** Use the FAR rules for the foil ***
        # Make sure far_foil_rules were actually loaded, otherwise fallback or skip
        if not far_foil_rules:
            print(f"  Warning: No 'far' rules found for concept {concept_name}. Skipping this concept for image generation.")
            continue # Skip to the next iteration of the while loop
            
        foil_rules = far_foil_rules     # Use the FAR negative as the foil
        
        print(f"  Using concept: {concept_name}")

        # Generate components, passing concept_name
        print(f"  Generating Standard ({concept_name})...")
        standard_objects = generate_single_scene(concept_name, standard_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
        if standard_objects is None: 
            print("  Skipping image due to Standard generation failure.")
            continue

        print(f"  Generating Match ({concept_name})...")
        match_objects = generate_single_scene(concept_name, relational_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
        if match_objects is None: 
            print("  Skipping image due to Match generation failure.")
            continue

        print(f"  Generating Foil ({concept_name} - Far Foil)...") # Update log message
        foil_objects = generate_single_scene(concept_name, foil_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
        if foil_objects is None: 
            print("  Skipping image due to Foil generation failure.")
            continue

        # --- Randomize choice position --- 
        correct_label = random.choice([0, 1]) # 0 for left, 1 for right
        if correct_label == 0:
            left_choice_objs = match_objects
            right_choice_objs = foil_objects
        else: # correct_label == 1
            left_choice_objs = foil_objects
            right_choice_objs = match_objects
        
        # --- Create and Plot Figure --- 
        # Use 2x2 layout like original relational_match_task.py for consistency with VGG input processing
        print("  Plotting...")
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.delaxes(axes[0,1]) # Remove top-right axis
        plt.tight_layout(pad=1.0) # Reduce padding

        # Standard (Top Left)
        ax_stim = axes[0,0]
        ax_stim.set_aspect('equal', adjustable='box')
        ax_stim.axis('off')
        for obj in standard_objects: ax_stim = plot_obj(ax_stim, obj, color="black")
        ax_stim.set_xlim(0, CANVAS_SIZE); ax_stim.set_ylim(0, CANVAS_SIZE)

        # Left Choice (Bottom Left - Index 0)
        ax_left = axes[1,0]
        ax_left.set_aspect('equal', adjustable='box')
        ax_left.axis('off')
        for obj in left_choice_objs: ax_left = plot_obj(ax_left, obj, color="black")
        ax_left.set_xlim(0, CANVAS_SIZE); ax_left.set_ylim(0, CANVAS_SIZE)

        # Right Choice (Bottom Right - Index 1)
        ax_right = axes[1,1]
        ax_right.set_aspect('equal', adjustable='box')
        ax_right.axis('off')
        for obj in right_choice_objs: ax_right = plot_obj(ax_right, obj, color="black")
        ax_right.set_xlim(0, CANVAS_SIZE); ax_right.set_ylim(0, CANVAS_SIZE)

        # --- Save the combined plot --- 
        output_filename = f"{OUTPUT_FILENAME_PREFIX}{actual_img_num_to_generate:04d}.png"
        save_path = os.path.join(target_dir, output_filename)
        try:
            plt.savefig(save_path)
            print(f"  Saved: {save_path}")
            # *** Add concept_name to the saved label data ***
            labels_list.append({
                'filename': output_filename, 
                'label': correct_label,
                'concept_name': concept_name # Added concept name
            })
            img_saved_count += 1
            current_img_idx += 1 # Increment index only on successful save
        except Exception as e:
            print(f"Error saving image {save_path}: {e}")
            # Decide how to handle save error - skip index for now
            current_img_idx += 1
        finally:
            plt.close(fig) # Close the specific figure

    # --- Save Labels CSV --- 
    if labels_list:
        try:
            labels_df = pd.DataFrame(labels_list)
            # Ensure unique filenames, keep latest
            labels_df = labels_df.drop_duplicates(subset=['filename'], keep='last')
             # Sort by index
            # Ensure extraction handles potential leading zeros if prefix changes
            labels_df['img_index'] = labels_df['filename'].str.extract(rf'{re.escape(OUTPUT_FILENAME_PREFIX)}(\d+)\.png$').astype(int)
            labels_df = labels_df.sort_values('img_index').drop(columns=['img_index'])
            labels_df.to_csv(label_file_path, index=False)
            print(f"Saved/Updated {len(labels_df)} labels to {label_file_path}")
        except Exception as e:
            print(f"Error saving label file {label_file_path}: {e}")
    else:
        print("No labels generated to save.")

    print(f"--- Finished {os.path.basename(target_dir)} Set Generation ---")

# --- Main Execution --- 
if __name__ == "__main__":
    # 1. Load concepts from disk
    loaded_concepts = load_concepts_from_disk(GEOCLIDEAN_ELEMENTS_DIR)

    if not loaded_concepts:
        print("Exiting: No concepts loaded successfully.")
        exit()

    # 2. Create output directories (using new OUTPUT_DIR)
    TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
    TEST_DIR = os.path.join(OUTPUT_DIR, "test")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_label_file = os.path.join(OUTPUT_DIR, "train_labels.csv")
    test_label_file = os.path.join(OUTPUT_DIR, "test_labels.csv")

    # 3. Generate datasets using loaded concepts
    generate_dataset(loaded_concepts, NUM_TRAIN_IMAGES, TRAIN_DIR, train_label_file)
    generate_dataset(loaded_concepts, NUM_TEST_IMAGES, TEST_DIR, test_label_file)

    print(f"\nNew dataset generation complete in {OUTPUT_DIR}.")
