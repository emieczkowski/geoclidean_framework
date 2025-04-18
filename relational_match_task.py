import os
import matplotlib.pyplot as plt
from geoclidean_env_euclid import CANVAS_SIZE, generate_objects_from_concept
from plot_utils import initial_plot, plot_obj
import random
import copy
import re
import csv 
import pandas as pd

# --- Configuration ---
OUTPUT_DIR = "data_relational_match"
NUM_TRAIN_IMAGES = 1000 
NUM_TEST_IMAGES = 0  
MAX_GENERATION_ATTEMPTS = 30 
VISIBILITY_THRESHOLD = 150
OUTPUT_FILENAME_PREFIX = "relmatch_example_"
MARGIN = 2.0


# --- Base Concepts ---
CONCEPT_2L_1C = [
    'l1 = line(p1(), p2())',
    'l2 = line(p3(), p4())',
    'c1 = circle(p5(), p6())'
]
CONCEPT_1C = [
    'c1 = circle(p1(), p2())'
]
CONCEPT_3_LINES = [
    'l1 = line(p1(), p2())',
    'l2 = line(p3(), p4())',
    'l3 = line(p5(), p6())'
]
CONCEPT_2_LINES_INDEP = [
    'l1 = line(p1(), p2())',
    'l2 = line(p3(), p4())'
]
CONCEPT_INTERSECTING_LINES = [
    'l1 = line(p1(), p2())',
    'l2 = line(p1(), p3())' # Share p1
]
# Non-intersecting lines use independent points
CONCEPT_NON_INTERSECTING_LINES = CONCEPT_2_LINES_INDEP # Re-use definition

CONCEPT_INTERSECTING_CIRCLES = [
    'c1 = circle(p1(), p2())',
    'c2 = circle(p2(), p3())' # Center on circumference
]
CONCEPT_NON_INTERSECTING_CIRCLES = [
    'c1 = circle(p1(), p2())',
    'c2 = circle(p3(), p4())' # Independent points
]
CONCEPT_2L_2C = [
    'l1 = line(p1(), p2())',
    'l2 = line(p3(), p4())',
    'c1 = circle(p5(), p6())',
    'c2 = circle(p7(), p8())'
]
# Foil for 2L_2C uses CONCEPT_2L_1C (2 lines, 1 circle)
CONCEPT_3L_1C = [
    'l1 = line(p1(), p2())',
    'l2 = line(p3(), p4())',
    'l3 = line(p5(), p6())',
    'c1 = circle(p7(), p8())'
]

# --- List of possible relations (Revised) ---
RELATIONS = [
    ("Count_2L1C", CONCEPT_2L_1C, CONCEPT_2L_1C, CONCEPT_1C),
    ("Count_3Lines", CONCEPT_3_LINES, CONCEPT_3_LINES, CONCEPT_2_LINES_INDEP),
    ("Intersect_Lines", CONCEPT_INTERSECTING_LINES, CONCEPT_INTERSECTING_LINES, CONCEPT_NON_INTERSECTING_LINES),
    ("Count_2L2C", CONCEPT_2L_2C, CONCEPT_2L_2C, CONCEPT_2L_1C),
    ("Count_2L2C", CONCEPT_2L_2C, CONCEPT_2L_2C, CONCEPT_2L_1C),
    ("Count_3L1C", CONCEPT_3L_1C, CONCEPT_3L_1C, CONCEPT_3_LINES)
]

# --- Function to Generate One image task --- 
def generate_single_scene(concept_rules, max_attempts, vis_threshold):
    """Generates objects for a single scene using specific rules, ensuring objects are within bounds."""
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        # Clear previous plots immediately before generation attempt
        plt.close('all') 
        try:
            generated_objs = generate_objects_from_concept(concept_rules, visibility_threshold=vis_threshold)

            if generated_objs:
                 # --- Add bounds check --- 
                 all_visible = True
                 for obj in generated_objs:
                     # Check if the object has bounds (assuming shapely object)
                     if not hasattr(obj, 'bounds'):
                         print(f"    Warning: Cannot check bounds for object type {type(obj)}. Assuming visible.")
                         continue # Skip check for this object
                     
                     minx, miny, maxx, maxy = obj.bounds
                     if minx < MARGIN or miny < MARGIN or maxx > CANVAS_SIZE - MARGIN or maxy > CANVAS_SIZE - MARGIN:
                         all_visible = False
                         print(f"    Generation attempt {attempt}: Object out of bounds or margin {obj.bounds}. Retrying...")
                         # No need to close plot here as it's closed at the start of the loop
                         break # Exit inner loop, retry generation
                 
                 if all_visible:
                     return generated_objs # Success!
                 else:
                     continue # Retry generation (all_visible is False)
                 # --- End bounds check ---
            else:
                 print(f"    Generation attempt {attempt} yielded no objects. Retrying...")
                 # No need to close plot here
                 continue
        except Exception as e:
            print(f"    Error generating concept ({concept_rules}) attempt {attempt}: {e}. Retrying...")
            plt.close()

    print(f"Failed to generate scene for rules {concept_rules} after {max_attempts} attempts.")
    return None

# --- Main Execution --- 

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")
if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

# --- Generate Training Set --- 
print(f"Generating {NUM_TRAIN_IMAGES} Training images...")
train_labels = []
start_img_idx = 0
label_file_path = os.path.join(OUTPUT_DIR, "train_labels.csv")

'''
# --- Resume Logic (if generation halted with existing files) --- 
print("Checking for existing training images to resume...")
existing_nums = []
if os.path.isdir(TRAIN_DIR):
    try:
        for filename in os.listdir(TRAIN_DIR):
            match = re.match(rf"{OUTPUT_FILENAME_PREFIX}(\d+)\.png$", filename)
            if match:
                existing_nums.append(int(match.group(1)))
    except Exception as e:
        print(f"Error listing existing files in {TRAIN_DIR}: {e}")

if existing_nums:
    start_img_idx = max(existing_nums)
    print(f"Found existing images. Resuming training image generation after index {start_img_idx}")
    # Load existing labels to append to, filtering potentially orphaned labels
    if os.path.exists(label_file_path):
        try:
            existing_labels_df = pd.read_csv(label_file_path)
            # Filter labels to only include those whose image number <= start_img_idx
            valid_labels_df = existing_labels_df[existing_labels_df['filename'].str.extract(r'_(\d{4})\.png$').astype(int)[0] <= start_img_idx]
            train_labels = valid_labels_df.values.tolist()
            print(f"Loaded {len(train_labels)} valid labels from existing CSV.")
        except Exception as e:
            print(f"Error reading/filtering existing label file: {e}. Starting label list fresh.")
            train_labels = [] # Start fresh if error reading/filtering
    else:
        print("Label file not found, but found existing images. Labels will be regenerated/appended.")
        train_labels = [] # Will append all labels
else:
    print("No existing training images found in {TRAIN_DIR}. Starting from index 0.")
    start_img_idx = 0
    train_labels = [] # Ensure labels list is empty if starting fresh

# --- End Resume Logic ---

'''
img_saved_count = start_img_idx 

while img_saved_count < NUM_TRAIN_IMAGES:
    # Image counter for printing should be based on the target index
    current_img_num = img_saved_count + 1 
    print(f"\nGenerating training image {current_img_num}/{NUM_TRAIN_IMAGES}...")

    # --- Randomly select a relation type for this image ---
    relation_name, standard_rules, relational_rules, nonrelational_rules = random.choice(RELATIONS)
    print(f"  Using relation: {relation_name}")

    # Generate components based on selected rules
    print(f"  Generating Standard ({relation_name})...")
    standard_objects = generate_single_scene(standard_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
    if standard_objects is None: 
        print("  Skipping image due to generation failure.")
        continue

    print(f"  Generating Relational Match ({relation_name})...")
    relational_objects = generate_single_scene(relational_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
    if relational_objects is None: 
        print("  Skipping image due to generation failure.")
        continue

    print(f"  Generating Non-Relational Match ({relation_name})...")
    nonrelational_objects = generate_single_scene(nonrelational_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
    if nonrelational_objects is None: 
        print("  Skipping image due to generation failure.")
        continue

    # --- Randomize choice position --- 
    correct_label = random.choice([0, 1]) # 0 for left, 1 for right
    if correct_label == 0:
        left_choice_objs = relational_objects
        right_choice_objs = nonrelational_objects
        left_title = "Relational Match" # Keep titles for now, will remove later
        right_title = "Non-Relational Match"
    else: # correct_label == 1
        left_choice_objs = nonrelational_objects
        right_choice_objs = relational_objects
        left_title = "Non-Relational Match"
        right_title = "Relational Match"
    # --- End Randomization --- 

    # Create and Plot Figure
    print("  Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.delaxes(axes[0,1]) # Remove top-right axis

    # Standard (Top Left)
    ax_stim = axes[0,0]
    # ax_stim.set_title("Standard") # REMOVE TITLE
    ax_stim.set_aspect('equal', adjustable='box')
    ax_stim.axis('off')
    for obj in standard_objects: ax_stim = plot_obj(ax_stim, obj, color="black")
    ax_stim.set_xlim(0, CANVAS_SIZE); ax_stim.set_ylim(0, CANVAS_SIZE)

    # Left Choice (Bottom Left - Index 0)
    ax_left = axes[1,0]
    # ax_left.set_title(left_title) # REMOVE TITLE
    ax_left.set_aspect('equal', adjustable='box')
    ax_left.axis('off')
    for obj in left_choice_objs: ax_left = plot_obj(ax_left, obj, color="black")
    ax_left.set_xlim(0, CANVAS_SIZE); ax_left.set_ylim(0, CANVAS_SIZE)

    # Right Choice (Bottom Right - Index 1)
    ax_right = axes[1,1]
    # ax_right.set_title(right_title) # REMOVE TITLE
    ax_right.set_aspect('equal', adjustable='box')
    ax_right.axis('off')
    for obj in right_choice_objs: ax_right = plot_obj(ax_right, obj, color="black")
    ax_right.set_xlim(0, CANVAS_SIZE); ax_right.set_ylim(0, CANVAS_SIZE)

    plt.tight_layout(pad=3.0) 

    # Save the combined plot to the train directory
    output_filename = f"{OUTPUT_FILENAME_PREFIX}{current_img_num:04d}.png" 
    save_path = os.path.join(TRAIN_DIR, output_filename)
    print(f"  Saving training image to {save_path}...")
    try:
        fig.savefig(save_path)
        # Check if this label already exists from resume (shouldn't, but as safeguard)
        if not any(fname == output_filename for fname, _ in train_labels):
             # Append new label with the CORRECT label (0 or 1)
             train_labels.append([output_filename, correct_label]) # USE correct_label
        img_saved_count += 1 # Increment the count of *successfully saved* images
    except Exception as e:
        print(f"  Error saving image {save_path}: {e}")
    finally:
        plt.close(fig) # Ensure plot is closed even on error

# Save training labels to CSV
# Decide write mode based on whether we resumed
write_mode = 'a' if start_img_idx > 0 else 'w'
write_header = start_img_idx == 0 # Write header only if starting fresh

print(f"\nSaving training labels to {label_file_path} (mode: {write_mode}, header: {write_header})...")
# Find the index in train_labels corresponding to start_img_idx + 1
first_new_label_index = 0
if start_img_idx > 0:
    # Find the index where the new labels start
    try:
       # Find the position where filenames correspond to numbers > start_img_idx
       for i, (fname, _) in enumerate(train_labels):
           num_match = re.match(rf"{OUTPUT_FILENAME_PREFIX}(\d+)\.png$", fname)
           if num_match and int(num_match.group(1)) > start_img_idx:
               first_new_label_index = i
               break
       else: # If loop finishes without break, all labels are old
           first_new_label_index = len(train_labels)
    except Exception as e:
         print(f"Warning: Error finding index for appending labels: {e}. Saving all labels.")
         first_new_label_index = 0 # Default to rewriting all if error

labels_to_save = train_labels[first_new_label_index:]

try:
    # If appending, we only write the new rows
    # If writing fresh (w mode), labels_to_save contains everything anyway
    with open(label_file_path, write_mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(['filename', 'label']) # Header
        writer.writerows(labels_to_save) 
    print(f"Saved {len(labels_to_save)} new training labels. Total labels in memory: {len(train_labels)}.")
except Exception as e:
    print(f"Error writing labels to CSV: {e}")

# --- Generate Test Set (Skip for this run) --- 
# print(f"\nGenerating {NUM_TEST_IMAGES} Testing images...")
# ... (ensure test loop is commented out or handled by NUM_TEST_IMAGES=0)

print(f"\nDataset generation complete. Total training images generated up to index {img_saved_count}.") 

# --- Placeholder for Model Training/Evaluation --- (Updated)
print("\n--- Next Steps (Model Training - Placeholder) ---")
print(f"1. Load training images from '{TRAIN_DIR}' and labels from '{label_file_path}'.")
print(f"2. Load test images from '{TEST_DIR}' (no labels provided).")
print("3. Preprocess images: Crop Standard, Relational Match, and Non-Relational Match panels.")
print("4. Define a model architecture (e.g., Siamese/Triplet with VGG features) taking the three panels as input.")
print("5. Train the model on the training set to predict which choice (left/0 or right/1) is the Relational Match.")
print("6. Evaluate the model on the test set.") 