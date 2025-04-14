import os
import matplotlib.pyplot as plt
from geoclidean_env_euclid import CANVAS_SIZE, generate_objects_from_concept
from plot_utils import initial_plot, plot_obj
import random
import numpy as np
from shapely.geometry import MultiPolygon, Point, MultiPoint

# --- Configuration ---
OUTPUT_DIR = "data_match_to_sample"
MAX_GENERATION_ATTEMPTS = 20 # Reduced attempts per concept generation
VISIBILITY_THRESHOLD = 100
NUM_MATCH_IMAGES = 5 # How many example images to generate
OUTPUT_FILENAME_PREFIX = "mts_house_example_"

# --- House Concept Approximations (Corrected V3 - Valid Syntax) ---
CONCEPT_HOUSE_TRIANGLE_TOP_V3 = [
    # Base quad lines (implicitly define p1, p2, p3, p4)
    'l1 = line(p1(), p2())',
    'l2 = line(p2(), p3())',
    'l3 = line(p3(), p4())',
    'l4 = line(p4(), p1())',
    # Triangle top helpers
    'l5* = line(p3(), p4())',                # Invisible helper line between p3 and p4
    # Helper circles using points ON l5 (implicitly defines p5, p6)
    'c_p3_roof* = circle(p3(), p5(l5))',
    'c_p4_roof* = circle(p4(), p6(l5))',
    # Visible roof lines. Point p7 is defined implicitly here by constraints.
    'l_roof_1 = line(p4(), p7(c_p3_roof, c_p4_roof))', # Define p7(constraints) here
    'l_roof_2 = line(p3(), p7())',                     # Reuse p7() here
    # Distractor circle (implicitly defines p8, p9)
    'c_dist = circle(p8(), p9())'
]

CONCEPT_HOUSE_FLAT_TOP_V3 = [
    # Base lines trying to form a quad (implicitly defines p1, p2, p3, p4)
    'l1 = line(p1(), p2())',
    'l2 = line(p2(), p3())',
    'l3 = line(p3(), p4())', # Defines p4. This line is the flat top
    'l4 = line(p4(), p1())',
    # Distractor line (implicitly defines p5, p6)
    'l_dist = line(p5(), p6())'
]

# --- Function to Generate One Scene --- (Removed spacing checks)
def generate_single_scene(concept_rules, max_attempts, vis_threshold):
    """Generates objects for a single scene using specific rules."""
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            # Generate objects using the provided rules
            generated_objs = generate_objects_from_concept(concept_rules, visibility_threshold=vis_threshold)
            # No internal spacing check needed for single concept rendering
            if generated_objs: # Check if generation was successful
                 return generated_objs # Success
            else:
                 # Handle case where generate_objects_from_concept might return None/empty
                 print(f"    Generation attempt {attempt} yielded no objects. Retrying...")
                 plt.close()
                 continue
        except Exception as e:
            print(f"    Error generating concept ({concept_rules}) attempt {attempt}: {e}. Retrying...")
            plt.close()
            # Fall through to continue the loop
        
    print(f"Failed to generate scene for rules {concept_rules} after {max_attempts} attempts.")
    return None # Failure

# --- Main Execution for Match-to-Sample Image Generation ---

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Generating {NUM_MATCH_IMAGES} Match-to-Sample images...")

img_saved_count = 0
while img_saved_count < NUM_MATCH_IMAGES:
    print(f"Generating image {img_saved_count+1}/{NUM_MATCH_IMAGES}...")

    # Generate Stimulus (Triangle Top)
    print("  Generating Stimulus...")
    stimulus_objects = generate_single_scene(CONCEPT_HOUSE_TRIANGLE_TOP_V3, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
    if stimulus_objects is None: 
        print("  Failed Stimulus generation, skipping this image.")
        continue 

    # Generate Match (Triangle Top)
    print("  Generating Match...")
    match_objects = generate_single_scene(CONCEPT_HOUSE_TRIANGLE_TOP_V3, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
    if match_objects is None: 
        print("  Failed Match generation, skipping this image.")
        continue 

    # Generate Foil (Flat Top)
    print("  Generating Foil...")
    foil_objects = generate_single_scene(CONCEPT_HOUSE_FLAT_TOP_V3, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
    if foil_objects is None: 
        print("  Failed Foil generation, skipping this image.")
        continue 

    # --- Create and Plot Figure --- 
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.delaxes(axes[0,1]) # Remove top-right axis

    # Stimulus (Top Left)
    ax_stim = axes[0,0]
    ax_stim.set_title("Stimulus (Rule A)")
    ax_stim.set_aspect('equal', adjustable='box')
    ax_stim.axis('off')
    for obj in stimulus_objects: ax_stim = plot_obj(ax_stim, obj, color="black")
    ax_stim.set_xlim(0, CANVAS_SIZE); ax_stim.set_ylim(0, CANVAS_SIZE)

    # Match (Bottom Left)
    ax_match = axes[1,0]
    ax_match.set_title("Choice 1 (Rule A)")
    ax_match.set_aspect('equal', adjustable='box')
    ax_match.axis('off')
    for obj in match_objects: ax_match = plot_obj(ax_match, obj, color="black")
    ax_match.set_xlim(0, CANVAS_SIZE); ax_match.set_ylim(0, CANVAS_SIZE)

    # Foil (Bottom Right)
    ax_foil = axes[1,1]
    ax_foil.set_title("Choice 2 (Rule B)")
    ax_foil.set_aspect('equal', adjustable='box')
    ax_foil.axis('off')
    for obj in foil_objects: ax_foil = plot_obj(ax_foil, obj, color="black")
    ax_foil.set_xlim(0, CANVAS_SIZE); ax_foil.set_ylim(0, CANVAS_SIZE)

    plt.tight_layout(pad=3.0) # Add padding

    # Save the combined plot
    output_filename = f"{OUTPUT_FILENAME_PREFIX}_v3_{img_saved_count+1:03d}.png"
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    print(f"  Saving image to {save_path}...")
    fig.savefig(save_path)
    plt.close(fig)
    img_saved_count += 1 # Increment count only on success

print(f"Match-to-Sample image generation complete. Saved {img_saved_count} images.") 