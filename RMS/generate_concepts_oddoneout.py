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
import json

'''
Generates Odd-One-Out task data from Geoclidean concepts
Creates episodes with 5 inliers (same concept) + 1 outlier (far concept)
'''

# --- Configuration ---
OUTPUT_DIR = "data_concepts_oddoneout_loaded" # Output dir for odd-one-out task
GEOCLIDEAN_ELEMENTS_DIR = "../geoclidean/elements" # Base dir for author concepts
GENERATED_CONCEPTS_DIR = "../geoclidean/generated_concepts" # Base dir for our generated concepts
NUM_EPISODES_PER_CONCEPT = 100 # Number of odd-one-out episodes per concept
MAX_GENERATION_ATTEMPTS = 50 # Max attempts for generating each scene
VISIBILITY_THRESHOLD = 150
MARGIN = 2.0

# Odd-one-out task configuration
INLIERS_PER_EPISODE = 5  # Number of reference/inlier images
OUTLIERS_PER_EPISODE = 1 # Number of oddball/outlier images

# --- Function to Load Concepts ---
def load_concepts_from_multiple_dirs(base_dirs):
    """Loads concept rules from multiple directories."""
    all_loaded_concepts = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Directory not found: {base_dir}")
            continue
            
        loaded_concepts = load_concepts_from_disk(base_dir)
        all_loaded_concepts.extend(loaded_concepts)
    
    return all_loaded_concepts

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
                pos_rules = [line.strip().strip("'\"") for line in f if line.strip()] # Read non-empty lines, remove quotes
            with open(close_path, 'r') as f:
                close_rules = [line.strip().strip("'\"") for line in f if line.strip()]
            # Far rules are required for odd-one-out task
            far_rules = []
            if os.path.exists(far_path):
                 with open(far_path, 'r') as f:
                    far_rules = [line.strip().strip("'\"") for line in f if line.strip()]
            else:
                print(f"    Note: far_concept.txt not found for {concept_name}, skipping concept.")
                continue

            if pos_rules and close_rules and far_rules:
                loaded_concepts.append((concept_name, pos_rules, close_rules, far_rules))
                print(f"    Successfully loaded {concept_name} (Pos: {len(pos_rules)}, Close: {len(close_rules)}, Far: {len(far_rules)} rules)")
            else:
                 print(f"    Warning: Missing rules for {concept_name}. Skipping concept.")

        except FileNotFoundError as e:
            print(f"    Warning: Could not find expected file for {concept_name}: {e}. Skipping concept.")
        except Exception as e:
            print(f"    Error reading files for {concept_name}: {e}. Skipping concept.")
            
    print(f"Finished loading concepts from {base_dir}. Found {len(loaded_concepts)} valid concepts.")
    return loaded_concepts

def get_existing_episode_count(concept_output_dir):
    """Count existing episodes in a concept directory."""
    if not os.path.exists(concept_output_dir):
        return 0
    
    episode_files = glob.glob(os.path.join(concept_output_dir, "episode_*.png"))
    return len(episode_files)

def get_next_episode_idx(concept_output_dir):
    """Get the next available episode index (0-based)."""
    if not os.path.exists(concept_output_dir):
        return 0
    
    episode_files = glob.glob(os.path.join(concept_output_dir, "episode_*.png"))
    if not episode_files:
        return 0
    
    # Extract episode numbers and find the maximum
    episode_nums = []
    for filepath in episode_files:
        filename = os.path.basename(filepath)
        match = re.match(r"episode_(\d+)\.png", filename)
        if match:
            episode_nums.append(int(match.group(1)))
    
    return max(episode_nums) + 1 if episode_nums else 0

# --- Function to Generate Single Scene ---
def generate_single_scene(concept_name, concept_rules, max_attempts, vis_threshold):
    """Generates objects for a single scene using specific rules, ensuring objects are within bounds."""
    attempt = 0

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
                        break 
                    minx, miny, maxx, maxy = obj.bounds
                    if minx < MARGIN or miny < MARGIN or maxx > CANVAS_SIZE - MARGIN or maxy > CANVAS_SIZE - MARGIN:
                        all_within_bounds = False
                        break
                
                # --- Return or Retry --- 
                if all_within_bounds: # Passed generation and bounds checks
                    return generated_objs # Return the list of viewable objects
                else:
                    continue 
            else:
                 continue 
        except Exception as e:
            print(f"    Unexpected Error generating concept ({concept_name}) attempt {attempt}: {e}")
            plt.close()
            continue # Retry on any exception
            
    # If loop finishes without returning, all attempts failed
    print(f"Failed to generate valid scene for concept {concept_name} after {max_attempts} attempts.")
    return None # Return None if all attempts failed

def generate_odd_one_out_episode(concept_name, standard_rules, far_rules, episode_idx, concept_output_dir):
    """Generate a single odd-one-out episode with 5 inliers + 1 outlier."""
    
    print(f"  Generating episode {episode_idx} for {concept_name}...")
    
    # Generate 5 inlier images (standard concept)
    inlier_objects = []
    for i in range(INLIERS_PER_EPISODE):
        objects = generate_single_scene(concept_name, standard_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
        if objects is None:
            print(f"    Failed to generate inlier {i+1}, skipping episode")
            return None
        inlier_objects.append(objects)
    
    # Generate 1 outlier image (far concept)
    outlier_objects = generate_single_scene(concept_name, far_rules, MAX_GENERATION_ATTEMPTS, VISIBILITY_THRESHOLD)
    if outlier_objects is None:
        print(f"    Failed to generate outlier, skipping episode")
        return None
    
    # Create 2x3 layout: 5 inliers + 1 outlier
    # Top row: 3 images, Bottom row: 3 images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.tight_layout(pad=1.0)
    
    # Randomize outlier position (0-5)
    outlier_position = random.randint(0, 5)
    
    positions_layout = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]  # row, col positions
    
    for pos_idx, (row, col) in enumerate(positions_layout):
        ax = axes[row, col]
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        
        # Use outlier at outlier_position, otherwise use inliers in order
        if pos_idx == outlier_position:
            objects_to_plot = outlier_objects
        else:
            # Map position to inlier index (account for outlier position)
            inlier_idx = pos_idx if pos_idx < outlier_position else pos_idx - 1
            objects_to_plot = inlier_objects[inlier_idx]
        
        # Plot objects
        for obj in objects_to_plot:
            ax = plot_obj(ax, obj, color="black")
        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(0, CANVAS_SIZE)
    
    # Save episode image
    episode_filename = f"episode_{episode_idx:04d}.png"
    episode_path = os.path.join(concept_output_dir, episode_filename)
    
    try:
        plt.savefig(episode_path, dpi=150, bbox_inches='tight')
        print(f"    Saved episode: {episode_path}")
        
        # Return episode metadata
        episode_data = {
            'concept_name': concept_name,
            'episode_idx': episode_idx,
            'filename': episode_filename,
            'outlier_position': outlier_position,
            'num_inliers': INLIERS_PER_EPISODE,
            'num_outliers': OUTLIERS_PER_EPISODE
        }
        
        return episode_data
        
    except Exception as e:
        print(f"    Error saving episode {episode_idx}: {e}")
        return None
    finally:
        plt.close(fig)

# --- Function to Generate Dataset --- 
def generate_odd_one_out_dataset(loaded_concepts):
    """Generate odd-one-out dataset organized by concept."""
    print(f"--- Generating Odd-One-Out Dataset ---")
    
    if not loaded_concepts:
        print("Error: No concepts were loaded. Cannot generate dataset.")
        return
    
    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_episodes_data = []
    successful_concepts = 0
    
    for concept_name, standard_rules, close_rules, far_rules in loaded_concepts:
        print(f"\n=== Processing Concept: {concept_name} ===")
        
        # Create concept-specific directory
        concept_output_dir = os.path.join(OUTPUT_DIR, concept_name)
        os.makedirs(concept_output_dir, exist_ok=True)
        
        # Check existing episodes
        existing_count = get_existing_episode_count(concept_output_dir)
        next_episode_idx = get_next_episode_idx(concept_output_dir)
        episodes_needed = NUM_EPISODES_PER_CONCEPT - existing_count
        
        print(f"  Existing episodes: {existing_count}")
        print(f"  Episodes needed: {episodes_needed}")
        print(f"  Next episode index: {next_episode_idx}")
        
        if episodes_needed <= 0:
            print(f"  ✅ {concept_name} already has {existing_count} episodes (>= {NUM_EPISODES_PER_CONCEPT})")
            successful_concepts += 1
            continue
        
        concept_episodes = []
        successful_episodes = 0
        
        # Generate additional episodes for this concept
        current_idx = next_episode_idx
        for _ in range(episodes_needed):
            episode_data = generate_odd_one_out_episode(
                concept_name, standard_rules, far_rules, 
                current_idx, concept_output_dir
            )
            
            if episode_data:
                concept_episodes.append(episode_data)
                successful_episodes += 1
                current_idx += 1
            else:
                print(f"    Failed to generate episode {current_idx}, continuing...")
                current_idx += 1
        
        total_episodes = existing_count + successful_episodes
        if total_episodes > 0:
            # Save concept-specific metadata
            concept_metadata = {
                'concept_name': concept_name,
                'total_episodes': total_episodes,
                'existing_episodes': existing_count,
                'new_episodes': successful_episodes,
                'target_episodes': NUM_EPISODES_PER_CONCEPT,
                'inliers_per_episode': INLIERS_PER_EPISODE,
                'outliers_per_episode': OUTLIERS_PER_EPISODE,
                'new_episodes_data': concept_episodes
            }
            
            metadata_path = os.path.join(concept_output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(concept_metadata, f, indent=2)
            
            all_episodes_data.extend(concept_episodes)
            successful_concepts += 1
            print(f"✅ Completed {concept_name}: {successful_episodes}/{NUM_EPISODES_PER_CONCEPT} episodes")
        else:
            print(f"❌ Failed {concept_name}: No successful episodes generated")
    
    # Save overall dataset metadata
    overall_metadata = {
        'dataset_type': 'odd_one_out',
        'total_concepts': successful_concepts,
        'total_episodes': len(all_episodes_data),
        'inliers_per_episode': INLIERS_PER_EPISODE,
        'outliers_per_episode': OUTLIERS_PER_EPISODE,
        'episodes_per_concept': NUM_EPISODES_PER_CONCEPT,
        'concepts': list(set([ep['concept_name'] for ep in all_episodes_data])),
        'all_episodes': all_episodes_data
    }
    
    overall_metadata_path = os.path.join(OUTPUT_DIR, 'dataset_metadata.json')
    with open(overall_metadata_path, 'w') as f:
        json.dump(overall_metadata, f, indent=2)
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"✅ Successfully processed: {successful_concepts} concepts")
    print(f"📊 Total episodes generated: {len(all_episodes_data)}")
    print(f"💾 Dataset saved to: {OUTPUT_DIR}")
    print(f"📋 Metadata saved to: {overall_metadata_path}")

# --- Main Execution --- 
if __name__ == "__main__":
    # 1. Load concepts from multiple directories (prioritize our generated ones)
    concept_dirs = [GENERATED_CONCEPTS_DIR, GEOCLIDEAN_ELEMENTS_DIR]
    loaded_concepts = load_concepts_from_multiple_dirs(concept_dirs)

    if not loaded_concepts:
        print("Exiting: No concepts loaded successfully.")
        exit()

    # Filter for concepts that are likely to work well (simple ones first)
    simple_concepts = []
    complex_concepts = []
    
    for concept_name, pos_rules, close_rules, far_rules in loaded_concepts:
        # Prioritize our generated concepts and simple original ones
        if (concept_name.startswith('concept_') and 
            any(simple_name in concept_name for simple_name in 
                ['triangle', 'angle', 'pentagon', 'hexagon', 'star', 'circle', 'tangent', 'secant'])):
            simple_concepts.append((concept_name, pos_rules, close_rules, far_rules))
        else:
            complex_concepts.append((concept_name, pos_rules, close_rules, far_rules))
    
    # Start with simple concepts
    test_concepts = simple_concepts[:5] if simple_concepts else loaded_concepts[:3]
    
    print(f"\nTesting with {len(test_concepts)} concepts: {[c[0] for c in test_concepts]}")
    
    # 2. Generate odd-one-out dataset
    generate_odd_one_out_dataset(test_concepts)

    print(f"\nOdd-One-Out dataset generation complete in {OUTPUT_DIR}.")