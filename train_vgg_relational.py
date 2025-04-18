import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# --- Constants ---
DATA_DIR = "data_relational_match"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABEL_FILE = os.path.join(DATA_DIR, "train_labels_final.csv")
TEST_LABEL_FILE = os.path.join(DATA_DIR, "test_labels_final.csv")

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Checkpoint directory
CHECKPOINT_DIR = "checkpoints/vgg_relational"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model_epoch_{epoch:02d}.weights.h5")


def load_and_preprocess_data(is_train=True):
    print(f"Loading data... (is_train={is_train})")
    
    image_dir = TRAIN_DIR if is_train else TEST_DIR
    label_path = LABEL_FILE if is_train else TEST_LABEL_FILE
    
    if not os.path.exists(label_path):
        print(f"Error: Label file {label_path} not found.")
        return None, None, None, None

    try:
        labels_df = pd.read_csv(label_path)
        filenames = labels_df['filename'].tolist()
        labels = labels_df['label'].values
        print(f"  Loaded labels for {len(labels)} images from {label_path}.")
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return None, None, None, None

    if not filenames:
        print("  No image files found in label file.")
        return [], [], [], [] # Return empty lists if no files

    # --- Determine Crop Boxes from First Image ---
    crop_std, crop_match, crop_foil = None, None, None
    try:
        example_img_path = os.path.join(image_dir, filenames[0])
        with Image.open(example_img_path) as img:
            actual_img_width, actual_img_height = img.size
            panel_w = actual_img_width // 2
            panel_h = actual_img_height // 2
            crop_std = (0, 0, panel_w, panel_h) # Top-left
            crop_match = (0, panel_h, panel_w, actual_img_height) # Bottom-left
            crop_foil = (panel_w, panel_h, actual_img_width, actual_img_height) # Bottom-right
            print(f"  Using crop boxes based on {filenames[0]} ({actual_img_width}x{actual_img_height}): Std={crop_std}, Match={crop_match}, Foil={crop_foil}")
    except FileNotFoundError:
        print(f"Error: Example image {example_img_path} not found. Cannot determine crop boxes.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading example image {example_img_path}: {e}")
        return None, None, None, None
        
    # --- Process Images --- 
    all_panels_std = []
    all_panels_match = []
    all_panels_foil = []
    valid_indices = [] # Keep track of successfully processed images

    print(f"  Processing {len(filenames)} images...")
    for idx, fname in enumerate(filenames):
        try:
            img_path = os.path.join(image_dir, fname)
            with Image.open(img_path) as img:
                img = img.convert('RGB') # Ensure 3 channels
                
                panel_std = img.crop(crop_std).resize((IMG_WIDTH, IMG_HEIGHT))
                panel_match = img.crop(crop_match).resize((IMG_WIDTH, IMG_HEIGHT))
                panel_foil = img.crop(crop_foil).resize((IMG_WIDTH, IMG_HEIGHT))
                
                # Preprocess using VGG16 specific function
                std_arr = tf.keras.applications.vgg16.preprocess_input(np.array(panel_std))
                match_arr = tf.keras.applications.vgg16.preprocess_input(np.array(panel_match))
                foil_arr = tf.keras.applications.vgg16.preprocess_input(np.array(panel_foil))

                all_panels_std.append(std_arr)
                all_panels_match.append(match_arr)
                all_panels_foil.append(foil_arr)
                valid_indices.append(idx) # Add index if processing succeeded

        except FileNotFoundError:
            print(f"  Warning: Image {fname} not found. Skipping.")
        except Exception as e:
            print(f"  Warning: Error processing image {fname}: {e}. Skipping.")

    if not valid_indices:
         print("  No images processed successfully.")
         return [], [], [], []

    # Filter labels to match successfully processed images
    filtered_labels = labels[valid_indices]

    print(f"  Finished processing. Returning {len(all_panels_std)} sets of panels.")
    return np.array(all_panels_std), np.array(all_panels_match), np.array(all_panels_foil), filtered_labels


# --- Model Definition ---
def build_vgg_relational_model(input_shape):
    print("Building VGG relational model...")
    
    input_std = layers.Input(shape=input_shape, name="StandardPanel")
    input_match = layers.Input(shape=input_shape, name="MatchPanel")
    input_foil = layers.Input(shape=input_shape, name="FoilPanel")

    base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    feature_extractor = Model(inputs=base_vgg.input, 
                              outputs=base_vgg.get_layer("block5_pool").output, 
                              name="VGG16_FeatureExtractor")
    feature_extractor.trainable = True 

    features_std = feature_extractor(input_std)
    features_match = feature_extractor(input_match)
    features_foil = feature_extractor(input_foil)

    flat_std = layers.Flatten()(features_std)
    flat_match = layers.Flatten()(features_match)
    flat_foil = layers.Flatten()(features_foil)

    # --- Relational Comparison (L1 Distance) ---
    l1_dist_match = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([flat_std, flat_match])
    l1_dist_foil = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([flat_std, flat_foil])

    # Concatenate the distance vectors
    concatenated_distances = layers.Concatenate()([l1_dist_match, l1_dist_foil])

    # --- Classification Head ---
    x = layers.Dense(256, activation='relu')(concatenated_distances) # Reduced size
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    # Output: Predicts P(Match is correct choice). Label=0 means Match is correct.
    output = layers.Dense(1, activation='sigmoid', name="output")(x)

    # --- Create and Compile Model ---
    model = Model(inputs=[input_std, input_match, input_foil], outputs=output, name="VGG_RelationalMatchModel")
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("Model built and compiled.")
    model.summary()
    return model

# --- Training Loop ---
def train_model(model, train_data, train_labels, val_data, val_labels, epochs=20, batch_size=16): # Smaller batch size
    print(f"Starting training for {epochs} epochs...")
    
    if not isinstance(train_data, (list, tuple)) or len(train_data) != 3:
        print("Error: train_data must be a list/tuple of (std, match, foil) arrays.")
        return None
    if not isinstance(val_data, (list, tuple)) or len(val_data) != 3:
        print("Error: val_data must be a list/tuple of (std, match, foil) arrays.")
        return None
        
    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, # Stop after 5 epochs with no improvement in val_loss
        verbose=1, 
        restore_best_weights=True
    )

    history = model.fit(
        train_data, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(val_data, val_labels),
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    
    print("Training finished.")
    return history

# --- Main Execution ---
if __name__ == "__main__":
    # Define epochs and batch_size here to use them later for saving
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4 # Ensure this matches the compiled model LR
    
    print("--- Starting VGG Relational Model Training ---")
    
    # 1. Load Data
    print("\n--- Loading Training Data ---")
    panels_std_train, panels_match_train, panels_foil_train, labels_train = load_and_preprocess_data(is_train=True)
    
    print("\n--- Loading Test Data ---")
    panels_std_test, panels_match_test, panels_foil_test, labels_test = load_and_preprocess_data(is_train=False)

    if panels_std_train is None or panels_std_test is None:
        print("Failed to load data. Exiting.")
        exit()
    
    if len(panels_std_train) == 0 or len(panels_std_test) == 0:
        print("No data loaded successfully. Exiting.")
        exit()

    # Combine into list format for model input
    train_dataset = [panels_std_train, panels_match_train, panels_foil_train]
    test_dataset = [panels_std_test, panels_match_test, panels_foil_test]

    # 2. Build Model
    print("\n--- Building Model ---")
    model = build_vgg_relational_model(IMG_SHAPE)

    # 3. Train Model
    print("\n--- Training Model ---")
    # Using test set as validation for simplicity here, consider a separate validation split
    history = train_model(model, train_dataset, labels_train, test_dataset, labels_test, epochs=EPOCHS, batch_size=BATCH_SIZE) 

    # 4. Evaluate Final Model (using the best weights restored by EarlyStopping)
    print("\n--- Evaluating Final Model on Test Set ---")
    loss, accuracy = model.evaluate(test_dataset, labels_test, verbose=0)
    print(f"Final Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy:.4f}")

    # 5. Save final metrics if training was successful
    if history:
        results_dir = "results/vgg_relational"
        os.makedirs(results_dir, exist_ok=True)
        import json
        
        metrics_file = os.path.join(results_dir, "final_metrics.json")
        final_metrics = {
            'test_loss': float(loss),
            'test_accuracy': float(accuracy),
            'training_params': {
                'epochs_configured': EPOCHS,
                'epochs_run': len(history.epoch), # Actual epochs run due to early stopping
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'backbone': 'VGG16 (end-to-end)'
            }
        }
        try:
            with open(metrics_file, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            print(f"Final metrics saved to {metrics_file}")
        except Exception as e:
            print(f"Error saving final metrics: {e}")
    else:
        print("Training did not complete successfully. Skipping saving metrics.")

    print("\n--- Script Finished ---") 