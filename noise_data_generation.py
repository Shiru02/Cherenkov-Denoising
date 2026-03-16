import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import os
from datetime import datetime
import time
from tqdm import tqdm
import warnings
import mat73

# Ignore potential warnings for cleaner output
warnings.filterwarnings('ignore')

# --- HELPER FUNCTIONS ---

def load_and_prepare_mat_image(mat_path, variable_name='data', target_shape=(512, 512)):
    """Load and prepare a MATLAB .mat file containing image data."""
    try:
        mat_contents = mat73.loadmat(mat_path)
        
        if variable_name not in mat_contents:
            raise Exception(f"Variable '{variable_name}' not found in .mat file. Available variables: {list(mat_contents.keys())}")
        
        img_array = mat_contents[variable_name]
        print(f"Loaded .mat file. Original shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Convert to 2D if needed
        if img_array.ndim > 2:
            print(f"Array has {img_array.ndim} dimensions. Taking first 2D slice.")
            img_array = img_array[:, :, 0] if img_array.shape[2] == 1 else img_array[:, :]
        
        # Convert to float64
        img_array = img_array.astype(np.float64)
        
        # Resize if needed
        if img_array.shape != target_shape:
            print(f"Resizing image from {img_array.shape} to {target_shape}")
            img_pil = Image.fromarray(img_array)
            img_pil = img_pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
            img_array = np.array(img_pil, dtype=np.float64)
        
        # Normalize to [0, 1]
        min_val, max_val = img_array.min(), img_array.max()
        if max_val > min_val:
            img_normalized = (img_array - min_val) / (max_val - min_val)
        else:
            img_normalized = np.ones_like(img_array)
        
        print(f"Image loaded and normalized successfully. Final shape: {img_normalized.shape}")
        return img_normalized
        
    except Exception as e:
        raise Exception(f"Failed to load or prepare .mat file: {mat_path}\nError: {str(e)}")

def load_and_prepare_image(image_path, target_shape=(100, 100)):
    """Load, convert to grayscale, normalize, and resize an image for the simulation."""
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        if img.size != target_shape:
            print(f"Resizing image from {img.size} to {target_shape}")
            img = img.resize(target_shape, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float64)
        min_val, max_val = img_array.min(), img_array.max()
        if max_val > min_val:
            img_normalized = (img_array - min_val) / (max_val - min_val)
        else:
            img_normalized = np.ones_like(img_array) # For flat images
        print("Image loaded and normalized successfully.")
        return img_normalized
    except Exception as e:
        raise Exception(f"Failed to load or prepare image: {image_path}\nError: {str(e)}")

def apply_gaussian_blur_matlab_equivalent(image, sigma, filter_size):
    """Applies a Gaussian blur equivalent to MATLAB's imgaussfilt."""
    truncate_val = ((filter_size - 1) / 2) / sigma
    return gaussian_filter(image, sigma=sigma, truncate=truncate_val)

def save_image_as_16bit_png(image_array, filepath):
    """Normalizes a numpy array to the 16-bit range and saves it as a PNG."""
    image_array = image_array.astype(np.float64)
    min_val, max_val = image_array.min(), image_array.max()
    if max_val > min_val:
        normalized_array = 65535 * (image_array - min_val) / (max_val - min_val)
    else:
        normalized_array = np.zeros_like(image_array)
    img_16bit = normalized_array.astype(np.uint16)
    Image.fromarray(img_16bit).save(filepath)

# ==============================================================================
# =================== MAIN SIMULATION FUNCTION =============================
# ==============================================================================
def extract_patches(image, patch_size=128):
    """Extract non-overlapping patches from an image."""
    h, w = image.shape
    patches = []
    patch_positions = []
    
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            patch_positions.append((i, j))
    
    return patches, patch_positions

def noise_simulation_python():
    """Process all images in a folder: add noise, extract patches, cumulate frames, and save as .npy stacks."""
    
    # ========== USER INPUTS ==========
    input_folder = r"Acc_image"  # Folder containing .mat files
    output_base_dir = 'output_noisy_patches'  # Base output directory
    mat_variable_name = "I_patch"  # Variable name inside the .mat file

    # --- Fixed Parameters ---
    f_gain = 5000.0
    blur_sigma, blur_size = 2.0, 35
    snsr_noise_dn = 30  # RMS read noise in DN
    illumination_level = 0.001  # Fixed illumination level
    noise_scale_factor = 2.0  # Fixed noise scaling factor
    patch_size = 128  # Size of patches to extract
    
    # --- Cumulative frame counts ---
    cumulative_frame_counts = [1, 5, 8, 10, 20, 40, 50, 80, 100, 120, 150, 200]
    numframes = max(cumulative_frame_counts)  # Generate enough frames
    
    # --- General Simulation Settings ---
    snsr_resy, snsr_resx = 512, 512
    snsr_pixel_pitch = 5.86
    t_int = 76

    start_time = time.time()
    
    # ========== SETUP OUTPUT DIRECTORY ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base_dir, f'noisy_patches_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Created output directory: {output_dir}')
    
    # ========== FIND ALL .MAT FILES IN FOLDER ==========
    if not os.path.exists(input_folder):
        print(f"FATAL: Input folder does not exist: {input_folder}")
        return None
    
    mat_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.mat')]
    
    if len(mat_files) == 0:
        print(f"FATAL: No .mat files found in folder: {input_folder}")
        return None
    
    print(f"Found {len(mat_files)} .mat files to process")
    
    # ========== PROCESS EACH IMAGE ==========
    for idx, mat_filename in enumerate(mat_files, 1):
        mat_path = os.path.join(input_folder, mat_filename)
        print(f"\n--- Processing {idx}/{len(mat_files)}: {mat_filename} ---")
        
        # Load image
        try:
            base_image = load_and_prepare_mat_image(mat_path, 
                                                    variable_name=mat_variable_name,
                                                    target_shape=(snsr_resy, snsr_resx))
        except Exception as e:
            print(f"ERROR: Could not load {mat_filename}. Skipping. Error: {e}")
            continue
        
        # Generate frames with noise
        print(f"Generating {numframes} noisy frames with illumination={illumination_level}, scale={noise_scale_factor}...")
        frame_stack = np.zeros((snsr_resy, snsr_resx, numframes))
        
        for frame in tqdm(range(numframes), desc="Generating frames"):
            mean_photons = base_image * illumination_level * t_int * (snsr_pixel_pitch**2)
            photoelectrons = np.random.poisson(mean_photons)
            signal_after_gain = photoelectrons * f_gain
            blurred_signal = apply_gaussian_blur_matlab_equivalent(signal_after_gain, sigma=blur_sigma, filter_size=blur_size)
            
            # --- INTENSITY-DEPENDENT NOISE SCALING ---
            max_signal = blurred_signal.max()
            if max_signal > 0:
                normalized_intensity = blurred_signal / max_signal
            else:
                normalized_intensity = np.zeros_like(blurred_signal)
            
            intensity_noise_scale = 1.0 + noise_scale_factor * normalized_intensity
            base_read_noise = np.random.normal(0, snsr_noise_dn, (snsr_resy, snsr_resx))
            scaled_read_noise = base_read_noise * intensity_noise_scale
            signal_with_noise = blurred_signal + scaled_read_noise
            final_signal = np.floor(signal_with_noise)
            
            frame_stack[:, :, frame] = final_signal
        
        # Generate cumulative images for each cumulative level
        print(f"Creating cumulative image stack...")
        cumulative_images = []
        for n_frames in cumulative_frame_counts:
            if n_frames <= numframes:
                cumulative_image = np.mean(frame_stack[:, :, :n_frames], axis=2)
                cumulative_images.append(cumulative_image)
        
        # Stack cumulative images: shape will be (512, 512, num_cumulative_levels)
        cumulative_stack = np.stack(cumulative_images, axis=2)
        print(f"Cumulative stack shape: {cumulative_stack.shape}")
        
        # Extract patches from the cumulative stack
        print(f"Extracting {patch_size}x{patch_size} patches...")
        h, w, num_cumulative = cumulative_stack.shape
        patch_count = 0
        
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                # Extract patch across all cumulative levels
                patch_stack = cumulative_stack[i:i+patch_size, j:j+patch_size, :]
                
                # Save patch as .npy with shape (128, 128, num_cumulative_levels)
                base_name = os.path.splitext(mat_filename)[0]
                output_filename = f'{base_name}_patch_{i}_{j}.npy'
                output_path = os.path.join(output_dir, output_filename)
                np.save(output_path, patch_stack)
                patch_count += 1
        
        print(f"Extracted and saved {patch_count} patches from {mat_filename}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print('\n=== PROCESSING COMPLETE ===')
    print(f'Processed {len(mat_files)} images')
    print(f'Patch stacks saved in: {output_dir}')
    print(f'Each patch has shape: ({patch_size}, {patch_size}, {len(cumulative_frame_counts)})')
    print(f'Total execution time: {execution_time/60:.1f} minutes')
    
    return output_dir

# --- Main execution block ---
if __name__ == "__main__":
    try:
        output_path = noise_simulation_python()
        if output_path:
            print(f"\nSimulation completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred during the simulation: {str(e)}")
        import traceback
        traceback.print_exc()