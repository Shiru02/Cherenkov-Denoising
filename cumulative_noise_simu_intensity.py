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
def noise_simulation_python():
    """Main noise simulation function with cumulative image generation at fixed illumination levels."""
    
    # ========== USER INPUTS ==========
    output_base_dir = 'output_cumulative_noise_simu'  # Base output directory
    input_image_path = r"Acc_image/Chr_I1.mat"  # .mat file path
    mat_variable_name = "I_patch"  # Variable name inside the .mat file

    # MODIFICATION 2: User choice for input image
    use_flat_field_image = False  # Set to False to use the .mat file

    # --- Parameters to try ---
    f_gains_to_try = [5000.0]
    blur_configs_to_try = [(2.0, 35)]
    snsr_noise_dn_levels_to_try = [30] # RMS read noise in DN

    # --- NEW: Fixed illumination levels to test ---
    illumination_levels = [0.001]  # Photon flux levels to test

    # --- Number of frames to cumulate ---
    cumulative_frame_counts = [1, 5, 8, 10, 20, 40, 50, 80, 100, 120, 150, 200]

    # --- Scaling factor for intensity-dependent noise ---
    noise_scaling_factors = [2.0]  # 0=uniform, 1.0=2x at max intensity, 2.0=3x at max intensity

    # --- General Simulation Settings ---
    numframes = max(cumulative_frame_counts)  # Generate enough frames for the largest cumulative count
    snsr_pixel_pitch = 5.86
    t_int = 76

    start_time = time.time()
    
    # ========== SETUP OUTPUT DIRECTORY ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base_dir, f'cumulative_images_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Created base output directory: {output_dir}')

    # ========== IMAGE LOADING LOGIC ==========
    snsr_resy, snsr_resx = 512, 512
    if use_flat_field_image == False and os.path.exists(input_image_path):
        # Check if it's a .mat file
        if input_image_path.lower().endswith('.mat'):
            print(f"Loading .mat file: {input_image_path}")
            try:
                base_image = load_and_prepare_mat_image(input_image_path, 
                                                        variable_name=mat_variable_name,
                                                        target_shape=(snsr_resy, snsr_resx))
            except Exception as e:
                print(f"FATAL: Could not load .mat file. Error: {e}")
                return None
        else:
            print(f"Loading user-provided image: {input_image_path}")
            try:
                base_image = load_and_prepare_image(input_image_path, target_shape=(snsr_resy, snsr_resx))
            except Exception as e:
                print(f"FATAL: Could not load image. Error: {e}")
                return None
    else:
        print(f"Input image path not found or not provided. Generating a {snsr_resy}x{snsr_resx} flat-field image.")
        base_image = np.ones((snsr_resy, snsr_resx), dtype=np.float64)
    
    save_image_as_16bit_png(base_image * 65535, os.path.join(output_dir, '_input_image.png'))

    # ========== MAIN SIMULATION LOOPS ==========
    total_sims = len(f_gains_to_try) * len(blur_configs_to_try) * len(snsr_noise_dn_levels_to_try) * len(illumination_levels) * len(noise_scaling_factors)
    sim_count = 0

    for f_gain in f_gains_to_try:
        for blur_params in blur_configs_to_try:
            for snsr_noise_dn in snsr_noise_dn_levels_to_try:
                for illum_level in illumination_levels:
                    for noise_scale_factor in noise_scaling_factors:
                        sim_count += 1
                        blur_sigma, blur_size = blur_params
                        print(f"\n--- Running Sim {sim_count}/{total_sims}: F Factor={f_gain}, Blur=(σ:{blur_sigma}, size:{blur_size}), Noise={snsr_noise_dn} DN, Illumination={illum_level}, Noise Scale={noise_scale_factor} ---")

                        run_dir_name = f'run_{sim_count}_FFactor_{f_gain}_blur_{blur_sigma}s_{blur_size}p_noise_{snsr_noise_dn}DN_illum_{illum_level}_scale_{noise_scale_factor}'
                        run_output_dir = os.path.join(output_dir, run_dir_name.replace('.', 'p'))
                        os.makedirs(run_output_dir, exist_ok=True)
                        
                        # Generate frames for the current illumination level
                        frame_stack = np.zeros((snsr_resy, snsr_resx, numframes))
                        
                        print(f"Generating {numframes} frames at illumination level {illum_level} with noise scale {noise_scale_factor}...")
                        for frame in tqdm(range(numframes), desc="Generating frames"):
                            mean_photons = base_image * illum_level * t_int * (snsr_pixel_pitch**2)
                            photoelectrons = np.random.poisson(mean_photons)
                            signal_after_gain = photoelectrons * f_gain
                            blurred_signal = apply_gaussian_blur_matlab_equivalent(signal_after_gain, sigma=blur_sigma, filter_size=blur_size)
                            
                            # --- INTENSITY-DEPENDENT NOISE SCALING (OPTION 1) ---
                            # Normalize intensity based on the blurred signal
                            max_signal = blurred_signal.max()
                            if max_signal > 0:
                                normalized_intensity = blurred_signal / max_signal
                            else:
                                normalized_intensity = np.zeros_like(blurred_signal)
                            
                            # Calculate noise scale: 1.0 at dark regions, (1 + noise_scale_factor) at bright regions
                            intensity_noise_scale = 1.0 + noise_scale_factor * normalized_intensity
                            
                            # Generate base read noise and scale it by intensity
                            base_read_noise = np.random.normal(0, snsr_noise_dn, (snsr_resy, snsr_resx))
                            scaled_read_noise = base_read_noise * intensity_noise_scale
                            
                            # Add scaled noise to signal
                            signal_with_noise = blurred_signal + scaled_read_noise
                            
                            # Final step is to digitize (floor or round)
                            final_signal = np.floor(signal_with_noise)
                            frame_stack[:, :, frame] = final_signal
                        
                        # Generate cumulative images
                        print(f"Saving cumulative images...")
                        for n_frames in cumulative_frame_counts:
                            if n_frames <= numframes:
                                cumulative_image = np.mean(frame_stack[:, :, :n_frames], axis=2)
                                output_filename = os.path.join(run_output_dir, f'cumulative_{n_frames}_frames.png')
                                save_image_as_16bit_png(cumulative_image, output_filename)
                                print(f"  Saved: cumulative_{n_frames}_frames.png")
            
    end_time = time.time()
    execution_time = end_time - start_time
    print('\n=== SIMULATION COMPLETE ===')
    print(f'Cumulative images saved in subdirectories inside: {output_dir}')
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