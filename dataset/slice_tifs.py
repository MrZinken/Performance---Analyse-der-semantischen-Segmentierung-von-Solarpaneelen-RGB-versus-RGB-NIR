import os
import numpy as np
from PIL import Image
#slices the Tiff files and saves the slices as jpg and npy files
def slice_image_and_save(input_image_path, output_dir, slice_size=1000, max_slices=100):
    # Extract the base name of the input image without extension
    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    # Create subfolder for this image
    subfolder = os.path.join(output_dir, base_name)
    jpg_dir = os.path.join(subfolder, 'jpg_slices')
    rgb_ir_np_dir = os.path.join(subfolder, 'rgb_ir_numpy_slices')
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(rgb_ir_np_dir, exist_ok=True)

    # Load the image
    image = Image.open(input_image_path)
    image_array = np.array(image)

    # Check if the image has 4 channels (RGB + IR)
    if image_array.shape[2] != 4:
        raise ValueError(f"Image {input_image_path} does not have 4 channels (RGB + IR). Found: {image_array.shape[2]} channels.")

    # Extract RGB and IR channels
    rgb_image = image_array[:, :, :3]
    ir_channel = image_array[:, :, 3]

    # Initialize slice counter
    slice_counter = 0

    # Iterate over the image and create slices
    height, width = rgb_image.shape[:2]
    for y in range(0, height, slice_size):
        for x in range(0, width, slice_size):
            if slice_counter >= max_slices:
                break

            # Slice the RGB image
            rgb_slice = rgb_image[y:y + slice_size, x:x + slice_size]
            
            # Slice the IR channel and combine with RGB to form a 4-layer slice
            ir_slice = ir_channel[y:y + slice_size, x:x + slice_size]
            rgb_ir_slice = np.dstack((rgb_slice, ir_slice))
            
            # Generate slice filenames with zero-padded counter
            slice_filename = f"{base_name}_slice_{slice_counter:03}.jpg"
            slice_path = os.path.join(jpg_dir, slice_filename)
            Image.fromarray(rgb_slice).save(slice_path, 'JPEG')

            # Save the RGB + IR slice as a NumPy array
            rgb_ir_np_filename = f"{base_name}_slice_{slice_counter:03}_rgb_ir.npy"
            rgb_ir_np_path = os.path.join(rgb_ir_np_dir, rgb_ir_np_filename)
            np.save(rgb_ir_np_path, rgb_ir_slice)

            # Increment slice counter
            slice_counter += 1

    print(f"Processing complete for {input_image_path}. Slices saved to {subfolder}.")

def process_all_tiffs_in_folder(input_folder, output_dir, slice_size=1000, max_slices=100):
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            input_image_path = os.path.join(input_folder, filename)
            print(f"Processing {input_image_path}...")
            slice_image_and_save(input_image_path, output_dir, slice_size=slice_size, max_slices=max_slices)

# Example usage
input_folder = '/home/kai/Documents/2slice'
output_dir = '/home/kai/Documents/2slice/out'
process_all_tiffs_in_folder(input_folder, output_dir, slice_size=1000)
