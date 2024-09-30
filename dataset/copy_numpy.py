import os
import shutil

def find_and_copy_numpy_arrays(image_dir, numpy_dir):
    # Traverse the dataset directories (train, valid, test)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                # Extract the base name from the jpg file (e.g., "67502350_slice_023")
                base_name = os.path.splitext(file)[0]

                # Define the expected names of the numpy arrays
                rgb_ir_npy = base_name + '_rgb_ir.npy'
                
                # Search for the numpy arrays in the numpy_dir
                numpy_rgb_ir_found = False

        

                for np_root, np_dirs, np_files in os.walk(numpy_dir):
                    if rgb_ir_npy in np_files:
                        numpy_rgb_ir_found = True
                        source_rgb_ir = os.path.join(np_root, rgb_ir_npy)
                        destination_rgb_ir = os.path.join(root, rgb_ir_npy)
                        shutil.copy2(source_rgb_ir, destination_rgb_ir)
                        print(f"Copied: {source_rgb_ir} -> {destination_rgb_ir}")


                # Report if matching numpy arrays are not found
                if not numpy_rgb_ir_found:
                    print(f"Warning: RGB-IR numpy array not found for image {file}")

# Example usage
image_directory = '/home/kai/Desktop/dataset_original'  # Path to dataset folder containing 'train', 'test', 'valid'
numpy_directory = '/media/kai/data/slices'  # Path to folder containing the numpy arrays

find_and_copy_numpy_arrays(image_directory, numpy_directory)
