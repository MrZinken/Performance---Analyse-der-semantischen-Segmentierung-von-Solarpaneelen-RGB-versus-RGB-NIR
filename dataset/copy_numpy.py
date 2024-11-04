import os
import shutil
import json
#Copies the npy files to the folder of dataset with the 
def find_and_copy_and_rename_numpy_arrays(annotation_file, numpy_dir, target_dir):
    # Load the annotation file to extract the relevant file names
    with open(annotation_file, "r") as f:
        annotation_data = json.load(f)

    # Extract file names from the annotations
    for image_info in annotation_data["images"]:
        original_name = (
            os.path.splitext(image_info["file_name"])[0] + "_rgb_ir.npy"
        )  # The original slice name with '_rgb_ir.npy'
        new_name = image_info[
            "file_name"
        ]  # The new name exactly as in the annotation file

        # Traverse the numpy directory and copy the matching files
        numpy_file_found = False
        for root, dirs, files in os.walk(numpy_dir):
            if original_name in files:
                source_path = os.path.join(root, original_name)
                destination_path = os.path.join(target_dir, new_name)

                # Ensure the target directory exists
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)

                # Copy and rename the file
                shutil.copy2(source_path, destination_path)
                print(f"Copied and renamed: {source_path} -> {destination_path}")
                numpy_file_found = True
                break

        # Report if the corresponding NumPy file is not found
        if not numpy_file_found:
            print(f"Warning: {original_name} from annotations not found in {numpy_dir}")


# Simplified function to copy and rename files for both training and validation datasets
def copy_and_rename_files_for_datasets(dataset_folder, slices_folder):
    # Define the paths for the training and validation annotation files
    training_annotation_file = os.path.join(
        dataset_folder, "train", "_annotations.coco.json"
    )
    validation_annotation_file = os.path.join(
        dataset_folder, "valid", "_annotations.coco.json"
    )
    test_annotation_file = os.path.join(
        dataset_folder, "test", "_annotations.coco.json"
    )

    # Define the target directories
    training_target_directory = os.path.join(dataset_folder, "train")
    validation_target_directory = os.path.join(dataset_folder, "valid")
    test_target_directory = os.path.join(dataset_folder, "test")

    print("\nCopying and renaming files for the training dataset...")
    find_and_copy_and_rename_numpy_arrays(
        training_annotation_file, slices_folder, training_target_directory
    )

    print("\nCopying and renaming files for the validation dataset...")
    find_and_copy_and_rename_numpy_arrays(
        validation_annotation_file, slices_folder, validation_target_directory
    )

    print("\nCopying and renaming files for the validation dataset...")
    find_and_copy_and_rename_numpy_arrays(
        test_annotation_file, slices_folder, test_target_directory
    )


# Example usage
dataset_folder = "/home/kai/Desktop/Downloads/dataset_75"  # Path to the dataset folder containing 'train' and 'valid'
slices_folder = (
    "/media/kai/data/slices"  # Path to the folder containing the NumPy arrays
)

copy_and_rename_files_for_datasets(dataset_folder, slices_folder)
