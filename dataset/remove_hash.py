import os
import json

def remove_hash_and_change_extension(filename):
    """
    Remove hash from the filename and change the extension from .jpg to .npy.
    """
    # Split the filename by '_rf.' to remove the hash
    parts = filename.split('_jpg.rf.')
    if len(parts) == 2:
        # Change extension from .jpg to .npy
        return parts[0] + '.npy'
    return filename  # Return the original if the format doesn't match

def rename_files_and_update_annotations(images_dir):
    # Traverse all subdirectories
    for root, dirs, files in os.walk(images_dir):
        # Process each JSON file in the directory
        for file in files:
            if file.endswith('.json'):
                annotations_file = os.path.join(root, file)

                # Load the annotations
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Iterate over each image in the annotations
                for image_info in data['images']:
                    original_name = image_info['file_name']
                    new_name = remove_hash_and_change_extension(original_name)

                    # Rename the file in the filesystem
                    old_file_path = os.path.join(root, original_name)
                    new_file_path = os.path.join(root, new_name)

                    if os.path.exists(old_file_path):
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {original_name} -> {new_name}")
                    else:
                        print(f"File not found: {old_file_path}")

                    # Update the annotation JSON with the new filename
                    image_info['file_name'] = new_name

                # Save the updated annotations back to the file
                with open(annotations_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)

                print(f"Updated annotations saved to {annotations_file}")

# Usage example
images_directory = '/home/kai/Downloads/dataset'

rename_files_and_update_annotations(images_directory)





""" import os
import json

def remove_hash_from_filename(filename):
    # Split the filename by '_rf.' to remove the hash
    parts = filename.split('_jpg.rf.')
    if len(parts) == 2:
        return parts[0] + '.jpg'
    return filename  # Return the original if the format doesn't match

def rename_files_and_update_annotations(images_dir):
    # Traverse all subdirectories
    for root, dirs, files in os.walk(images_dir):
        # Process each JSON file in the directory
        for file in files:
            if file.endswith('.json'):
                annotations_file = os.path.join(root, file)

                # Load the annotations
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Iterate over each image in the annotations
                for image_info in data['images']:
                    original_name = image_info['file_name']
                    new_name = remove_hash_from_filename(original_name)

                    # Rename the file in the filesystem
                    old_file_path = os.path.join(root, original_name)
                    new_file_path = os.path.join(root, new_name)

                    if os.path.exists(old_file_path):
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {original_name} -> {new_name}")
                    else:
                        print(f"File not found: {old_file_path}")

                    # Update the annotation JSON with the new filename
                    image_info['file_name'] = new_name

                # Save the updated annotations back to the file
                with open(annotations_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)

                print(f"Updated annotations saved to {annotations_file}")

# Usage example
images_directory = '/home/kai/Desktop/dataset_original'

rename_files_and_update_annotations(images_directory) """
