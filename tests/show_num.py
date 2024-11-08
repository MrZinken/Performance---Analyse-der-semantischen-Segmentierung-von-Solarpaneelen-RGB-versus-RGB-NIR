import numpy as np
import matplotlib.pyplot as plt
"""
show numpy files generated by slicing script
"""

def display_rgb_and_individual_bands_in_grayscale(npy_file_path):
    # Load the NumPy file
    npy_data = np.load(npy_file_path)  # Expecting a 4-channel array (RGB + NIR)

    # Check if the NumPy array has 4 channels (RGB + NIR)
    if npy_data.shape[2] != 4:
        raise ValueError(
            f"Expected a 4-channel array (RGB + NIR), but got {npy_data.shape[2]} channels."
        )

    # Extract the RGB and NIR channels
    rgb_image = npy_data[:, :, :3]  # First three channels are RGB
    red_channel = npy_data[:, :, 0]  # Red band
    green_channel = npy_data[:, :, 1]  # Green band
    blue_channel = npy_data[:, :, 2]  # Blue band
    nir_channel = npy_data[:, :, 3]  # NIR band

    # Display the RGB and individual bands in grayscale
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # Display the RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    # Display the Red band in grayscale
    axes[1].imshow(red_channel, cmap="gray")
    axes[1].set_title("Red Band (Grayscale)")
    axes[1].axis("off")

    # Display the Green band in grayscale
    axes[2].imshow(green_channel, cmap="gray")
    axes[2].set_title("Green Band (Grayscale)")
    axes[2].axis("off")

    # Display the Blue band in grayscale
    axes[3].imshow(blue_channel, cmap="gray")
    axes[3].set_title("Blue Band (Grayscale)")
    axes[3].axis("off")

    # Display the NIR band in grayscale
    axes[4].imshow(nir_channel, cmap="gray")
    axes[4].set_title("NIR Band (Grayscale)")
    axes[4].axis("off")

    # Show the plots
    plt.tight_layout()
    plt.show()


# Example usage
npy_file_path = "/home/kai/Desktop/dataset/train/61752300_slice_001_rgb_ir.npy"  # Replace with your file path
display_rgb_and_individual_bands_in_grayscale(npy_file_path)
