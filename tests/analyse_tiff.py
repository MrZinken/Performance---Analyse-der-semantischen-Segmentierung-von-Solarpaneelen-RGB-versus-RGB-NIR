from PIL import Image
import numpy as np


def analyze_tiff_bit_depth(tiff_image_path):
    image = Image.open(tiff_image_path)

    # convert to NumPy-Array
    image_array = np.array(image)

    # Check for four channels
    if image_array.ndim == 3 and image_array.shape[2] == 4:
        print(f"Das Bild hat {image_array.shape[2]} Kanäle.")
    else:
        print("Das Bild hat nicht die erwartete Anzahl an Kanälen (4).")
        return

    # Check Bit-Depth
    bit_depth = image_array.dtype
    print(f"Das Bild hat eine Bit-Tiefe von: {bit_depth}")

    # Show channels
    r_channel = image_array[:, :, 0]
    g_channel = image_array[:, :, 1]
    b_channel = image_array[:, :, 2]
    nir_channel = image_array[:, :, 3]

    print(f"R-Kanal Min: {r_channel.min()}, Max: {r_channel.max()}")
    print(f"G-Kanal Min: {g_channel.min()}, Max: {g_channel.max()}")
    print(f"B-Kanal Min: {b_channel.min()}, Max: {b_channel.max()}")
    print(f"NIR-Kanal Min: {nir_channel.min()}, Max: {nir_channel.max()}")


# Path to TIFF-Image
tiff_image_path = "/home/kai/Documents/2slice/62752400.tif"

analyze_tiff_bit_depth(tiff_image_path)
