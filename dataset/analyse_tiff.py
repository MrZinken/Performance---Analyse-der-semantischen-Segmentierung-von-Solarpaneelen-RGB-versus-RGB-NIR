from PIL import Image
import numpy as np

def analyze_tiff_bit_depth(tiff_image_path):
    # TIFF Bild öffnen
    image = Image.open(tiff_image_path)

    # In ein NumPy-Array konvertieren
    image_array = np.array(image)

    # Überprüfen der Anzahl der Kanäle (RGB + NIR sollten 4 sein)
    if image_array.ndim == 3 and image_array.shape[2] == 4:
        print(f"Das Bild hat {image_array.shape[2]} Kanäle.")
    else:
        print("Das Bild hat nicht die erwartete Anzahl an Kanälen (4).")
        return

    # Bit-Tiefe ermitteln (Annahme: gleiche Bit-Tiefe für alle Kanäle)
    bit_depth = image_array.dtype

    # Ausgabe der Bit-Tiefe
    print(f"Das Bild hat eine Bit-Tiefe von: {bit_depth}")

    # Optional: Einzelne Kanäle (RGB + NIR) anzeigen
    r_channel = image_array[:, :, 0]
    g_channel = image_array[:, :, 1]
    b_channel = image_array[:, :, 2]
    nir_channel = image_array[:, :, 3]

    print(f"R-Kanal Min: {r_channel.min()}, Max: {r_channel.max()}")
    print(f"G-Kanal Min: {g_channel.min()}, Max: {g_channel.max()}")
    print(f"B-Kanal Min: {b_channel.min()}, Max: {b_channel.max()}")
    print(f"NIR-Kanal Min: {nir_channel.min()}, Max: {nir_channel.max()}")

# Pfad zur TIFF-Datei
tiff_image_path = '/media/kai/data/4channel_old/Testdaten_Kai_Mai/67502375.tif'

# Bild analysieren
analyze_tiff_bit_depth(tiff_image_path)
