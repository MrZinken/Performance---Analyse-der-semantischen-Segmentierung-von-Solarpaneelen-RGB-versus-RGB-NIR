import platform
import torch
import pkg_resources
import subprocess
import importlib

# Liste der spezifischen Bibliotheken, die du überprüfen möchtest
libraries = ["torch", "torchvision", "numpy", "scipy", "pandas", "matplotlib", "skimage", "cv2", "os", "json", "time", ] # Füge hier weitere Bibliotheken hinzu

# Öffne oder erstelle die Textdatei, in die die Versionen geschrieben werden sollen
with open("system_info.txt", "w") as file:
    # Betriebssystem-Informationen
    file.write("Systeminformationen\n")
    file.write(f"Linux Version: {platform.platform()}\n")
    
    # CUDA-Version
    try:
        cuda_version = torch.version.cuda
        file.write(f"CUDA Version: {cuda_version}\n")
    except AttributeError:
        file.write("CUDA nicht verfügbar\n")
    
    # Hardware-Informationen
    cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    file.write("\nHardwareinformationen:\n")
    file.write(cpu_info)
    
    # GPU-Informationen (falls verfügbar)
    gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    if gpu_info:
        file.write("\nGPU-Informationen:\n")
        file.write(gpu_info)
    else:
        file.write("GPU-Informationen nicht verfügbar oder keine NVIDIA-GPU erkannt\n")
    
    # Bibliotheksversionen
    file.write("\nBibliotheksversionen:\n")
    for library in libraries:
        try:
            # Für externe Bibliotheken wie skimage, cv2
            if importlib.util.find_spec(library) is not None:
                version = pkg_resources.get_distribution(library).version
                file.write(f"{library}=={version}\n")
            else:
                file.write(f"{library} ist nicht installiert\n")
        except pkg_resources.DistributionNotFound:
            file.write(f"{library} ist nicht installiert\n")

    # Informationen zu Standardbibliotheken
    file.write("\nStandardbibliotheken:\n")
    standard_libs = ["os", "json", "time"]
    for lib in standard_libs:
        file.write(f"{lib} ist in der Standardbibliothek vorhanden\n")

print("Die Systeminformationen wurden erfolgreich in 'system_info.txt' gespeichert.")
