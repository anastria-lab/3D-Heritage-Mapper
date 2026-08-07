from pathlib import Path
import subprocess
#Folder containing the "images" directory
PROJECT = Path(r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\3d\data\frames")

COLMAP_PATH = r"C:\Users\akara\OneDrive\Desktop\D_Chatz\3D-Heritage-Mapper\projects\3d\colmap-x64-windows-cuda\COLMAP.bat"

# Replace these with your OpenCV calibration values.
fx = 4356.64037
fy = 4353.77492
cx = 2896.09430
cy = 2167.37811

k1 = 0.224889166
k2 = -1.44274419
p1 = 0.000268994704
p2 = 0.000790904953

camera_parameters = f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}"



command = [
    COLMAP_PATH,
    "automatic_reconstructor",
    "--workspace_path", str(PROJECT),
    "--image_path", str(PROJECT),

    # The frames came from video
    "--data_type", "VIDEO",

    # Start with MEDIUM; change to HIGH later
    "--quality", "MEDIUM",

    # Robust reconstruction method
    "--mapper", "INCREMENTAL",

    # Create a mesh
    "--mesher", "POISSON",

    # # Every frame uses the same calibrated camera
    # "--single_camera", "1",  # <-- Διορθώθηκε
    # "--camera_model", "OPENCV",
    # "--camera_params", camera_parameters,
]

print("Starting reconstruction...")
subprocess.run(command, check=True)
print("Finished.")