import os
import subprocess

# # Crear las carpetas
# os.makedirs("list_folder", exist_ok=True)
# os.makedirs("right_camera/images", exist_ok=True)
# os.makedirs("right_camera/masks", exist_ok=True)
# os.makedirs("sparse", exist_ok=True)
# os.makedirs("dense", exist_ok=True)

# Directorio de trabajo
working_dir = "/working"

# Cambiar directorio de trabajo
os.chdir(working_dir)

# En este paso puedo ejecutar un ciclo for para cambiar los nombres de las carpetas y de los archivos para cada submodelo 
# Ejecutar comandos COLMAP
commands = [
    "colmap feature_extractor --database_path dataset.db --image_path camera_images/images --ImageReader.mask_path camera_images/masks --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1",
    "colmap sequential_matcher --database_path dataset.db",
    "colmap mapper --database_path database.db --image_path camera_images/images --output_path sparse --image_list_path list_folder/list0.txt"
#    "colmap image_undistorter --image_path camera_images/images --input_path sparse/0 --output_path dense --output_type COLMAP",
#    "colmap patch_match_stereo --workspace_path dense",
#    "colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply"
]

for cmd in commands:
    subprocess.run(cmd, shell=True)
