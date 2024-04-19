# import os
# import subprocess

# # # Crear las carpetas
# # os.makedirs("list_folder", exist_ok=True)
# # os.makedirs("right_camera/images", exist_ok=True)
# # os.makedirs("right_camera/masks", exist_ok=True)
# # os.makedirs("sparse", exist_ok=True)
# # os.makedirs("dense", exist_ok=True)

# # Directorio de trabajo
# working_dir = "/working"

# # Cambiar directorio de trabajo
# os.chdir(working_dir)

# # En este paso puedo ejecutar un ciclo for para cambiar los nombres de las carpetas y de los archivos para cada submodelo 
# # Ejecutar comandos COLMAP --Mapper.multiple_models 0
# # colmap model_converter --input_path sparse/0 --output_path sparse/0/sparseModel.ply --output_type PLY
# commands = [
#     "colmap feature_extractor --database_path dataset.db --image_path camera_images/images --ImageReader.mask_path camera_images/masks --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1",
#     "colmap sequential_matcher --database_path dataset.db",
#     "colmap mapper --database_path dataset.db --image_path camera_images/images --output_path sparse --image_list_path lists_folder/right/list0.txt --Mapper.multiple_models 0" ,
#     "colmap image_undistorter --image_path camera_images/images --input_path sparse/0 --output_path dense --output_type COLMAP",
#     "colmap patch_match_stereo --workspace_path dense",
#     "colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply"
# ]

# for cmd in commands:
#     subprocess.run(cmd, shell=True)


import os

def run_colmap_mapper(image_folder, list_folder, output_folder):
    # Obtener la lista de archivos de lista en el folder
    #list_files = [f for f in os.listdir(list_folder) if f.startswith("list")]
    list_files = sorted([f for f in os.listdir(list_folder) if f.startswith("list")])


    # Crear el folder de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Iterar sobre los archivos de lista
    for list_file in list_files:
        # Construir el nombre del archivo de lista completo
        list_path = os.path.join(list_folder, list_file)
        
        # Construir el nombre de la carpeta de salida
        list_number = list_file.split(".")[0]
        output_subfolder = list_number
        output_subfolder_path = os.path.join(output_folder, output_subfolder)

        # Crear la carpeta de salida para esta lista
        os.makedirs(output_subfolder_path, exist_ok=True)

        # Ejecutar colmap mapper con los parámetros especificados
        command = f"colmap mapper --database_path dataset.db --image_path {image_folder} --output_path {output_subfolder_path} --image_list_path {list_path}"
        os.system(command)
        output_model_path = os.path.join(output_subfolder_path, f"SparseModel{list_number}.ply")
        command_converter = f"colmap model_converter --input_path {output_subfolder_path}/0 --output_path {output_model_path} --output_type PLY"
        os.system(command_converter)
        #colmap model_merger --input_path1 sparse/012 --input_path2 sparse/list3/0 --output_path sparse/01234

# Definir las rutas de los directorios
image_folder = "camera_images/images"
list_folder = "lists_folder/right"
output_folder = "sparse"

# Ejecutar la función con las rutas especificadas
run_colmap_mapper(image_folder, list_folder, output_folder)

