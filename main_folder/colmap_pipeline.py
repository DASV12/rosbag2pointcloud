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


# import os

# def run_colmap_mapper(image_folder, list_folder, output_folder):
#     # Obtener la lista de archivos de lista en el folder
#     #list_files = [f for f in os.listdir(list_folder) if f.startswith("list")]
#     list_files = sorted([f for f in os.listdir(list_folder) if f.startswith("list")])


#     # Crear el folder de salida si no existe
#     os.makedirs(output_folder, exist_ok=True)

#     # Iterar sobre los archivos de lista
#     for list_file in list_files:
#         # Construir el nombre del archivo de lista completo
#         list_path = os.path.join(list_folder, list_file)
        
#         # Construir el nombre de la carpeta de salida
#         list_number = list_file.split(".")[0]
#         output_subfolder = list_number
#         output_subfolder_path = os.path.join(output_folder, output_subfolder)

#         # Crear la carpeta de salida para esta lista
#         os.makedirs(output_subfolder_path, exist_ok=True)

#         # Ejecutar colmap mapper con los parámetros especificados
#         command = f"colmap mapper --database_path dataset.db --image_path {image_folder} --output_path {output_subfolder_path} --image_list_path {list_path}"
#         os.system(command)
#         output_model_path = os.path.join(output_subfolder_path, f"SparseModel{list_number}.ply")
#         command_converter = f"colmap model_converter --input_path {output_subfolder_path}/0 --output_path {output_model_path} --output_type PLY"
#         os.system(command_converter)
#         #colmap model_merger --input_path1 sparse/012 --input_path2 sparse/list3/0 --output_path sparse/01234

# # Definir las rutas de los directorios
# image_folder = "colmap_ws/images"
# list_folder = "colmap_ws/lists_folder"
# output_folder = "colmap_ws/sparse"

# # Ejecutar la función con las rutas especificadas
# run_colmap_mapper(image_folder, list_folder, output_folder)

# colmap mapper --database_path /working/colmap_ws/dataset.db --image_path /working/colmap_ws/images --output_path /working/colmap_ws/sparse/right/complete --Mapper.multiple_models 0
# colmap model_converter --input_path /working/colmap_ws/sparse/right/01234/0 --output_path /working/colmap_ws/sparse/right/01234/sparseModel01234.ply --output_type PLY
# colmap mapper --database_path /working/colmap_ws/dataset.db --image_path /working/colmap_ws/images --output_path /working/colmap_ws/sparse/right/01234 --Mapper.multiple_models 0 --image_list_path /working/colmap_ws/lists_folder/right/list.txt
# colmap model_aligner --input_path /working/colmap_ws/sparse/right/012/0 --output_path /working/colmap_ws/sparse/right/012/georeferenced --database_path /working/colmap_ws/dataset.db --ref_is_gps 0 --alignment_type plane --merge_image_and_ref_origins 1
# olmap model_merger --input_path1 sparse/012 --input_path2 sparse/list3/0 --output_path sparse/01234
# colmap point_triangulator --database_path colmap_ws/dataset.db  --image_path colmap_ws/images --input_path colmap_ws/triangulator/ --output_path colmap_ws/sparse
# colmap point_triangulator --database_path colmap_ws/dataset.db  --image_path colmap_ws/images --input_path colmap_ws/sparse/it1/bundle/ --output_path colmap_ws/sparse/it2 
# colmap bundle_adjuster --input_path colmap_ws/sparse/it1/bundle/ --output_path 

import os
import subprocess

def run_colmap_mapper(image_folder, list_folder, output_folder):
    os.makedirs("colmap_ws", exist_ok=True)
    # Run feature extractor and feature matching
    commands = [
     "colmap feature_extractor --database_path /working/colmap_ws/dataset.db --image_path /working/dataset_ws/images --ImageReader.mask_path /working/dataset_ws/masks --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera_per_folder 1",
     "colmap sequential_matcher --database_path /working/colmap_ws/dataset.db"
    ]
    for cmd in commands:
     subprocess.run(cmd, shell=True)

    # Get the list of subfolder names in the image folder
    image_subfolders = sorted([name for name in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, name))])

    # Get the list of subfolder names in the list folder
    list_subfolders = sorted([name for name in os.listdir(list_folder) if os.path.isdir(os.path.join(list_folder, name))])

    # Create output folders for each subfolder in the image and list folders
    for image_subfolder, list_subfolder in zip(image_subfolders, list_subfolders):
        # Build the full paths for image and list subfolders
        image_subfolder_path = os.path.join(image_folder, image_subfolder)
        list_subfolder_path = os.path.join(list_folder, list_subfolder)

        # Build the output subfolder path in the sparse directory
        output_subfolder_path = os.path.join(output_folder, image_subfolder)

        # Create the output subfolder if it doesn't exist
        os.makedirs(output_subfolder_path, exist_ok=True)

        # Get the list of list files in the current list subfolder
        list_files = sorted([f for f in os.listdir(list_subfolder_path) if f.startswith("list")])

        # Iterate over the list files and run COLMAP mapper for each
        for list_file in list_files:
            # Construct the full path for the list file
            list_path = os.path.join(list_subfolder_path, list_file)

            # Execute COLMAP mapper command with the specified parameters
            command = f"colmap mapper --database_path /working/colmap_ws/dataset.db --image_path {image_folder} --output_path {output_subfolder_path} --image_list_path {list_path} --Mapper.multiple_models 0"
            os.system(command)
            
            # Define the output model path
            output_model_path = os.path.join(output_subfolder_path, f"SparseModel{list_file.split('.')[0]}.ply")

            # Convert the model to PLY format
            command_converter = f"colmap model_converter --input_path {output_subfolder_path}/0 --output_path {output_model_path} --output_type PLY"
            #os.system(command_converter)

# Define the paths of the directories
image_folder = "/working/colmap_ws/images"
list_folder = "/working/colmap_ws/lists_folder"
output_folder = "/working/colmap_ws/sparse"

# Execute the function with the specified paths
run_colmap_mapper(image_folder, list_folder, output_folder)

