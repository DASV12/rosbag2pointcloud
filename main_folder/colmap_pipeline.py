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
# colmap feature_extractor --database_path /working/colmap_ws/dataset.db --image_path /working/dataset_ws/images --ImageReader.mask_path /working/dataset_ws/masks --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera_per_folder 1 --ImageReader.camera_params 765.9,640.0,360.0



# import os
# import subprocess

# def run_colmap_mapper(image_folder, list_folder, output_folder):
#     os.makedirs("colmap_ws", exist_ok=True)
#     # Run feature extractor and feature matching
#     commands = [
#      "colmap feature_extractor --database_path /working/colmap_ws/dataset.db --image_path /working/dataset_ws/images --ImageReader.mask_path /working/dataset_ws/masks --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera_per_folder 1",
#      "colmap sequential_matcher --database_path /working/colmap_ws/dataset.db"
#     ]
#     for cmd in commands:
#      subprocess.run(cmd, shell=True)

#     # Get the list of subfolder names in the image folder
#     image_subfolders = sorted([name for name in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, name))])

#     # Get the list of subfolder names in the list folder
#     list_subfolders = sorted([name for name in os.listdir(list_folder) if os.path.isdir(os.path.join(list_folder, name))])

#     # Create output folders for each subfolder in the image and list folders
#     for image_subfolder, list_subfolder in zip(image_subfolders, list_subfolders):
#         # Build the full paths for image and list subfolders
#         image_subfolder_path = os.path.join(image_folder, image_subfolder)
#         list_subfolder_path = os.path.join(list_folder, list_subfolder)

#         # Build the output subfolder path in the sparse directory
#         output_subfolder_path = os.path.join(output_folder, image_subfolder)

#         # Create the output subfolder if it doesn't exist
#         os.makedirs(output_subfolder_path, exist_ok=True)

#         # Get the list of list files in the current list subfolder
#         list_files = sorted([f for f in os.listdir(list_subfolder_path) if f.startswith("list")])

#         # Iterate over the list files and run COLMAP mapper for each
#         for list_file in list_files:
#             # Construct the full path for the list file
#             list_path = os.path.join(list_subfolder_path, list_file)

#             # Execute COLMAP mapper command with the specified parameters
#             command = f"colmap mapper --database_path /working/colmap_ws/dataset.db --image_path {image_folder} --output_path {output_subfolder_path} --image_list_path {list_path} --Mapper.multiple_models 0"
#             os.system(command)
            
#             # Define the output model path
#             output_model_path = os.path.join(output_subfolder_path, f"SparseModel{list_file.split('.')[0]}.ply")

#             # Convert the model to PLY format
#             command_converter = f"colmap model_converter --input_path {output_subfolder_path}/0 --output_path {output_model_path} --output_type PLY"
#             #os.system(command_converter)

# # Define the paths of the directories
# image_folder = "/working/colmap_ws/images"
# list_folder = "/working/colmap_ws/lists_folder"
# output_folder = "/working/colmap_ws/sparse"

# # Execute the function with the specified paths
# run_colmap_mapper(image_folder, list_folder, output_folder)



import yaml
import os
import subprocess
import shutil

def leer_configuracion(archivo):
    with open(archivo, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_colmap_point_triangulator(database_path, image_path, input_path, output_path):
    command = [
        "colmap", "point_triangulator",
        "--database_path", database_path,
        "--image_path", image_path,
        "--input_path", input_path,
        "--output_path", output_path
    ]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Point triangulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Point triangulation failed.")

def run_colmap_bundle_adjuster(input_path, output_path):
    command = [
        "colmap", "bundle_adjuster",
        "--input_path", input_path,
        "--output_path", output_path
    ]
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Bundle adjustment completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Bundle adjustment failed.")

def PT_reconstruction(config):
    print("PT reconstruction")
    # if os.path.exists(os.path.expanduser(config["output_dir"])):
    #         shutil.rmtree(os.path.expanduser(config["output_dir"]))
    os.makedirs(os.path.expanduser(config["output_dir"]), exist_ok=True)
    database_path = os.path.join(os.path.expanduser(config["output_dir"]), "database.db")
    if os.path.exists(database_path):
        os.remove(database_path)
    image_path = os.path.join(os.path.expanduser(config["dataset_path"]), "images")
    mask_path = os.path.join(os.path.expanduser(config["dataset_path"]), "masks")
    list_path = os.path.join(os.path.expanduser(config["dataset_path"]), "lists")
    camera_model = "SIMPLE_PINHOLE"
    single_camera_per_folder = str(1) 
    input_path = os.path.join(os.path.expanduser(config["output_dir"]), "sparse")
    output_path = input_path

    for camera in config["cameras"]:
        mask_flag = False
        ##
        # image_path = os.path.join(os.path.expanduser(os.path.join(config["dataset_path"], camera)), "images")
        # mask_path = os.path.join(os.path.expanduser(os.path.join(config["dataset_path"], camera)), "masks")
        # list_path = os.path.join(os.path.expanduser(os.path.join(config["dataset_path"], camera)), "lists")
        # print(image_path)
        # print(mask_path)
        # print(list_path)
        ##
        #comprobar que si exitan las imagenes
        image_path_camera = os.path.join(image_path, camera)
        if not os.path.exists(image_path):
            raise ValueError(f"Images Not Fund for: {camera}")
        else:
            print("Images found for:", camera)
        # verificar si hay masks para agregarlas al comando de colmap
        mask_path_camera = os.path.join(mask_path, camera)
        if not os.path.exists(mask_path):
            print("No masks found for:", camera)
        else:
            print("Masks found for:", camera)
            mask_flag = True
        # leer camera params
        list_path_camera = os.path.join(list_path, camera)
        cameras_file = os.path.join(list_path_camera, "cameras.txt")
        with open(cameras_file, 'r') as f:
            line = f.readline()
            camera_description = line.split()
            # Obtener los elementos de las posiciones 5, 6 y 7 (índices 4, 5 y 6 en Python)
            f = camera_description[4]
            cx = camera_description[5]
            cy = camera_description[6]
            # Concatenar los elementos separados por comas y agregarlos a la lista
            camera_params = ','.join([f, cx, cy])
        print("Camera params:", camera_params)
        # leer lista de imagenes
        image_list_path = os.path.join(list_path_camera, "image_list.txt")
        # ejecutar feature extractor
        command = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera_per_folder", single_camera_per_folder,
        "--image_list_path", image_list_path,
        "--ImageReader.camera_params", camera_params
        ]
        if mask_flag:
            # Agregar un nuevo comando
            new_command = [
                "--ImageReader.mask_path", mask_path
            ]
            # Extender la lista de comandos original con el nuevo comando
            command.extend(new_command)
        # print("Lista de comandos:", command)
        # input("Wait")
        try:
            # Run the command
            subprocess.run(command, check=True)
            print("Feature extractor completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print("Feature extractor failed.")

    #colmap sequential_matcher --database_path dataset.db (--SiftExtraction.use_gpu 0)
    use_gpu = int(bool(config["use_gpu"]))
    matcher = config["matcher"]
    command = [
        "colmap",  # Comando base
        "spatial_matcher",  # Valor predeterminado
        "--database_path", database_path,
        "--SiftMatching.use_gpu", str(use_gpu)
    ]

    # Switch para determinar el comando en función del valor de matcher
    if matcher == "exhaustive":
        command[1] = "exhaustive_matcher"
    elif matcher == "sequential":
        command[1] = "sequential_matcher"
    elif matcher == "spatial":
        command[1] = "spatial_matcher"
        new_command = [
                "--SpatialMatching.is_gps", str(0)
            ]
        command.extend(new_command)
    else:
        raise ValueError(f"No valid matcher, only spatial, sequential or exhaustive.")

    # print("Using ", command[1])
    # print(command)
    # input("wait")
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Matcher completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Matcher failed.")
    
    # generar empty model en /sparse con las camaras requeridas y la image_list.txt de cada camara
    contador_camara = 1
    contador = 1
    output_file_path = os.path.expanduser(os.path.join(config["output_dir"], "sparse"))
    if os.path.exists(output_file_path):
        shutil.rmtree(output_file_path)
    os.makedirs(output_file_path, exist_ok=True)
    cameras_output_file_path = os.path.join(output_file_path, "cameras.txt")
    points3D_output_file_path = os.path.join(output_file_path, "points3D.txt")
    output_file_path = os.path.join(output_file_path, "images.txt")
    with open(points3D_output_file_path, 'w'):
        pass
    #input("wait")
    for camera in config["cameras"]:
        list_path = os.path.join(os.path.expanduser(os.path.join(config["dataset_path"], "lists")), camera)
        cameras_file_path = os.path.join(list_path, "cameras.txt")
        image_list_file_path = os.path.join(list_path, "image_list.txt")
        images_file_path = os.path.join(list_path, "images.txt")
    
        #with open(cameras_file_path, 'r') as cameras_file:
        with open(cameras_file_path, 'r') as origen, open(cameras_output_file_path, 'a') as destino:
            # Lee la primera línea del archivo de origen
            primera_linea = origen.readline()
            elements = primera_linea.split()
            elements[0] = str(contador_camara)
            output_line = ' '.join(elements)
            # Escribe la primera línea en el archivo de destino
            destino.write(output_line + "\n")


        # Abre los archivos de entrada y salida
        with open(image_list_file_path, 'r') as image_list_file, \
                open(images_file_path, 'r') as images_file, \
                open(output_file_path, 'a') as output_file:

            # Resto del código permanece igual
            for image_list_line in image_list_file:
                image_filename = image_list_line.strip()
                # print(image_filename)
                # input("wait")
                images_file.seek(0)
                found_match = False
                for images_line in images_file:
                    elements = images_line.split()
                    if elements:
                        images_filename = elements[-1]
                        # print(images_filename)
                        # input("wait")
                        if image_filename == images_filename:
                            # Reemplazar el primer elemento por el contador
                            elements[0] = str(contador)
                            elements[8] = str(contador_camara)
                            # print(elements)
                            # input("wait")
                            output_line = ' '.join(elements)
                            # print(output_line)
                            # input("wait")
                            output_file.write(output_line + "\n\n")
                            contador += 1
                            found_match = True
                            break
                if not found_match:
                    print(f"No se encontró una coincidencia para el archivo: {image_filename}" )
        contador_camara += 1
        # print("camara: ", contador_camara-1)
        #input("wait")
    ###
    #llamar funcion de generar archivo legible de poses de imagenes del modelo


    num_iterations = config["PT_cycle"]
    # ejecutar los ciclos de PT y BA sobre la misma carpeta /sparse
    for i in range(num_iterations):
        print(f"Iteration {i+1}:")
        run_colmap_point_triangulator(database_path, image_path, input_path, output_path)
        run_colmap_bundle_adjuster(input_path, output_path)
        print("-" * 50)

    # convertir el modelo a .PLY
    #colmap model_converter --input_path sparse/0 --output_path sparse/0/sparseModel.ply --output_type PLY
    output_ply = os.path.join(output_path, "sparseModelPT.ply")
    command = [
        "colmap", "model_converter",  
        "--input_path", input_path,
        "--output_path", output_ply,
        "--output_type", "PLY"
    ]
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Converter completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Converter failed.")
    ###


    # Extract final poses
    SMReader_path = os.path.expanduser("/working/main_folder/SparseModelReader.py")
    output_poses = os.path.expanduser(os.path.join(input_path, "final_model"))
    # print(SMReader_path)
    # print(output_poses)
    # print(input_path)
    # input("wait")
    command = [
        "python3", SMReader_path,
        "--input_model", input_path,
        "--output_model", output_poses,
        "--input_format", ".bin"
    ]

    try:
        # Ejecutar el comando
        subprocess.run(command, check=True)
        print("SparseModelReader completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("SparseModelReader failed.")






def SfM_reconstruction(config):
    print("SfM reconstruction")

def dense_reconstruction(config):
    print("Dense reconstruction")
    image_path = os.path.join(os.path.expanduser(config["dataset_path"]), "images")
    input_path = os.path.expanduser(os.path.join(config["output_dir"], "sparse"))
    output_path = os.path.expanduser(os.path.join(config["output_dir"], "dense"))
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    dense_file = os.path.join(output_path, "fused.ply")
    mesh_poisson_file = os.path.join(output_path, "meshed-poisson.ply")
    mesh_delaunay_file = os.path.join(output_path, "meshed-delaunay.ply")
    # print(image_path)
    # print(input_path)
    # print(output_path)
    # print(dense_file)
    # print(mesh_poisson_file)
    # print(mesh_delaunay_file)
    # input("wait")
    command = [
        "colmap", "image_undistorter",
        "--image_path", image_path,
        "--input_path", input_path,
        "--output_path", output_path,
        "--output_type", "COLMAP"
    ]
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Undistorter completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Undistorter failed.")
    
    command = [
        "colmap", "patch_match_stereo",
        "--workspace_path", output_path
    ]
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Patch match stereo completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Patch match stereo failed.")

    command = [
        "colmap", "stereo_fusion",
        "--workspace_path", output_path,
        "--output_path", dense_file
    ]
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Stereo fusion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Stereo fusion failed.")

    command = [
        "colmap", "poisson_mesher",
        "--input_path", dense_file,
        "--output_path", mesh_poisson_file
    ]
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Poisson mesher completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Poisson mesher failed.")

    command = [
        "colmap", "delaunay_mesher",
        "--input_path", output_path,
        "--output_path", mesh_delaunay_file
    ]
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Delaunay mesher completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Delaunay mesher failed.")

    
# dataset_path: "/working/dataset_ws/rightGPS_right_GPS_timefilters"
# output_dir: "working/colmap_sw/poblado2camarasRight" #is taken as output dir for sparse reconstruction and input dir for dense reconstruction
# pose_type: "tf" # "tf": known pose reconstruction, "GPS": SfM pose estimator reconstruction
# cameras: [right_time0, right_time1] # cameras to reconstruct under dataset_path
# matcher: "spatial" # "exhaustive", "sequential". Sequential only for 1 camera because it would match last camera1 image with first camera2 image
# GPU: True
# PT_cycle: 10 # point refinement cycles for known pose reconstructions
# reconstruction: "sparse" # "dense"
# segmentation_flag: 0
# GPS_scalation_type: = "ENU" 

def main():
    #archivo_configuracion = input("Please enter the path to the YAML file: ")
    archivo_configuracion = "main_folder/config_colmap.yaml"
    config = leer_configuracion(archivo_configuracion)
    pose_type = config["pose_type"]
    reconstruction = config["reconstruction"]
    sparse_flag = False
    dense_flag = False
    if reconstruction == "sparse":
        sparse_flag = True
    if reconstruction == "dense":
        dense_flag = True
    if reconstruction == "both":
        sparse_flag = True
        dense_flag = True

    if sparse_flag:
        if pose_type == "tf":
            PT_reconstruction(config)
        elif pose_type == "GPS":
            SfM_reconstruction(config)

    if dense_flag:
        dense_reconstruction(config)
    
if __name__ == "__main__":
    main()


