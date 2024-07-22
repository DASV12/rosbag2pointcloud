## COLMAP pipeline
Reconstructing all images from all cameras at once will cause COLMAP to crash, first because images from different cameras will not have enough visual overlap, and second because a reconstruction from too many images can have large error drift and not give a detailed model or crash. Also, due to the structure of COLMAP, reconstruction by subsets is even faster than reconstructing all cameras at once.
To deal with this, the next pipeline is proposed:
1. Take only 1 camera folder at a time
2. Run feature_extractor on all images in the folder and their respective masks
3. Run sequential_matcher on all the images in the folder
4. Run mapper on subsets of images defined by image_path_file_"subset".txt until run all subsets
5. Run model_merger on all submodels generated in the last step
6. Run point_triangulator and bundle_adjuster to correct alignment errors between models
7. Run model_aligner to scale and align the model with the position of each image
8. Run a point cloud densification process form the sparse model generated in the last step
9. Run a mesh generation process from dense point clouf generated in last step
10. Repeat the process for all the cameras
All COLMAP CLI commands could be found at main_folder.
### COLMAP CLI
1. Crear carpetas, los archivos no mencionados acá se pueden dejar en otro directorio diferente.
Por ahora solo camara derecha en folder "right_camera" crear:
	list.txt -> archivo de texto con lista de las 100 primeras imagenes de la camara derecha
	camera_images
		images -> todas las imagenes de solo la camara derecha
		masks -> todas las mascaras de solo la camara derecha
	sparse -> carpeta donde se creará el sparse pointcloud
	dense -> carpeta donde se creará el dense pointcloud
		
2. Desde la carpeta "camera_images" ejecutar:
	colmap feature_extractor --database_path dataset.db --image_path images --ImageReader.mask_path masks --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1
Si la GPU si es reconocida por el container este comando debería funcionar. Si sale error intentar agregar: --SiftExtraction.use_gpu 0
colmap feature_extractor --database_path /working/colmap_ws/dataset.db --image_path /working/dataset_ws/images --ImageReader.mask_path /working/dataset_ws/masks --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera_per_folder 1 --ImageReader.camera_params 765.9,640.0,360.0 
3. colmap sequential_matcher --database_path dataset.db (--SiftExtraction.use_gpu 0)

4. colmap mapper --database_path dataset.db --image_path images --output_path sparse --image_list_path list.txt

5. Opcional para ver PLY del sparse model: colmap model_converter --input_path sparse/0 --output_path sparse/0/sparseModel.ply --output_type PLY
Antes de ejecutar el comando revisar si en la carpeta sparse se creó la carpeta "0" o si los archivos "images, cameras, points3D" se crearon directamente en "sparse". En ese caso borrar "/0" de la linea de comandos.

6. colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense --output_type COLMAP

7. colmap patch_match_stereo --workspace_path dense

8. colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply

9. colmap poisson_mesher --input_path dense/fused.ply --output_path dense/meshed-poisson.ply

9. colmap delaunay_mesher --input_path dense --output_path dense/meshed-delaunay.ply



sudo apt-get remove libxcb-xinerama0
sudo apt-get purge libxcb-xinerama0
sudo apt-get install libxcb-xinerama0
sudo apt install qtchooser
sudo apt install libqt5gui5