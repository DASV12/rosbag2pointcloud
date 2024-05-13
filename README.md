																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																													# SfM
The goal of this project is to generate a suitable dataset of images with the necessary information from a ROS2 rosbag and then run a command pipeline with the COLMAP tool to generate a point cloud of the scene. All this executed form a docker container.

## Dataset Generation
### Images
Start specifying only the camera's topic to reconstruct at the bottom of the script and then run the script. The code will stract automatically intrinsics for each camera to undistort the images and syncronize them with their aproximated position taken from /tf.
The output of this script is "images" folder with subfolders for each camera: 
->"front" folder for topic: /camera/color/image_raw
->"left" folder for topic:/video_mapping/left/image_raw
->"right" folder for topic:/video_mapping/right/image_raw
### Masks
Moving objects lead to failures in the COLMAP pipeline. COLMAP integrates an option to filter areas of the images with masks. This masks are black & withe copies of the image where black pixels will not be readed.
Currently this script takes images from a folder, masks possible moving objects (people, cars, trucks and motorcycles) and creates a new folder with masks. To be read by COLMAP, a "masks" folder should be created at the same level and with the same subfolders as "images" folder.
### Future work
Merge image and mask generation in 1 script and automate mask folder creation.
Create image_path_file_"subset".txt files for further COLMAP processes. These files will contain a list of image names, with the folder divided into subsets of 100-150 images overlaped in 10 images.

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

3. colmap sequential_matcher --database_path dataset.db (--SiftExtraction.use_gpu 0)

4. colmap mapper --database_path database.db --image_path images --output_path sparse --image_list_path list.txt

5. Opcional para ver PLY del sparse model: colmap model_converter --input_path sparse/0 --output_path sparse/0/sparseModel.ply --output_type PLY
Antes de ejecutar el comando revisar si en la carpeta sparse se creó la carpeta "0" o si los archivos "images, cameras, points3D" se crearon directamente en "sparse". En ese caso borrar "/0" de la linea de comandos.

6. colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense --output_type COLMAP

7. colmap patch_match_stereo --workspace_path dense

8. colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply

9. colmap poisson_mesher --input_path dense/fused.ply --output_path dense/meshed-poisson.ply

9. colmap delaunay_mesher --input_path dense --output_path dense/meshed-delaunay.ply
### Notes
Feature extraction and sequential matching can be accelerated using GPU, and cloud densification and mesh generation can only be performed using GPU.
An Ubuntu-ROS2 container can only run COLMAP without GPU.
A pure colmap/colmap:latest image can´t run Dataset Generation stage
This project should be run in an nvidia/cuda image as suggested in the COLMAP repository. This option is still being tested due to issues with the version of the GPU used in this project.

## Run Project
### Requirements
Docker:
sudo apt update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

Cuda toolkit:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
-# Add to .bashrc
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
-# verify: nvcc --version


Nvidia-docker toolkit:
NVIDIA GPU/CUDA and installed drivers.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

### Docker Container
From SfM_CUDA path run:

docker build -t="colmap:cuda" .

-# In some cases, you may have to explicitly specify the compute architecture (compute capability):
-#   docker build -t="colmap:latest" --build-arg CUDA_ARCHITECTURES=86 .
docker run --gpus all -w /working -v $1:/working -it colmap:cuda
-# Replace with your working directory (path to cloned repository) as this:

docker run \
    --gpus all \
    -w /working \
    -v /home/user/Documents/.../SfM_CUDA:/working \
    -it colmap:cuda

docker run \
    -e QT_XCB_GL_INTEGRATION=xcb_egl \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /working \
    -v /home/user/Documents/.../SfM_CUDA:/working \
    --gpus all \
    --privileged \
    -it colmap:cuda \
    # colmap gui

GUI Troubleshooting:
xhost +
sudo apt-get remove libxcb-xinerama0
sudo apt-get purge libxcb-xinerama0
sudo apt-get install libxcb-xinerama0
sudo apt install qtchooser
sudo apt install libqt5gui5

### Dataset generator
Download your rosbag under /main_folder and from Sfm_CUDA run:
python3 main_folder/sync_rosbag_tf_seek_read_all_GPS_fused.py
python3 main_folder/colmap_pipeline.py

