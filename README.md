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
### Notes
Feature extraction and sequential matching can be accelerated using GPU, and cloud densification and mesh generation can only be performed using GPU.
An Ubuntu-ROS2 container can only run COLMAP without GPU.
A pure colmap/colmap:latest image can´t run Dataset Generation stage
This project should be run in an nvidia/cuda image as suggested in the COLMAP repository. This option is still being tested due to issues with the version of the GPU used in this project.