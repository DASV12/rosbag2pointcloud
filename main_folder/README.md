# COLMAP CLI
# Documentation: https://colmap.github.io/cli.html#cli
# or $ colmap help

# Create a new project under colmap_ws/
# Create images/ folder under your_new_project/

# This option runs an automatic recostruction, by default exhaustive matching (Global).
# The project folder must contain a folder "images" with all the images.
# The available options can either be provided directly
# from the command-line or through a .ini file provided to --project_path.

$ DATASET_PATH=/path/to/project
$ DATASET_PATH=/home/david_serna/Documents/SfM_Kiwibot_Repository/SfM/colmap_ws/rosbag_office/reconstructions/fuse_2_street_front_gps_spatial
$ echo $DATASET_PATH

$ colmap automatic_reconstructor \
    --workspace_path $DATASET_PATH \
    --image_path $DATASET_PATH/images

# You can manually select to use CPU-based feature extraction and matching
# by setting the --SiftExtraction.use_gpu 0 and --SiftMatching.use_gpu 0 options.
# My machine (David Serna) does not have GPU so this parameters are turned off.

# This option runs a reconstruction step by step.
# The project folder must contain a folder "images" with all the images.

$ DATASET_PATH=/path/to/project

$ colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.camera_model SIMPLE_PINHOLE \
   --ImageReader.single_camera 1 \
   --SiftExtraction.use_gpu 0

# You can run diferent matchers based on your data (Individual images, ordered frames from video...)
# Exhaustive is considered  Global matching, and sequential is recommended for videos.

$ colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

$ colmap sequential_matcher \
   --database_path $DATASET_PATH/database.db

$ colmap spatial_matcher \
   --database_path $DATASET_PATH/database.db
   --SiftMatching.use_gpu 0

# Generate sparse point cloud

$ mkdir $DATASET_PATH/sparse

$ colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

# Export as ply and text. --output_type {BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}

$ colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0/ \
    --output_type TXT
     
$ colmap model_converter \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/sparse/0/sparseModel.ply \
    --output_type PLY
    
$ colmap model_converter \
    --input_path $DATASET_PATH \
    --output_path $DATASET_PATH \
    --output_type BIN

$ python3 /COLMAP/colmap/scripts/python/read_write_model.py --input_model $DATASET_PATH/sparse/0 --output_model $DATASET_PATH/sparse/0 --input_format .bin --output_format .txt

# If you want to generate dense point cloud or mesh, run the following commands.

$ mkdir $DATASET_PATH/dense

$ colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

$ colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

$ colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

# Dense point cloud generated, now you can generate a mesh.

$ colmap poisson_mesher \
    --input_path $DATASET_PATH/dense/fused.ply \
    --output_path $DATASET_PATH/dense/meshed-poisson.ply

$ colmap delaunay_mesher \
    --input_path $DATASET_PATH/dense \
    --output_path $DATASET_PATH/dense/meshed-delaunay.ply

# Align reconstruction with GPS coordinates to obtain absolute scale
# https://colmap.github.io/faq.html#geo-registration
$ colmap model_aligner \
    --input_path /path/to/model \
    --output_path /path/to/geo-registered-model \
    --ref_images_path /path/to/text-file (or --database_path /path/to/database.db) \
    --ref_is_gps 1 \
    --alignment_type ecef \
    --robust_alignment 1 \
    --robust_alignment_max_error 3.0 (where 3.0 is the error threshold to be used in RANSAC)
    
$ colmap model_aligner \
    --input_path $DATASET_PATH \
    --output_path $DATASET_PATH \
    --ref_images_path $DATASET_PATH/gps_data.txt \
    --ref_is_gps 1 \
    --alignment_type ecef \
    --alignment_max_error 3.0
    
$ colmap model_aligner \
    --input_path $DATASET_PATH \
    --output_path $DATASET_PATH \
    --ref_images_path $DATASET_PATH/gps_data.txt \
    --ref_is_gps 1 \
    --alignment_type ecef \
    --alignment_max_error 3.0
