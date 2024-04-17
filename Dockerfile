FROM osrf/ros:iron-desktop as base

# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git

# Install Python 3 and pip
RUN sudo apt install -y python3 python3-pip


# Install typer, tqdm, and zstandard de rosbag serializer
RUN pip3 install typer tqdm zstandard piexif

# COLMAP image


# ### Start COLMAP

# # Actualiza los paquetes e instala las dependencias necesarias de COLMAP
# # Para GPU: https://github.com/colmap/colmap/blob/main/docker/Dockerfile
# RUN sudo apt-get install -y \
#     git \
#     cmake \
#     ninja-build \
#     build-essential \
#     libboost-program-options-dev \
#     libboost-filesystem-dev \
#     libboost-graph-dev \
#     libboost-system-dev \
#     libeigen3-dev \
#     libflann-dev \
#     libfreeimage-dev \
#     libmetis-dev \
#     libgoogle-glog-dev \
#     libgtest-dev \
#     libsqlite3-dev \
#     libglew-dev \
#     qtbase5-dev \
#     libqt5opengl5-dev \
#     libcgal-dev \
#     libceres-dev

# # Install CUDA drivers
# # RUN sudo apt -y install ubuntu-drivers-common
# # RUN sudo ubuntu-drivers devices
# # Install recommended
# # For GeForce 920M:
# # RUN sudo apt -y install nvidia-driver-470

# RUN sudo apt-get -y install gcc-10 g++-10 \
#     && export CC=/usr/bin/gcc-10 \
#     && export CXX=/usr/bin/g++-10 \
#     && export CUDAHOSTCXX=/usr/bin/g++-10
    
# # Install wget
# RUN sudo apt-get update \
#     && sudo apt-get install -y wget

# RUN sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
#     && sudo dpkg -i cuda-keyring_1.1-1_all.deb \
#     && sudo apt-get update \
#     && sudo apt-get -y install cuda-toolkit-12-4

# RUN echo "export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}" >> ~/.bashrc
# RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\
# ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc


# # Install COLMAP
# # ARG UBUNTU_VERSION=22.04
# # ARG NVIDIA_CUDA_VERSION=12.4

# #
# # Docker builder stage.
# #
# # FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

# # ARG COLMAP_GIT_COMMIT=main
# ARG CUDA_ARCHITECTURES=native
# # ENV QT_XCB_GL_INTEGRATION=xcb_egl

# # Prevent stop building ubuntu at time zone selection.
# # ENV DEBIAN_FRONTEND=noninteractive

# RUN sudo mkdir COLMAP \
#     && sudo chmod 777 COLMAP \
#     && cd COLMAP \
#     && git clone https://github.com/colmap/colmap.git \
#     && cd colmap \
#     && mkdir build && cd build \
#     && cmake .. -GNinja -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4 \
#     && ninja \
#     && sudo ninja install
#     #&& cmake .. && make -j$(nproc) \
#     #&& make install
    
# #    && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
# #    -DCMAKE_INSTALL_PREFIX=/colmap_installed \ -DCUDA_TOOLKIT_ROOT_DIR=PATH

# # Docker runtime stage.
# #
# # FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime

#     # RUN sudo mkdir COLMAP \
#     # && sudo chmod 777 COLMAP \
#     # && cd COLMAP \
#     # && git clone https://github.com/colmap/colmap.git \
#     # && cd colmap \
#     # && mkdir build && cd build \
#     # && cmake .. -GNinja \
#     # && ninja \
#     # && sudo ninja install
#     # #&& cmake .. && make -j$(nproc) \
#     # #&& make install

# ### End COLMAP
### COLMAP DOCKER
# To compile with CUDA support, also install Ubuntu’s default CUDA package:
#RUN sudo apt-get install -y \
#    nvidia-cuda-toolkit \
#    nvidia-cuda-toolkit-gcc

# Rosdep update
RUN rosdep update
#RUN rosdep install --from-paths ~/serial/ros2_ws --ignore-src -r -y

#RUN apt-get update && apt-get install -y python3-rosdep
#RUN rosdep init && rosdep update


# Install ROS2 packages
#RUN apt-get update && apt-get install --no-install-recommends -y ros-${ROS_DISTRO}-message_filters

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source /serial/ros2_ws/install/setup.bash" >> ~/.bashrc

# Install message-filters
#RUN apt-get update && \
#    apt-get install -y \
#    ros-iron-message-filters
FROM colmap/colmap:latest as colmap

# Stage 3: Final image
FROM base


# Copy necessary files from the colmap stage
COPY --from=colmap / /
