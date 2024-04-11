FROM osrf/ros:iron-desktop

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
# RUN sudo apt-get install -y \
#     nvidia-cuda-toolkit \
#     nvidia-cuda-toolkit-gcc

# RUN sudo apt-get install -y nvidia-driver-430

# RUN sudo apt-get install gcc-10 g++-10 \
# && export CC=/usr/bin/gcc-10 \
# && export CXX=/usr/bin/g++-10 \
# && export CUDAHOSTCXX=/usr/bin/g++-10

# # Install COLMAP

# RUN sudo mkdir COLMAP \
#     && sudo chmod 777 COLMAP \
#     && cd COLMAP \
#     && git clone https://github.com/colmap/colmap.git \
#     && cd colmap \
#     && mkdir build && cd build \
#     && cmake .. -GNinja \
#     && ninja \
#     && sudo ninja install
#     #&& cmake .. && make -j$(nproc) \
#     #&& make install

# ### End COLMAP
### COLMAP DOCKER
ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.3.1

#
# Docker builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=native
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git fetch https://github.com/colmap/colmap.git ${COLMAP_GIT_COMMIT} && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap_installed && \
    ninja install

#
# Docker runtime stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime

# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-filesystem1.74.0 \
        libboost-program-options1.74.0 \
        libc6 \
        libceres2 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v5 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5

# Copy all files from /colmap_installed/ in the builder stage to /usr/local/ in
# the runtime stage. This simulates installing COLMAP in the default location
# (/usr/local/), which simplifies environment variables. It also allows the user
# of this Docker image to use it as a base image for compiling against COLMAP as
# a library. For instance, CMake will be able to find COLMAP easily with the
# command: find_package(COLMAP REQUIRED).
COPY --from=builder /colmap_installed/ /usr/local/
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
