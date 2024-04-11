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

# Actualiza los paquetes e instala las dependencias necesarias de COLMAP
# Para GPU: https://github.com/colmap/colmap/blob/main/docker/Dockerfile
RUN sudo apt-get install -y \
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

# Install CUDA drivers
RUN sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc

RUN sudo apt-get install -y nvidia-driver-430

RUN sudo apt-get install gcc-10 g++-10 \
&& export CC=/usr/bin/gcc-10 \
&& export CXX=/usr/bin/g++-10 \
&& export CUDAHOSTCXX=/usr/bin/g++-10

# Install COLMAP

RUN sudo mkdir COLMAP \
    && sudo chmod 777 COLMAP \
    && cd COLMAP \
    && git clone https://github.com/colmap/colmap.git \
    && cd colmap \
    && mkdir build && cd build \
    && cmake .. -GNinja \
    && ninja \
    && sudo ninja install
    #&& cmake .. && make -j$(nproc) \
    #&& make install

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
