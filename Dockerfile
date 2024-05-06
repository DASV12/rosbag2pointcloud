ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.3.1

#
# Docker builder stage.
#
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=86
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
        libqt5widgets5 \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel

#Install ROS2
# sudo add-apt-repository universe needs confirmation
# sudo apt update && sudo apt install ros-dev-tools needs
# Install ROS 2
# Set DEBIAN_FRONTEND to noninteractive
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#         curl \
#         gnupg2 \
#         lsb-release \
#         && \
#     curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
#     sh -c 'echo "deb [arch=amd64,arm64,armhf] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list' && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#         ros-iron-desktop \
#         && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
# Install ROS2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN apt install -y software-properties-common
RUN add-apt-repository -y universe
RUN apt update && apt install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-iron-desktop
RUN echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc

# Install other tools
RUN pip3 install opencv-python \
        typer \
        piexif \
        torch \
        ultralytics \
        zstandard
# Copy all files from /colmap_installed/ in the builder stage to /usr/local/ in
# the runtime stage. This simulates installing COLMAP in the default location
# (/usr/local/), which simplifies environment variables. It also allows the user
# of this Docker image to use it as a base image for compiling against COLMAP as
# a library. For instance, CMake will be able to find COLMAP easily with the
# command: find_package(COLMAP REQUIRED).
COPY --from=builder /colmap_installed/ /usr/local/
# /usr/bin/python3
