FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# System configuration and basic tools installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    curl \
    wget \
    git \
    vim \
    net-tools \
    lsb-release \
    gnupg2 \
    libjpeg-dev \
    libpng-dev \
    bzip2 \
    build-essential \
    cmake \
    software-properties-common \
    sudo \
    && locale-gen en_US.UTF-8 zh_CN.UTF-8 \
    && update-locale LANG=en_US.UTF-8 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Environment variables setup
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    CUDA_HOME=/usr/local/cuda-12.1 \
    PATH=/usr/local/cuda-12.1/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Install ROS 2
RUN mkdir -p /etc/apt/keyrings \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-pip \
    ros-humble-rqt* \
    ros-humble-rviz2 \
    ros-humble-rosbridge-suite \
    ros-humble-nav2-bringup \
    ros-humble-navigation2 \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-velodyne \
    ros-humble-pointcloud-to-laserscan \
    ros-humble-depthimage-to-laserscan \
    ros-humble-imu-tools \
    ros-humble-laser-filters \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-turtlebot3-simulations \
    ros-humble-turtlebot3-navigation2 \
    ros-humble-tf2-tools \
    ros-humble-diagnostic-updater \
    ros-humble-rmw-cyclonedds-cpp \
    && rosdep init && rosdep update \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Update all system packages
RUN apt-get update && apt upgrade -y && apt full-upgrade -y && apt autoremove -y && apt clean

# Miniconda installation and environment setup
ENV CONDA_DIR=/opt/conda

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && chmod +x /miniconda.sh \
    && bash /miniconda.sh -b -p $CONDA_DIR \
    && rm /miniconda.sh \
    && $CONDA_DIR/bin/conda clean -ya

# Create Conda environment and install dependencies
RUN $CONDA_DIR/bin/conda create -y -n dl_env python=3.10 \
    && $CONDA_DIR/bin/conda install -y -c conda-forge -n dl_env libstdcxx-ng \
    && $CONDA_DIR/bin/conda run -n dl_env pip install "numpy<2.0" pandas matplotlib scikit-learn jupyterlab \
    && $CONDA_DIR/bin/conda run -n dl_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && $CONDA_DIR/bin/conda clean -ya

# Install Segment Anything and Grounding DINO dependencies
RUN $CONDA_DIR/bin/conda run -n dl_env pip install opencv-python pycocotools matplotlib onnxruntime \
    && $CONDA_DIR/bin/conda run -n dl_env pip install git+https://github.com/facebookresearch/sam2.git \
    && $CONDA_DIR/bin/conda run -n dl_env pip install git+https://github.com/IDEA-Research/GroundingDINO.git \
    && $CONDA_DIR/bin/conda clean -ya

# Install Livox-SDK2
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git /workspace/Livox-SDK2 \
    && cd /workspace/Livox-SDK2 \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / && rm -rf /workspace/Livox-SDK2

# Configure ROS and Gazebo environment
RUN echo 'source $CONDA_DIR/etc/profile.d/conda.sh' >> /root/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /usr/share/gazebo/setup.bash" >> /root/.bashrc

# Set working directory
WORKDIR /workspace

# Set default shell configuration to system Python
SHELL ["/bin/bash", "-c"]

# Set default command
CMD ["bash"]