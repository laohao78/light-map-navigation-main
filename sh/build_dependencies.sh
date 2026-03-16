#!/bin/bash

# Define color output
GREEN='\033[0;32m'
NC='\033[0m'

# Define conda variables
CONDA_ENV="dl_env"
CONDA_PATH="/opt/conda/bin/conda"
CONDA_PIP="pip"

# Define helper functions
log_step() {
    echo -e "\n${GREEN}=== $1 ===${NC}"
}

install_in_conda() {
    echo "Installing $1..."
    $CONDA_PATH run -n $CONDA_ENV $CONDA_PIP install $2
}

# System dependencies installation
log_step "Starting Dependency Installation"
apt update -y
apt install -y python3-pip
pip3 install -r requirements.txt

# ROS dependencies installation
log_step "Installing ROS Dependencies"
DEBIAN_FRONTEND=noninteractive apt install -y
rosdep install --from-paths src --ignore-src -r -y

# Conda environment configuration
log_step "Configuring Python Environment"
echo "Installing libstdcxx-ng..."
$CONDA_PATH install -y -c conda-forge -n $CONDA_ENV libstdcxx-ng

# Python package installation
log_step "Installing Python Packages"
install_in_conda "Flask" "flask"
install_in_conda "grounded_sam2" "-e src/grounded_sam2/grounded_sam2/grounded_sam2"
install_in_conda "grounding_dino" "--no-build-isolation -e src/grounded_sam2/grounded_sam2/grounded_sam2/grounding_dino"