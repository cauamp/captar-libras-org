# Use CUDA 11.8 base with cuDNN8 for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install OS-level dependencies
RUN apt-get -y update

RUN apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    rsync \
    wget \
    ffmpeg \
    htop \
    libatlas-base-dev \
    libboost-all-dev \
    libeigen3-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopenblas-dev \
    libopenblas-base \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y  && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda for environment management
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

ENV PATH=$CONDA_DIR/bin:$PATH

# Copy and create the Conda environment
COPY environment.yml .

RUN conda config --set channel_priority strict
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml && \
    conda clean -afy

# Use Conda as the default shell for RUN commands
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Default to interactive shell in the environment
CMD ["conda", "run", "--no-capture-output", "-n", "myenv", "bash"]