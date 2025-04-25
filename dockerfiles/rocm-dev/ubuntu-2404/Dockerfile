FROM ubuntu:24.04

LABEL maintainer=raayaiardakani.m@northeastern.edu

# Build Arguments
ARG ROCM_VERSION=6.4

# Install dependencies
RUN apt-get clean && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget ca-certificates curl  \
    build-essential software-properties-common cmake g++-12 libstdc++-12-dev rpm libelf-dev libnuma-dev sudo  \
    libdw-dev git python3 python3-pip gnupg unzip ripgrep libelf1 file pkg-config xxd ninja-build zsh git npm  \
    nodejs kmod
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Install neovim from AppImage
RUN wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux-x86_64.appimage && \
    chmod +x ./nvim-linux-x86_64.appimage && ./nvim-linux-x86_64.appimage --appimage-extract &&  \
    rm nvim-linux-x86_64.appimage && \
    cp -r ./squashfs-root/* / && rm -rf ./squashfs-root/

# Add the ROCm GPG keys
RUN sudo mkdir --parents --mode=0755 /etc/apt/keyrings && \
    wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Add the AMDGPU kernel-mode driver and ROCm repositories
RUN echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/$ROCM_VERSION/ubuntu noble main" \
    | sudo tee /etc/apt/sources.list.d/amdgpu.list && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM_VERSION noble main" \
    | sudo tee --append /etc/apt/sources.list.d/rocm.list
RUN echo 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600
RUN cat /etc/apt/preferences.d/rocm-pin-600 && apt-get update &&  \
    apt-get install -y rocm rocm-dev rocm-llvm-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/opt/rocm/lib"
