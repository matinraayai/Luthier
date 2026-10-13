ARG ROCM_VERSION=10.1.0
ARG UBUNTU_VERSION=26.04

FROM rocm/dev-ubuntu-$UBUNTU_VERSION:$ROCM_VERSION-full

# Environment variables and build arguments
ARG CMAKE_BUILD_TYPE=RelWithDebInfo
ARG LUTHIER_INSTALL_PREFIX=/opt/luthier
ARG THEROCK_GIT_VERSION=735456b3c9b61b4b67b49f470d18860c8743d063
ARG GOTCHA_VERSION=v1.0.10

# Install boto3 required by the fetch_sources.py script
RUN pip3 install --break-system-packages boto3

# Install deb package dependencies
RUN apt-get clean && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget ca-certificates curl  \
    build-essential automake software-properties-common cmake g++-12 libstdc++-12-dev rpm libelf-dev libnuma-dev sudo  \
    libdw-dev git python3 python3-pip gnupg unzip ripgrep libelf1 file pkg-config xxd ninja-build zsh git npm patchelf \
    python3-venv nodejs kmod meson libtool
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Install neovim from AppImage
RUN wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux-x86_64.appimage && \
    chmod +x ./nvim-linux-x86_64.appimage && ./nvim-linux-x86_64.appimage --appimage-extract &&  \
    rm nvim-linux-x86_64.appimage && \
    cp -r ./squashfs-root/* / && rm -rf ./squashfs-root/

# Install GOTCHA
RUN mkdir -p $LUTHIER_INSTALL_PREFIX/src/GOTCHA && cd $LUTHIER_INSTALL_PREFIX/src/GOTCHA &&  \
      git init && git remote add origin https://github.com/LLNL/GOTCHA/ && \
      git fetch origin --depth 1 $GOTCHA_VERSION && \
      git checkout FETCH_HEAD && mkdir build && cd build && \
      cmake -DCMAKE_INSTALL_PREFIX=$LUTHIER_INSTALL_PREFIX -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE \
        -GNinja .. && ninja install && cd ../ && rm -rf build/

# Clone the LLVM sources using TheRock
RUN mkdir -p $LUTHIER_INSTALL_PREFIX/src/ && cd $LUTHIER_INSTALL_PREFIX/src/ &&  \
    git clone https://github.com/ROCm/TheRock/ && cd TheRock && git checkout $THEROCK_GIT_VERSION && \
    python3 build_tools/fetch_sources.py --depth 1 --progress --jobs 4 --include-system-projects \
    --no-include-rocm-libraries --include-rocm-systems --no-include-math-libraries --no-include-media-libs \
    --no-include-debug-tools --no-include-ml-frameworks --include-compilers --system-projects rocm-cmake

# Copy the LLVM build overrides into the container
COPY llvm_overrides.cmake $LUTHIER_INSTALL_PREFIX/src/llvm_overrides.cmake

# Run the build process
RUN mkdir $LUTHIER_INSTALL_PREFIX/src/TheRock/build && cd $LUTHIER_INSTALL_PREFIX/src/TheRock/build &&  \
    cmake -DCMAKE_INSTALL_PREFIX=$LUTHIER_INSTALL_PREFIX -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -GNinja \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTHEROCK_DIST_AMDGPU_FAMILIES="dgpu-all;igpu-all;dcgpu-all" -DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_BASE=ON \
    -DTHEROCK_ENABLE_COMPILER=ON -DTHEROCK_BUNDLE_SYSDEPS=OFF -DTHEROCK_ENABLE_SYSDEPS=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=8 -DFLANG_PARALLEL_COMPILE_JOBS=8 \
    -Damd-llvm_CMAKE_ARGS="-DCMAKE_PROJECT_LLVM_INCLUDE=$LUTHIER_INSTALL_PREFIX/src/llvm_overrides.cmake;-DTHEROCK_BUILD_LLVM_TOOLS=ON;-DLLVM_INCLUDE_UTILS=ON;-DCMAKE_POSITION_INDEPENDENT_CODE=ON" \
    -Dhipcc_CMAKE_ARGS="-DCMAKE_POSITION_INDEPENDENT_CODE=ON"  \
    -D"amd-comgr_CMAKE_ARGS"="-DTHEROCK_BUILD_COMGR_TESTS=OFF" ..  \
    && ninja amd-llvm hipcc -v && ninja install && rm -rf $LUTHIER_INSTALL_PREFIX/src/TheRock/build

## Install LLVM Lit
RUN pip install $LLVM_SRC_DIR/llvm-project/llvm/utils/lit/ --break-system-packages

ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/luthier/llvm/lib/:/opt/luthier/lib"