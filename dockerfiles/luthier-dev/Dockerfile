ARG ROCM_VERSION=10.1.0
ARG UBUNTU_VERSION=26.04

FROM rocm/dev-ubuntu-$UBUNTU_VERSION:$ROCM_VERSION-full

# Environment variables and build arguments
ARG LLVM_BUILD_TYPE=RelWithDebInfo
ARG BUILD_SHARED_LIBS=OFF
ARG LLVM_SRC_DIR=/opt/luthier/src/
ARG LLVM_ENABLE_EXPENSIVE_CHECKS=OFF
ARG LLVM_ENABLE_ASSERTIONS=ON
ARG LLVM_GIT_REVISION=amd-staging

# Install dependencies
RUN apt-get clean && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget ca-certificates curl  \
    build-essential  software-properties-common cmake g++-12 libstdc++-12-dev rpm libelf-dev libnuma-dev sudo  \
    libdw-dev git python3 python3-pip gnupg unzip ripgrep libelf1 file pkg-config xxd ninja-build zsh git npm  \
    python3-venv nodejs kmod
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Install neovim from AppImage
RUN wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux-x86_64.appimage && \
    chmod +x ./nvim-linux-x86_64.appimage && ./nvim-linux-x86_64.appimage --appimage-extract &&  \
    rm nvim-linux-x86_64.appimage && \
    cp -r ./squashfs-root/* / && rm -rf ./squashfs-root/

# Clone all LLVM-based ROCm projects in the $LLVM_SRC_DIR directory
RUN mkdir -p $LLVM_SRC_DIR && cd $LLVM_SRC_DIR && \
    mkdir llvm-project && cd llvm-project &&  \
    git init && git remote add origin https://github.com/ROCm/llvm-project/ && \
    git fetch origin --depth 1 $LLVM_GIT_REVISION && \
    git checkout FETCH_HEAD

# Build straight into the installation directory so we don't have to resort to shenanigans
# of moving the dwp files from the build folder to the install folder while ensuring every
# moved object points to the correct dwp file (especially for static libraries).
RUN mkdir -p /opt/luthier/llvm && cd /opt/luthier/llvm &&  \
    cmake -G Ninja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
    -DCMAKE_INSTALL_PREFIX=/opt/luthier/llvm/  \
    -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_SHLIB_SYMBOL_VERSION=LUTHIER_LLVM \
    -DLLVM_VERSION_SUFFIX=luthier \
    -DCMAKE_C_FLAGS="-gz=zlib" \
    -DCMAKE_CXX_FLAGS="-fno-gnu-unique -gz=zlib" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--compress-debug-sections=zstd" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--compress-debug-sections=zstd" \
    -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--compress-debug-sections=zstd" \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;Native;SPIRV"  \
    -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;compiler-rt;clang-tools-extra" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
    -DLLVM_PARALLEL_LINK_JOBS=4 \
    -DLLVM_ENABLE_EXPENSIVE_CHECKS=$LLVM_ENABLE_EXPENSIVE_CHECKS \
    -DLIBCXX_ENABLE_SHARED=OFF \
    -DLIBCXX_ENABLE_STATIC=ON \
    -DLIBCXX_INSTALL_LIBRARY=OFF \
    -DLIBCXX_INSTALL_HEADERS=OFF \
    -DLIBCXXABI_ENABLE_SHARED=OFF \
    -DLIBCXXABI_ENABLE_STATIC=ON \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DLIBCXXABI_INSTALL_STATIC_LIBRARY=OFF \
    -DLLVM_ENABLE_RTTI=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DCLANG_ENABLE_AMDCLANG=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_INSTALL_GTEST=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCLANG_INCLUDE_TESTS=OFF \
    -DLLVM_BUILD_DOCS=OFF \
    -DLLVM_ENABLE_SPHINX=OFF \
    -DSPHINX_WARNINGS_AS_ERRORS=OFF \
    -DSPHINX_OUTPUT_MAN=OFF \
    -DLLVM_ENABLE_ASSERTIONS=$LLVM_ENABLE_ASSERTIONS \
    -DLLVM_ENABLE_Z3_SOLVER=OFF \
    -DLLVM_ENABLE_ZLIB=ON \
    -DLLVM_AMDGPU_ALLOW_NPI_TARGETS=ON \
    -DCLANG_DEFAULT_PIE_ON_LINUX=0 \
    -DCLANG_DEFAULT_LINKER=lld \
    -DCLANG_DEFAULT_RTLIB=compiler-rt \
    -DCLANG_DEFAULT_UNWINDLIB=libgcc \
    -DSANITIZER_AMDGPU=OFF \
    -DPACKAGE_VENDOR="AMD" \
    -DCLANG_LINK_FLANG_LEGACY=ON \
    -DCMAKE_SKIP_BUILD_RPATH=TRUE \
    -DCMAKE_SKIP_INSTALL_RPATH=TRUE \
    -DFLANG_INCLUDE_DOCS=OFF \
    $LLVM_SRC_DIR/llvm-project/llvm && LD_LIBRARY_PATH=/opt/luthier/llvm/lib ninja
## Install LLVM Lit
RUN pip install $LLVM_SRC_DIR/llvm-project/llvm/utils/lit/ --break-system-packages
## Install ROCm Device Libs
RUN cd $LLVM_SRC_DIR/llvm-project/amd/device-libs && mkdir build && cd build &&  \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE  \
    -DCMAKE_INSTALL_PREFIX=/opt/luthier/  \
    -DCMAKE_PREFIX_PATH="/opt/luthier/" ../ && LD_LIBRARY_PATH=/opt/luthier/llvm/lib/ ninja install &&  \
    rm -rf $LLVM_SRC_DIR/llvm-project/amd/device-libs/build/
## HIPCC Installation
RUN cd $LLVM_SRC_DIR/llvm-project/amd/hipcc/ && mkdir build && cd build &&  \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/luthier/ -DCMAKE_PREFIX_PATH="/opt/luthier/;/opt/rocm/"  \
    -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE  \
    .. && LD_LIBRARY_PATH=/opt/luthier/llvm/lib/ ninja install && rm -rf $LLVM_SRC_DIR/llvm-project/amd/hipcc/build/
## SPIRV Installation
RUN git clone --depth 1 https://github.com/ROCm/SPIRV-LLVM-Translator && \
    mkdir SPIRV-LLVM-Translator/build && cd SPIRV-LLVM-Translator/build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE -DCMAKE_INSTALL_PREFIX=/opt/luthier \
    -DCMAKE_PREFIX_PATH=/opt/luthier -DLLVM_DIR=/opt/luthier/llvm/lib/cmake/llvm .. && \
    LD_LIBRARY_PATH=/opt/luthier/llvm/lib ninja install && \
    cd ../../ && rm -rf SPIRV-LLVM-Translator

ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/luthier/llvm/lib/:/opt/luthier/lib"