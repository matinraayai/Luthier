FROM ubuntu:24.04

# Environment variables and build arguments

ENV DEBIAN_FRONTEND=noninteractive
ARG ROCM_VERSION=6.2.2
ARG ROCM_INSTALL_DEB_VERSION=6.2.60202-1
ARG NVIM_VER=0.10.2

# Install dependencies
RUN apt-get clean && apt-get update && apt-get install -y wget build-essential software-properties-common \
        cmake g++-12 libstdc++-12-dev rpm libelf-dev \
        libdw-dev git python3 python3-pip gnupg unzip ripgrep \
        pkg-config xxd ninja-build zsh git npm python3.12-venv nodejs
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12
RUN pip3 install cxxheaderparser pcpp --break-system-packages

# Install neovim from AppImage
RUN wget https://github.com/neovim/neovim/releases/download/v${NVIM_VER}/nvim.appimage && \
        chmod +x ./nvim.appimage && ./nvim.appimage --appimage-extract && rm nvim.appimage && \
        cp -r ./squashfs-root/* / && rm -rf ./squashfs-root/

# Install NvChad
RUN git clone https://github.com/NvChad/starter /root/.config/nvim --depth 1

# Proceed with Normal ROCm installation
RUN wget https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${ROCM_INSTALL_DEB_VERSION}_all.deb
RUN apt-get install -y ./amdgpu-install_${ROCM_INSTALL_DEB_VERSION}_all.deb
RUN amdgpu-install -y --no-dkms --usecase=rocm
RUN rm amdgpu-install_${ROCM_INSTALL_DEB_VERSION}_all.deb

# Remove any existing ROCm LLVM packages and build the latest version of LLVM and its byproducts against AMD's staging branch
# This is for Luthier's compiler plugin usage
RUN rm -rf /opt/rocm/lib/llvm && git clone https://github.com/ROCm/llvm-project/ -b amd-staging --depth 1 amd-llvm/

RUN mkdir /amd-llvm/build && \
        cd amd-llvm/build && \
         cmake -G Ninja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/opt/rocm/lib/llvm \
        -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
        -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;compiler-rt;clang-tools-extra;flang;mlir" \
        -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
        -DLIBCXX_ENABLE_SHARED=OFF \
        -DLIBCXX_ENABLE_STATIC=ON \
        -DLIBCXX_INSTALL_LIBRARY=OFF \
        -DLIBCXX_INSTALL_HEADERS=OFF \
        -DLIBCXXABI_ENABLE_SHARED=OFF \
        -DLIBCXXABI_ENABLE_STATIC=ON \
        -DLIBCXXABI_INSTALL_STATIC_LIBRARY=OFF \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_OPTIMIZED_TABLEGEN=ON \
        -DCLANG_ENABLE_AMDCLANG=ON \
        -DLLVM_BUILD_TOOLS=ON \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_BUILD_TESTS=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DCLANG_INCLUDE_TESTS=OFF \
        -DLLVM_BUILD_DOCS=OFF \
        -DLLVM_ENABLE_SPHINX=OFF \
        -DSPHINX_WARNINGS_AS_ERRORS=OFF \
        -DSPHINX_OUTPUT_MAN=OFF \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
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
        -DLIBOMP_HAVE_QUAD_PRECISION=OFF \
        ../llvm
RUN cd /amd-llvm/build && ninja install
#RUN cd /amd-llvm/build && ninja install
## Install ROCm Device Libs
RUN rm -rf /opt/rocm/amdgcn && cd /amd-llvm/amd/device-libs && mkdir build && cd build && \
        cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX=/opt/rocm \
                -DCMAKE_PREFIX_PATH="/opt/rocm/cmake" ../ && ninja install && rm -rf *

RUN cd /amd-llvm/build && cmake -DLIBOMP_USE_QUAD_PRECISION=OFF -DLLVM_ENABLE_RUNTIMES="openmp;offload" ../llvm && ninja
## COMGr Installation
RUN cd /amd-llvm/amd/comgr/ && mkdir build && cd build && \
        cmake -G Ninja -DBUILD_TESTING=OFF -DCOMGR_DISABLE_SPIRV=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DCMAKE_PREFIX_PATH="/opt/rocm/cmake;/opt/rocm/llvm/lib/cmake" .. \
        && ninja install && rm -rf *
## HIPCC Installation
RUN cd /amd-llvm/amd/hipcc/ && mkdir build && cd build && \
        cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/rocm/ -DCMAKE_BUILD_TYPE=Release \
        .. && ninja install && rm -rf *
RUN cd / && rm -rf amd-llvm/
#
# Build CLR from scratch to workaround https://github.com/ROCm/clr/issues/96
RUN git clone https://github.com/ROCm/HIP -b rocm-$ROCM_VERSION --depth 1 && \
        git clone https://github.com/matinraayai/clr -b rocm-6.2.x --depth 1
RUN pip3 install --break-system-packages CppHeaderParser
RUN mkdir /clr/build && cd /clr/build && \
        cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/rocm \
                        -DCLR_BUILD_HIP=ON \
                        -DCLR_BUILD_OPENCL=ON \
                        -DHIP_COMMON_DIR=/HIP/ .. && ninja install
RUN cd / && rm -rf HIP/ clr/

## Install the LLVM version used for development directly by Luthier
RUN git clone https://github.com/llvm/llvm-project/ && \
        mkdir /llvm-project/build && cd /llvm-project/build && \
        cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_TARGETS_TO_BUILD="AMDGPU" \
                -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
                -DLLVM_ENABLE_RTTI=ON -DBUILD_SHARED_LIBS=ON \
                -DLLVM_ENABLE_EXPENSIVE_CHECKS=ON \
                -DLLVM_ENABLE_ASSERTIONS=ON \
                -DLLVM_OPTIMIZED_TABLEGEN=ON \
                -DLLVM_BUILD_TOOLS=ON \
                -DLLVM_BUILD_EXAMPLES=OFF ../llvm && ninja

RUN git clone -b rocm-6.2.2 --depth 1 https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-src
RUN python3 -m pip install --break-system-packages -r rocprofiler-sdk-src/requirements.txt
RUN cmake rocprofiler-sdk-src/. -B rocprofiler-sdk-build -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm  \
    -DROCPROFILER_BUILD_TESTS=OFF -DROCPROFILER_BUILD_SAMPLES=OFF -DCMAKE_BUILD_TYPE=$build_type  \
    -DGPU_TARGETS=$gpu_targets
RUN cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)
RUN cmake --build rocprofiler-sdk-build --target install --parallel $(nproc)
