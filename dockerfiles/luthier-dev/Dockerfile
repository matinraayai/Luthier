ARG ROCM_VERSION=7.2.4

FROM matinraayai/rocm:$ROCM_VERSION

# Environment variables and build arguments
ARG LLVM_BUILD_TYPE=RelWithDebInfo
ARG BUILD_SHARED_LIBS=ON
ARG LLVM_SRC_DIR=/opt/luthier/src/
ARG LLVM_ENABLE_EXPENSIVE_CHECKS=OFF
ARG LLVM_ENABLE_ASSERTIONS=ON
ARG LLVM_GIT_REVISION=e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893

# Install neovim from the unstable PPA
RUN add-apt-repository ppa:neovim-ppa/unstable -y && \
    apt-get update && apt-get install -y neovim

# Install other other useful development packages
RUN apt-get install -y npm python3.12-venv nodejs pkg-config xxd ninja-build zsh \
    unzip ripgrep libdw-dev rpm

# Install ROCm packages' debug information
RUN apt-get install -y amd-smi-lib-dbgsym comgr-dbgsym hip-runtime-amd-dbgsym hipblas-dbgsym \
    hipcc-dbgsym hipfft-dbgsym hiprand-dbgsym hipsolver-dbgsym hipsparse-dbgsym hsa-rocr-dbgsym \
    rccl-dbgsym rocalution-dbgsym rocfft-dbgsym rocm-core-dbgsym rocm-dbgapi-dbgsym \
    rocm-debug-agent-dbgsym rocm-opencl-dbgsym rocm-smi-lib-dbgsym rocminfo-dbgsym rocprofiler-dbgsym \
    rocprofiler-plugins-dbgsym rocprofiler-sdk-dbgsym rocprofiler-sdk-rocpd-dbgsym rocprofiler-sdk-roctx-dbgsym \
    rocrand-dbgsym rocsparse-dbgsym roctracer-dbgsym && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone all LLVM-based ROCm projects in the $LLVM_SRC_DIR directory
RUN mkdir -p $LLVM_SRC_DIR && cd $LLVM_SRC_DIR && \
    mkdir llvm-project && cd llvm-project &&  \
    git init && git remote add origin https://github.com/ROCm/llvm-project/ && \
    git fetch origin --depth 1 $LLVM_GIT_REVISION && \
    git checkout FETCH_HEAD

RUN mkdir $LLVM_SRC_DIR/llvm-project/build && cd $LLVM_SRC_DIR/llvm-project/build &&  \
    cmake -G Ninja -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
    -DCMAKE_INSTALL_PREFIX=/opt/luthier/llvm/  \
    -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86"  \
    -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;compiler-rt;clang-tools-extra" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
    -DLLVM_ENABLE_EXPENSIVE_CHECKS=$LLVM_ENABLE_EXPENSIVE_CHECKS \
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
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_INSTALL_GTEST=ON \
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
    ../llvm && LD_LIBRARY_PATH=$LLVM_SRC_DIR/llvm-project/build/lib ninja install &&  \
    rm -rf $LLVM_SRC_DIR/llvm-project/build
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
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/luthier/ -DCMAKE_PREFIX_PATH="/opt/luthier/"  \
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