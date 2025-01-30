ARG ROCM_VERSION=6.3.1

FROM rocm-$ROCM_VERSION-dev

# Environment variables and build arguments

ARG NVIM_VER=0.10.2
ARG LLVM_BUILD_TYPE=RelWithDebInfo
ARG BUILD_SHARED_LIBS=OFF
ARG LLVM_SRC_DIR=/src/
ARG LLVM_ENABLE_EXPENSIVE_CHECKS=OFF
ARG LLVM_ENABLE_ASSERTIONS=OFF


# Install neovim from AppImage
RUN wget https://github.com/neovim/neovim/releases/download/v${NVIM_VER}/nvim.appimage && \
        chmod +x ./nvim.appimage && ./nvim.appimage --appimage-extract && rm nvim.appimage && \
        cp -r ./squashfs-root/* / && rm -rf ./squashfs-root/

RUN pip3 install cxxheaderparser pcpp --break-system-packages
# Clone all LLVM-based ROCm projects in the $LLVM_SRC_DIR directory
RUN mkdir $LLVM_SRC_DIR && cd $LLVM_SRC_DIR && git clone https://github.com/ROCm/llvm-project/ --depth 1
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
## Install ROCm Device Libs
RUN cd $LLVM_SRC_DIR/llvm-project/amd/device-libs && mkdir build && cd build &&  \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release  \
    -DCMAKE_INSTALL_PREFIX=/opt/luthier/  \
    -DCMAKE_PREFIX_PATH="/opt/luthier/" ../ && LD_LIBRARY_PATH=/opt/luthier/llvm/lib/ ninja install &&  \
    rm -rf $LLVM_SRC_DIR/llvm-project/amd/device-libs/build/
## HIPCC Installation
RUN cd $LLVM_SRC_DIR/llvm-project/amd/hipcc/ && mkdir build && cd build &&  \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/luthier/ -DCMAKE_PREFIX_PATH="/opt/luthier/"  \
    -DCMAKE_BUILD_TYPE=Release  \
    .. && LD_LIBRARY_PATH=/opt/luthier/llvm/lib/ ninja install && rm -rf $LLVM_SRC_DIR/llvm-project/amd/hipcc/build/ \

ENV LD_LIBRARY_PATH "/opt/rocm/lib:/opt/luthier/llvm/lib/:/opt/luthier/lib"
