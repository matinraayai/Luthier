ARG ROCM_DEV_BASE=containers.rc.northeastern.edu/luthier/rocm-6.4-dev-ubuntu-2404

FROM ${ROCM_DEV_BASE}

# Environment variables and build arguments
ARG LLVM_BUILD_TYPE=RelWithDebInfo
ARG BUILD_SHARED_LIBS=ON
ARG LLVM_SRC_DIR=/opt/luthier/src/
ARG LLVM_ENABLE_EXPENSIVE_CHECKS=OFF
ARG LLVM_ENABLE_ASSERTIONS=OFF
ARG LLVM_GIT_REVISION=5e51f7702e2703df95f7a3d57284a1fdef4766b7

ENV PIP_BREAK_SYSTEM_PACKAGES=1

ENV LD_LIBRARY_PATH="/opt/rocm/lib:/opt/luthier/llvm/lib/:/opt/luthier/lib"

RUN pip3 install cxxheaderparser pcpp
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
    -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;clang-tools-extra" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind;compiler-rt" \
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
    ../llvm && LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$LLVM_SRC_DIR/llvm-project/build/lib ninja install &&  \
    rm -rf $LLVM_SRC_DIR/llvm-project/build
## Install LLVM Lit
RUN pip install $LLVM_SRC_DIR/llvm-project/llvm/utils/lit/
## Install ROCm Device Libs
RUN cd $LLVM_SRC_DIR/llvm-project/amd/device-libs && mkdir build && cd build &&  \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE  \
    -DCMAKE_INSTALL_PREFIX=/opt/luthier/  \
    -DCMAKE_PREFIX_PATH="/opt/luthier/" ../ && ninja install &&  \
    rm -rf $LLVM_SRC_DIR/llvm-project/amd/device-libs/build/
## HIPCC Installation
RUN cd $LLVM_SRC_DIR/llvm-project/amd/hipcc/ && mkdir build && cd build &&  \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/luthier/ -DCMAKE_PREFIX_PATH="/opt/luthier/"  \
    -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE  \
    .. && ninja install && rm -rf $LLVM_SRC_DIR/llvm-project/amd/hipcc/build/
## Comgr Installation
RUN cd $LLVM_SRC_DIR/llvm-project/amd/comgr/ && mkdir build && cd build && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/luthier/ -DCMAKE_PREFIX_PATH="/opt/luthier/llvm"  \
    -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE  \
    .. && LD_LIBRARY_PATH="/opt/rocm/lib:/opt/luthier/llvm/lib/:/opt/luthier/lib" ninja install  \
    && rm -rf $LLVM_SRC_DIR/llvm-project/amd/comgr/build/
## Create some symlinks for hipcc
RUN mkdir -p /opt/luthier/lib/ && ln -s /opt/luthier/llvm /opt/luthier/lib/llvm &&  \
    ln -s /opt/rocm/share/hip/ /opt/luthier/share/hip