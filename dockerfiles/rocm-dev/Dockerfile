FROM ubuntu:24.04

LABEL maintainer=raayaiardakani.m@northeastern.edu

# Build Arguments
ARG ROCM_VERSION=6.3.1

# Install dependencies
RUN apt-get clean && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget ca-certificates curl  \
    build-essential  software-properties-common cmake g++-12 libstdc++-12-dev rpm libelf-dev libnuma-dev sudo  \
    libdw-dev git python3 python3-pip gnupg unzip ripgrep libelf1 file pkg-config xxd ninja-build zsh git npm  \
    python3.12-venv nodejs kmod
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# ROCm installation
RUN echo "Package: *\nPin: origin ""\nPin-Priority: 600" > /etc/apt/preferences.d/rocm-pin-600
RUN curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
  && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ noble main" |  \
    tee --append /etc/apt/sources.list.d/rocm.list \
  && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$ROCM_VERSION/ubuntu noble main" |  \
    tee /etc/apt/sources.list.d/amdgpu.list \
  && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  \
    rocm-libs$ROCM_VERSION  \
    rocm-dev$ROCM_VERSION  \
    rocm-llvm-dev$ROCM_VERSION && apt-get clean && rm -rf /var/lib/apt/lists/*