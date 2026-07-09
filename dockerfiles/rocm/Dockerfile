FROM ubuntu:24.04

LABEL maintainer=raayaiardakani.m@northeastern.edu

# Build Arguments
ARG ROCM_VERSION=7.2.4
ARG AMDGPU_VERSION=31.30

# Install dependencies
RUN apt-get clean && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget ca-certificates curl  \
    build-essential software-properties-common cmake g++-12 libstdc++-12-dev libelf-dev libnuma-dev sudo  \
    python3 python3-pip gnupg libelf1 file git kmod
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 20 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# ROCm installation
RUN echo "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" > /etc/apt/preferences.d/rocm-pin-600
RUN curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
  && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ noble main" |  \
    tee --append /etc/apt/sources.list.d/rocm.list \
  && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu noble main" |  \
    tee /etc/apt/sources.list.d/amdgpu.list \
  && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  \
    rocm-libs  \
    rocm-dev  \
    rocm-llvm-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*