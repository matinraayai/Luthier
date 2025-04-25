FROM rockylinux/rockylinux:8-ubi

LABEL maintainer=raayaiardakani.m@northeastern.edu

# Build Arguments
ARG ROCM_VERSION=6.4

# Install dependencies
RUN dnf upgrade -y --refresh && dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --set-enabled powertools && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm

# Add the AMDGPU repository
RUN echo -e "[AMDGPU-${ROCM_VERSION}]\n\
name=AMDGPU${ROCM_VERSION}\n\
baseurl=https://repo.radeon.com/amdgpu/${ROCM_VERSION}/el/8.10/main/x86_64\n\
enabled=1\n\
priority=50\n\
gpgcheck=1\n\
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/amdgpu.repo

# Add the ROCm repository
RUN echo -e "[ROCm-${ROCM_VERSION}]\n\
name=ROCm${ROCM_VERSION}\n\
baseurl=https://repo.radeon.com/rocm/el8/${ROCM_VERSION}/main\n\
enabled=1\n\
priority=50\n\
gpgcheck=1\n\
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/rocm.repo

RUN dnf clean all && dnf install -y neovim git cmake ninja-build rocm \
    gcc-toolset-12 gcc-toolset-12-libstdc++-devel gcc-toolset-12-libasan-devel \
    gcc-toolset-12-liblsan-devel gcc-toolset-12-libtsan-devel gcc-toolset-12-libubsan-devel \
    elfutils-libelf-devel sudo libdwarf-devel unzip ripgrep elfutils-libelf-devel \
    file pkgconf-pkg-config vim-common zsh npm nodejs kmod

# Enable the toolset permenantly
RUN alternatives --install /usr/bin/gcc gcc  /opt/rh/gcc-toolset-12/root/bin/gcc 20 --slave \
    /usr/bin/g++ g++ /opt/rh/gcc-toolset-12/root/bin/g++

RUN alternatives --install /usr/bin/ar ar /opt/rh/gcc-toolset-12/root/bin/ar 20


