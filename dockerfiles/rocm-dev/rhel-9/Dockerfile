FROM registry.access.redhat.com/ubi9

LABEL maintainer=raayaiardakani.m@northeastern.edu

# Build Arguments
ARG ROCM_VERSION=6.3.3

# Install dependencies
RUN dnf update --releasever=9.5 --exclude=\*release\* && dnf install -y wget && \
    wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    rpm -ivh epel-release-latest-9.noarch.rpm && dnf install -y dnf-plugin-config-manager && \
    crb enable && rm -f epel-release-latest-9.noarch.rpm

# Add the AMDGPU repository
RUN echo -e "[AMDGPU-${ROCM_VERSION}]\n\
name=AMDGPU${ROCM_VERSION}\n\
baseurl=https://repo.radeon.com/amdgpu/${ROCM_VERSION}/rhel/9.5/main/x86_64\n\
enabled=1\n\
priority=50\n\
gpgcheck=1\n\
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/amdgpu.repo
# Add the ROCm repository
RUN echo -e "[ROCm-${ROCM_VERSION}]\n\
name=ROCm${ROCM_VERSION}\n\
baseurl=https://repo.radeon.com/rocm/el9/${ROCM_VERSION}/main\n\
enabled=1\n\
priority=50\n\
gpgcheck=1\n\
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" >> /etc/yum.repos.d/rocm.repo

RUN dnf clean all && dnf install rocm


