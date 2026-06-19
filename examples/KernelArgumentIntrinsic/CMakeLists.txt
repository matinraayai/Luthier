cmake_minimum_required(VERSION 3.21)

project(LuthierKernelArgumentIntrinsic LANGUAGES HIP)

set(CMAKE_HIP_STANDARD 20)

find_package(hip REQUIRED)
find_package(LLVM REQUIRED)

luthier_create_offload_bundle(LuthierKernelArgumentIntrinsic KernelArgumentIntrinsic.hip
        DEVICE_OBJECT_LIBRARIES LuthierKernelArgumentIntrinsic_DEVICE_TARGETS)

target_link_libraries(LuthierKernelArgumentIntrinsic PUBLIC LuthierTooling hip::host LLVMDemangle)
foreach (_DEVICE_TARGET IN LISTS LuthierKernelArgumentIntrinsic_DEVICE_TARGETS)
    target_link_libraries(${_DEVICE_TARGET} PRIVATE LuthierTooling hip::host)
endforeach ()
