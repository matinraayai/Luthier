#ifndef SRC_INTERNAL_H
#define SRC_INTERNAL_H

#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

// hip-clang fatbin format
constexpr unsigned __hipFatMAGIC2 = 0x48495046;  // "HIPF"

#define CLANG_OFFLOAD_BUNDLER_MAGIC "__CLANG_OFFLOAD_BUNDLE__"
#define AMDGCN_AMDHSA_TRIPLE "hip-amdgcn-amd-amdhsa"

struct __ClangOffloadBundleDesc {
  uint64_t offset;
  uint64_t size;
  uint64_t tripleSize;
  const char triple[1];

  const struct __ClangOffloadBundleDesc* next() const {
    return reinterpret_cast<const struct __ClangOffloadBundleDesc*>(triple +
                                                                    tripleSize);
  }
};

struct __ClangOffloadBundleHeader {
  const char magic[sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1];
  uint64_t numBundles;
  __ClangOffloadBundleDesc desc[1];
};

struct __CudaFatBinaryWrapper {
  unsigned int magic;
  unsigned int version;
  __ClangOffloadBundleHeader* binary;
  void* unused;
};

typedef struct hsa_executable_s {
  uint64_t handle;
} hsa_executable_t;

struct CodeObject {
  void* ptr;
  size_t size;
  nlohmann::json note;
};

struct ihipModule_t {
  std::string fileName;
  hsa_executable_t executable = {};
  //   hsa_code_object_reader_t coReader = {};
  std::string hash;
  std::unordered_map<std::string,
                     std::vector<std::pair<std::size_t, std::size_t>>>
      kernargs;

  ~ihipModule_t() {
    // if (executable.handle) hsa_executable_destroy(executable);
    // if (coReader.handle) hsa_code_object_reader_destroy(coReader);
  }
};

typedef struct ihipModule_t* hipModule_t;

#endif  // SRC_INTERNAL_H