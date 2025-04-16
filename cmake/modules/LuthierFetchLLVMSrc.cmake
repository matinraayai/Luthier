# Attempts to fetch the source code of the LLVM binary found by CMake by downloading it from the repo specified
# by its VCSRevision.h. ${llvm_src_out} will be set to the LLVM source code fetched by this function
function(luthier_fetch_llvm_src llvm_include_dirs llvm_src_out)
    message(STATUS "Fetching LLVM source code it automatically")
    include(FetchContent)
    # Read the contents of VCSRevision.h file from LLVM's include
    # dir. It contains the git repository and the git revision hash of
    # the LLVM binary
    file(READ "${llvm_include_dirs}/llvm/Support/VCSRevision.h" LLVM_VCS_REVISION_CONTENTS)

    # Extract Repository URL
    string(REGEX MATCH "#define LLVM_REPOSITORY \"([^\"]+)\"" LLVM_REPOSITORY "${LLVM_VCS_REVISION_CONTENTS}")
    set(LLVM_REPOSITORY "${CMAKE_MATCH_1}")
    # Extract Git hash
    string(REGEX MATCH "#define LLVM_REVISION \"([^\"]+)\"" LLVM_REVISION "${LLVM_VCS_REVISION_CONTENTS}")
    set(LLVM_REVISION "${CMAKE_MATCH_1}")

    message(STATUS "Determined LLVM repository URL: ${LLVM_REPOSITORY}, revision: ${LLVM_REVISION}")

    message(STATUS "Fetching LLVM source code")

    # Download the source code of LLVM
    FetchContent_Declare(
            LLVM_SOURCE_CODE
            URL ${LLVM_REPOSITORY}/archive/${LLVM_REVISION}.zip
            DOWNLOAD_EXTRACT_TIMESTAMP 1
    )
    FetchContent_MakeAvailable(LLVM_SOURCE_CODE)

    FetchContent_GetProperties(LLVM_SOURCE_CODE SOURCE_DIR LLVM_SOURCE_CODE_SOURCE_DIR)

    # Return the source directory of LLVM
    message(STATUS "LLVM source directory is set to ${LLVM_SOURCE_CODE_SOURCE_DIR}")
    set(${llvm_src_out} ${LLVM_SOURCE_CODE_SOURCE_DIR} PARENT_SCOPE)

    message(STATUS "Fetching the source code of LLVM project - done")
endfunction()