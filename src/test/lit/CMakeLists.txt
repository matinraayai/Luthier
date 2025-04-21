
include(AddLLVM)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

get_llvm_lit_path(
        LIT_BASE_DIR
        LIT_FILE_NAME
        ALLOW_EXTERNAL
)

add_custom_target(test-lit COMMAND "${LIT_BASE_DIR}/${LIT_FILE_NAME}"
        "${CMAKE_CURRENT_BINARY_DIR}" -v)

add_subdirectory(comgr)