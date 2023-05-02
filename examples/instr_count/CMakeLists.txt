add_library(instr_count SHARED instr_count.cpp inject_funcs.cpp)

target_link_libraries(instr_count PUBLIC sibir)

# set(ENV{LD_LIBRARY_PATH})
# set(ENV{LD_PRELOAD})