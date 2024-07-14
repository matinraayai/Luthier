#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "common/Cloning.hpp"
#include <doctest/doctest.h>

// #include <rocprofiler-sdk/registration.h>
// #include <rocprofiler-sdk/rocprofiler.h>

int factorial(int number) {
  return number <= 1 ? number : factorial(number - 1) * number;
}

TEST_CASE("testing the factorial function") {
  CHECK(factorial(0) == 1);
  CHECK(factorial(1) == 1);
  CHECK(factorial(2) == 2);
  CHECK(factorial(3) == 6);
  CHECK(factorial(10) == 3628800);
}

/*
// static rocprofiler_client_id_t       my_client_id;
static rocprofiler_client_finalize_t my_fini_func;
static int                           my_tool_data = 1234;

static int tool_ini(rocprofiler_client_finalize_t fini_func, void* tool_data) {
     std::cout << "Rocprofiler initialization was called" << std::endl;
     my_fini_func = fini_func;

     assert(*static_cast<int*>(tool_data) == 1234 && "tool_data is wrong");

     rocprofiler_context_id_t ctx;
     rocprofiler_create_context(&ctx);

     if(int valid_ctx = 0;
        rocprofiler_context_is_valid(ctx, &valid_ctx) != ROCPROFILER_STATUS_SUCCESS ||
        valid_ctx != 0)
     {
         // notify rocprofiler that initialization failed
         // and all the contexts, buffers, etc. created
         // should be ignored
         return -1;
     }

     if(rocprofiler_start_context(ctx) != ROCPROFILER_STATUS_SUCCESS)
     {
         // notify rocprofiler that initialization failed
         // and all the contexts, buffers, etc. created
         // should be ignored
         return -1;
     }

     // no errors
     return 0;
}

void tool_fini(void* tool_data) { 
  std::cout << "Tool was finalized\n";
  assert(*static_cast<int*>(tool_data) == 1234 && "tool_data is wrong");
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id) {
    std::cout << "The configure function was called.\n";
    // create configure data
    // static auto cfg =
    //     rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
    //                                         nullptr, //&tool_ini,
    //                                         &tool_fini,
    //                                         nullptr};

     // only activate if main tool
     if(priority > 0) return nullptr;

     // set the client name
     client_id->name = "ExampleTool";

     // make a copy of client info
     // my_client_id = *client_id;

     // compute major/minor/patch version info
     uint32_t major = version / 10000;
     uint32_t minor = (version % 10000) / 100;
     uint32_t patch = version % 100;

     // print info
     printf("Configuring %s with rocprofiler-sdk (v%u.%u.%u) [%s]\n",
            client_id->name, major, minor, patch, runtime_version);

     // create configure data
     static auto cfg = rocprofiler_tool_configure_result_t{ sizeof(rocprofiler_tool_configure_result_t),
                                                            &tool_ini,
                                                            &tool_fini,
                                                            &my_tool_data };

     // return pointer to configure data
    // return pointer to configure data
    return &cfg;
}
*/
