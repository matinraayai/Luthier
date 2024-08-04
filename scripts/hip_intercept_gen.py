# !/usr/bin/env python3
import os

import argparse

import cxxheaderparser.types
from cxxheaderparser.simple import ClassScope, ParsedData
import cxxheaderparser.types as cxx_types
from header_preprocessor import parse_header_file
from typing import *


def parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HIP API interception generation script for Luthier")
    parser.add_argument("--hip-api-trace-path", type=str,
                        default="/opt/rocm/include/hip/amd_detail/hip_api_trace.hpp",
                        help="directory of the HIP API Trace header file")
    parser.add_argument("--hpp-structs-save-path", type=str,
                        default="../include/luthier/hip_trace_api.h",
                        help="location of where the generated C++ header file containing the callback args struct "
                             "and api id enumerators will be saved")
    parser.add_argument("--cpp-compiler-implementation-save-path", type=str,
                        default="../src/lib/hip/hip_compiler_intercept.cpp",
                        help="location of where the generated C++ implementation for the HIP compiler "
                             "API will be saved")
    parser.add_argument("--cpp-runtime-implementation-save-path", type=str,
                        default="../src/lib/hip/hip_runtime_intercept.cpp",
                        help="location of where the generated C++ implementation for the HIP runtime"
                             " API will be saved")
    args = parser.parse_args()
    return args


defines = ("ROCPROFILER_EXTERN_C_INIT", "ROCPROFILER_EXTERN_C_FINI")


def parse_hip_functions(phf: ParsedData) -> dict[str, cxx_types.FunctionType]:
    functions = {}
    for f in phf.namespace.typedefs:
        function_name = f.name.removeprefix("t_")
        functions[function_name] = f.type.ptr_to
    return functions


def is_param_dim3_type(p: cxx_types.Parameter) -> bool:
    if isinstance(p.type, cxx_types.Type):
        return p.type.typename.segments[0].name == "dim3"
    else:
        return False


def get_api_tables(parsed_hip_api_trace_header: ParsedData, api_table_names: List[str]) -> Dict[str, ClassScope]:
    api_tables = {}
    for cls in parsed_hip_api_trace_header.namespace.classes:
        typename = cls.class_decl.typename
        if typename.classkey == "struct" and len(typename.segments) == 1 \
                and typename.segments[0].name in api_table_names:
            api_tables[typename.segments[0].name] = cls
    return api_tables


def generate_wrapper_functions(api_table_struct: ClassScope,
                               functions: dict[str, cxxheaderparser.types.FunctionType],
                               interceptor_header_file_name: str,
                               interceptor_name: str,
                               api_name: str):
    wrapper_defs = [
        # @formatter:off
f"""/* Generated by {os.path.basename(__file__)}. DO NOT EDIT! */
#include "{interceptor_header_file_name}"
#include "luthier/types.h"
#include "luthier/hip_trace_api.h"
    
"""
        # @formatter:on
    ]
    for f in api_table_struct.fields:
        # look for functions in the API Tables, not fields (e.g. size)
        # function fields in the API table are defined as pointers to typedefs of their target HIP function
        if isinstance(f.type.typename, cxx_types.PQName) and f.type.typename.segments[0].name != 'size_t':

            # The field of the API Table this function corresponds to (e.g.) hipApiName_fn
            function_api_table_field_name = f.name
            # Name of the function (e.g. hipApiName)
            function_name = function_api_table_field_name.removesuffix("_fn")
            # The hip function representation parsed by cxxheaderparser
            hip_function_cxx: cxx_types.FunctionType = functions[function_name]

            # Format the args for later
            formatted_params = [p.format() for p in hip_function_cxx.parameters]
            # Generate the callback
            return_type = hip_function_cxx.return_type.format()

            wrapper_defs.append(
                # @formatter:off
f"""static {return_type} {function_name}_wrapper({", ".join(formatted_params)}) {{
  auto& HipInterceptor = luthier::hip::{interceptor_name}::instance();
  HipInterceptor.freezeRuntimeApiTable();
  auto ApiId = luthier::hip::HIP_{api_name.upper()}_API_EVT_ID_{function_name};
  bool IsUserCallbackEnabled = HipInterceptor.isUserCallbackEnabled(ApiId);
  bool IsInternalCallbackEnabled = HipInterceptor.isInternalCallbackEnabled(ApiId);
  bool ShouldCallback = IsUserCallbackEnabled || IsInternalCallbackEnabled;
  if (ShouldCallback) {{
"""
                # @formatter:on
            )
            if return_type != "void":
                wrapper_defs.append(f"""    {return_type} Out{{}};\n""")
            wrapper_defs.append(
                # @formatter:off
"""    auto [HipUserCallback, UserCallbackLock] = HipInterceptor.getUserCallback();
    auto [HipInternalCallback, InternalCallbackLock] = HipInterceptor.getInternalCallback();
    luthier::hip::ApiEvtArgs Args;
"""
                # @formatter:on
            )
            for param in hip_function_cxx.parameters:
                is_dim3_type = is_param_dim3_type(param)
                # Special handling of HIP dim3 arguments, to convert them to Luthier dim3
                if is_dim3_type:
                    wrapper_defs.append(
                        # @formatter:off
f"""    Args.{function_name}.{param.name}.x = {param.name}.x;
    Args.{function_name}.{param.name}.y = {param.name}.y;
    Args.{function_name}.{param.name}.z = {param.name}.z;                    
"""
                        # @formatter:on
                    )
                else:
                    wrapper_defs.append(
                        f"    Args.{function_name}.{param.name} = {param.name};\n")
            wrapper_defs.append(
                # @formatter:off
"""    if (IsUserCallbackEnabled)
      (*HipUserCallback)(&Args, luthier::API_EVT_PHASE_BEFORE, ApiId);
    if (IsInternalCallbackEnabled)
      (*HipInternalCallback)(&Args, luthier::API_EVT_PHASE_BEFORE, ApiId);
"""
                # @formatter:on
            )
            wrapper_defs.append(
                f'{"Out =" if return_type != "void" else ""} HipInterceptor.getSavedApiTableContainer().'
                f"{function_api_table_field_name}(")

            for i, param in enumerate(hip_function_cxx.parameters):
                is_dim3_type = is_param_dim3_type(param)
                # Dim3 arguments must be passed as an initializer list for conversion to dim3
                if is_dim3_type:
                    wrapper_defs.append(f"*reinterpret_cast<dim3*>(&Args.{function_name}.{param.name})")
                else:
                    wrapper_defs.append(f'Args.{function_name}.{param.name}')
                if i != len(hip_function_cxx.parameters) - 1:
                    wrapper_defs.append(", ")
            wrapper_defs.append(");\n")
            wrapper_defs.append(
                # @formatter:off
f"""    if (IsUserCallbackEnabled)
      (*HipUserCallback)(&Args, luthier::API_EVT_PHASE_AFTER, ApiId);
    if (IsInternalCallbackEnabled)
      (*HipInternalCallback)(&Args, luthier::API_EVT_PHASE_AFTER, ApiId);
    {"return Out;" if return_type != "void" else ""}
  }}
  else {{
"""
                # @formatter:on
            )
            wrapper_defs.append(
                f"    return HipInterceptor.getSavedApiTableContainer().{function_api_table_field_name}(")
            for i, param in enumerate(hip_function_cxx.parameters):
                wrapper_defs.append(param.name)
                if i != len(hip_function_cxx.parameters) - 1:
                    wrapper_defs.append(", ")
            wrapper_defs.append(""");
  }
""")
            wrapper_defs.append("""}

""")

    return wrapper_defs


def generate_api_id_enums(api_tables, api_names, api_table_names):
    # Create the HIP_API_EVT_ID enums, enumerating all HIP APIs
    api_id_enums = [
        f"""/* Generated by {os.path.basename(__file__)}. DO NOT EDIT!*/
#ifndef LUTHIER_HIP_API_TRACE_H
#define LUTHIER_HIP_API_TRACE_H
#include <hip/amd_detail/hip_api_trace.hpp>
#include "luthier/types.h"

namespace luthier::hip {{

struct Dim3 {{
  uint32_t x;
  uint32_t y;
  uint32_t z;
}};
  
enum ApiEvtID : unsigned int {{
"""]
    next_enum_idx = 0
    for (api_name, api_table_name) in zip(api_names, api_table_names):
        api_table = api_tables[api_table_name]
        # Mark the beginning of the API Table in the enum
        api_id_enums.append(
            f"""  // {api_name} API
  HIP_{api_name.upper()}_API_EVT_ID_FIRST = {next_enum_idx},
""")
        for f in api_table.fields:
            # Skip the API table size field
            if f.name == "size" or f.type.typename.segments[0].name == "size_t":
                continue
            # Name of the HIP function is the name of the field minus the "_fn" suffix
            function_name = f.name.removesuffix("_fn")
            # Add the function to the enums list
            api_id_enums.append(f"""  HIP_{api_name.upper()}_API_EVT_ID_{function_name} = {next_enum_idx}, 
""")
            next_enum_idx += 1
        # Finalize the API EVT ID Enum
        api_id_enums.append(f"""  HIP_{api_name.upper()}_API_EVT_ID_LAST = {next_enum_idx - 1}, 
""")
    api_id_enums.append("""
};
""")
    return api_id_enums


def generate_wrapper_switch_functions(api_table_struct: ClassScope,
                                      api_table_name: str):
    # Generate wrapper switch functions, which can switch between the real and the wrapper version of each API function
    wrapper_switches = []
    for f in api_table_struct.fields:
        # look for functions in the API Tables, not fields (e.g. size)
        # function fields in the API table are defined as pointers to typedefs of their target HIP function
        if isinstance(f.type.typename, cxx_types.PQName) and f.type.typename.segments[0].name != 'size_t':
            # The field of the API Table this function corresponds to (e.g.) hipApiName_fn
            function_api_table_field_name = f.name
            # Name of the function (e.g. hipApiName)
            function_name = function_api_table_field_name.removesuffix("_fn")

            wrapper_switches.append(
                # @formatter:off
                f"""static void switch_{function_name}_wrapper({api_table_name} *RuntimeApiTable, 
    const {api_table_name} &SavedApiTable, bool State) {{
  if (State)
    RuntimeApiTable->{function_api_table_field_name} = {function_name}_wrapper;
  else
    RuntimeApiTable->{function_api_table_field_name} = SavedApiTable.{function_api_table_field_name};
}}\n\n"""
                # @formatter:off
            )
    return wrapper_switches


def generate_wrapper_check_functions(api_table_struct: ClassScope,
                                     api_table_name: str):
    # Generate wrapper installation functions, which checks whether the wrapper function or the real function
    # is currently installed over the API runtime table
    wrapper_checks = []
    for f in api_table_struct.fields:
        # look for functions in the API Tables, not fields
        # function fields in the API table are defined as pointers to typedefs of their target HIP function
        if isinstance(f.type.typename, cxx_types.PQName) and f.type.typename.segments[0].name != 'size_t':
            # The field of the API Table this function corresponds to (e.g.) hipApiName_fn
            function_api_table_field_name = f.name
            # Name of the function (e.g. hipApiName)
            function_name = function_api_table_field_name.removesuffix("_fn")
            wrapper_checks.append(
                # @formatter:off
                f"""static bool is_{function_name}_wrapper_installed({api_table_name} *RuntimeApiTable) {{
  return RuntimeApiTable->{function_api_table_field_name} == {function_name}_wrapper;
}}\n\n"""
                # @formatter:on
            )
    return wrapper_checks


def generate_wrapper_switch_functions_map(api_name, api_table_struct: ClassScope,
                                          api_table_name: str):
    # Generate a static const mapping between the API IDs and the switch wrapper functions
    # This is used to switch between wrappers and reals for each the API IDs
    switch_functions_map = [
        f"static constexpr void (*Hip{api_name}WrapperSwitchFunctionsMap[])"
        f"({api_table_name} *, const {api_table_name} &, bool) {{\n"]
    for f in api_table_struct.fields:
        # look for functions in the API Tables, not fields
        # function fields in the API table are defined as pointers to typedefs of their target HIP function
        if isinstance(f.type.typename, cxx_types.PQName) and f.type.typename.segments[0].name != 'size_t':
            # The field of the API Table this function corresponds to (e.g.) hipApiName_fn
            function_api_table_field_name = f.name
            # Name of the function (e.g. hipApiName)
            function_name = function_api_table_field_name.removesuffix("_fn")
            switch_functions_map.append(f"  switch_{function_name}_wrapper,\n")
    switch_functions_map.append("};")
    return switch_functions_map


def generate_wrapper_check_functions_map(api_name, api_table_struct: ClassScope,
                                         api_table_name: str):
    # Generate a static const mapping between the API IDs and the wrapper installation check functions
    # This is used to confirm wrapper installation status of each API
    wrapper_check_functions_map = [
        f"static constexpr bool (*Hip{api_name}WrapperInstallationCheckFunctionsMap[]) ({api_table_name} *) {{\n"]
    for f in api_table_struct.fields:
        # look for functions in the API Tables, not fields
        # function fields in the API table are defined as pointers to typedefs of their target HIP function
        if isinstance(f.type.typename, cxx_types.PQName) and f.type.typename.segments[0].name != 'size_t':
            # The field of the API Table this function corresponds to (e.g.) hipApiName_fn
            function_api_table_field_name = f.name
            # Name of the function (e.g. hipApiName)
            function_name = function_api_table_field_name.removesuffix("_fn")
            wrapper_check_functions_map.append(f"  is_{function_name}_wrapper_installed,\n")
    wrapper_check_functions_map.append("};")
    return wrapper_check_functions_map


def generate_wrapper_enable_disable_functions(api_name,
                                              interceptor_name: str):
    # Generate enable/disable callback functions for the interceptor
    enable_disable_funcs = f"""
llvm::Error luthier::hip::{interceptor_name}::enableUserCallback(ApiEvtID Op) {{
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Status != WAITING_FOR_API_TABLE));
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST) <= static_cast<unsigned int>(Op))); 
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_LAST) >= static_cast<unsigned int>(Op)));
  unsigned int OpIdx = static_cast<unsigned int>(Op) - 
                       static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST);
  if (Status != FROZEN)
    Hip{api_name}WrapperSwitchFunctionsMap[OpIdx](RuntimeApiTable, SavedRuntimeApiTable, true);
  else {{
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(!Hip{api_name}WrapperInstallationCheckFunctionsMap[OpIdx](RuntimeApiTable)));
  }};
  EnabledUserOps.insert(Op);
  return llvm::Error::success();
}}

llvm::Error luthier::hip::{interceptor_name}::disableUserCallback(ApiEvtID Op) {{
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Status != WAITING_FOR_API_TABLE));
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST) <= static_cast<unsigned int>(Op))); 
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_LAST) >= static_cast<unsigned int>(Op)));
  EnabledUserOps.erase(Op);
  if (Status != FROZEN && !EnabledInternalOps.contains(Op)) {{
    unsigned int OpIdx = static_cast<unsigned int>(Op) - 
                   static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST);
    Hip{api_name}WrapperSwitchFunctionsMap[OpIdx](RuntimeApiTable, SavedRuntimeApiTable, false);
  }}
    
  return llvm::Error::success();
}}

llvm::Error luthier::hip::{interceptor_name}::enableInternalCallback(ApiEvtID Op) {{
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Status != WAITING_FOR_API_TABLE));
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST) <= static_cast<unsigned int>(Op))); 
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_LAST) >= static_cast<unsigned int>(Op)));
  unsigned int OpIdx = static_cast<unsigned int>(Op) - 
                       static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST);
  if (Status != FROZEN)
    Hip{api_name}WrapperSwitchFunctionsMap[OpIdx](RuntimeApiTable, SavedRuntimeApiTable, true);
  else {{
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(!Hip{api_name}WrapperInstallationCheckFunctionsMap[OpIdx](RuntimeApiTable)));
  }};
  EnabledInternalOps.insert(Op);
  return llvm::Error::success();
}}

llvm::Error luthier::hip::{interceptor_name}::disableInternalCallback(ApiEvtID Op) {{
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(Status != WAITING_FOR_API_TABLE));
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST) <= static_cast<unsigned int>(Op))); 
  LUTHIER_RETURN_ON_ERROR(
     LUTHIER_ARGUMENT_ERROR_CHECK(
        static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_LAST) >= static_cast<unsigned int>(Op)));
  EnabledInternalOps.erase(Op);
  if (Status != FROZEN && !EnabledInternalOps.contains(Op)) {{
    unsigned int OpIdx = static_cast<unsigned int>(Op) - 
                   static_cast<unsigned int>(HIP_{api_name.upper()}_API_EVT_ID_FIRST);
    Hip{api_name}WrapperSwitchFunctionsMap[OpIdx](RuntimeApiTable, SavedRuntimeApiTable, false);
  }}
  return llvm::Error::success();
}}"""
    return enable_disable_funcs


def generate_api_args_struct(api_tables, api_names, api_table_names, hip_functions):
    # Create a union struct containing the arguments to all HIP APIs used in Luthier
    callback_arguments_struct = ["""typedef union {
"""
                                 ]
    for (api_name, api_table_name) in zip(api_names, api_table_names):
        api_table = api_tables[api_table_name]
        for f in api_table.fields:
            # Skip the API table size field
            if f.name == "size" or f.type.typename.segments[0].name == "size_t":
                continue
            # Name of the HIP function is the name of the field minus the "_fn" suffix
            function_name = f.name.removesuffix("_fn")
            # The hip function representation parsed by cxxheaderparser
            hip_function_cxx = hip_functions[function_name]
            # Format the args for later
            formatted_params = []
            for param in hip_function_cxx.parameters:
                formatted_param = param.format()
                # Dim3 structs need to be converted to the plain Dim3 in Luthier since they don't have a trivial
                # destructor
                if is_param_dim3_type(param):
                    formatted_param = formatted_param.replace("dim3", "luthier::hip::Dim3")
                # hipFunction_t cannot be const in the arg struct
                elif isinstance(param.type, cxx_types.Type) and \
                        param.type.const and \
                        param.type.typename.segments[0].name == "hipFunction_t":
                    formatted_param = formatted_param.replace("const hipFunction_t", "hipFunction_t")
                formatted_params.append(formatted_param)
            # Generate the argument struct field
            callback_arguments_struct.append("""  struct {
""")
            for p in formatted_params:
                callback_arguments_struct.append(f"""    {p};
""")
            callback_arguments_struct.append(f"""  }} {function_name};
""")
    callback_arguments_struct.append("""} ApiEvtArgs;

};



""")
    return callback_arguments_struct


def generate_api_id_dense_map_info():
    api_id_dense_map_info = [f"""namespace llvm {{

template <> struct DenseMapInfo<luthier::hip::ApiEvtID> {{
  static inline luthier::hip::ApiEvtID getEmptyKey() {{                                        
    return (luthier::hip::ApiEvtID)(DenseMapInfo<std::underlying_type_t<luthier::hip::ApiEvtID>>::getEmptyKey());
  }}
  
  static inline luthier::hip::ApiEvtID getTombstoneKey() {{
  return (luthier::hip::ApiEvtID)(DenseMapInfo<std::underlying_type_t<luthier::hip::ApiEvtID>>::getTombstoneKey());
  }}

  static unsigned getHashValue(const luthier::hip::ApiEvtID &P) {{
  return DenseMapInfo<std::underlying_type_t<luthier::hip::ApiEvtID>>::getHashValue(
    static_cast<std::underlying_type<luthier::hip::ApiEvtID>::type>(P));
  }}
  
  static bool isEqual(const luthier::hip::ApiEvtID &LHS, const luthier::hip::ApiEvtID &RHS) {{
    return static_cast<std::underlying_type<luthier::hip::ApiEvtID>::type>(LHS) ==
           static_cast<std::underlying_type<luthier::hip::ApiEvtID>::type>(RHS);
  }}
  
}};

}};

#endif
"""]
    return api_id_dense_map_info


def main():
    args = parse_and_validate_args()

    # Name of the API tables to capture in HIP
    api_table_names = ["HipCompilerDispatchTable", "HipDispatchTable"]
    # Name of the HIP APIs
    api_names = ["Compiler", "Runtime"]
    # Name of the header files for each API
    interceptor_header_file_names = ["hip/HipCompilerApiInterceptor.hpp", "hip/HipRuntimeApiInterceptor.hpp"]
    # Name of the interceptor singleton classes
    interceptor_names = ["HipCompilerApiInterceptor", "HipRuntimeApiInterceptor"]
    # Path to where the .cpp file of each interceptor will be written to
    interceptor_cpp_paths = [args.cpp_compiler_implementation_save_path, args.cpp_runtime_implementation_save_path]

    # Parse the hip_api_trace.hpp, which contains typedefs for each function in the API tables + the API tables
    # themselves
    parsed_hip_api_trace_header = parse_header_file(args.hip_api_trace_path, defines)
    # Map between the name of the function (e.g. hip_init) to its cxxheaderparser Function type
    hip_functions = parse_hip_functions(parsed_hip_api_trace_header)

    # Parse the API tables in hip_api_trace.hpp
    api_tables = get_api_tables(parsed_hip_api_trace_header, api_table_names)

    # Generate the hip_api_trace.h file for Luthier's public API
    api_id_enums = generate_api_id_enums(api_tables, api_names, api_table_names)
    callback_args_struct = generate_api_args_struct(api_tables, api_names, api_table_names, hip_functions)
    dense_map_info = generate_api_id_dense_map_info()

    with open(args.hpp_structs_save_path, "w") as f:
        f.writelines("// NOLINTBEGIN\n")
        f.writelines(api_id_enums)
        f.writelines(callback_args_struct)
        f.writelines(dense_map_info)
        f.writelines("\n// NOLINTEND\n")

    for api_table_name, api_name, api_table, interceptor_header_file_name, interceptor_name, interceptor_cpp_path in (
            zip(api_table_names, api_names, [api_tables["HipCompilerDispatchTable"], api_tables["HipDispatchTable"]],
                interceptor_header_file_names, interceptor_names,
                interceptor_cpp_paths)):
        cpp_file_contents = generate_wrapper_functions(api_table,
                                                       hip_functions,
                                                       interceptor_header_file_name,
                                                       interceptor_name,
                                                       api_name)
        cpp_file_contents += generate_wrapper_switch_functions(api_table,
                                                               api_table_name)

        cpp_file_contents += generate_wrapper_check_functions(api_table, api_table_name)

        cpp_file_contents += generate_wrapper_switch_functions_map(api_name,
                                                                   api_table,
                                                                   api_table_name)

        cpp_file_contents += generate_wrapper_check_functions_map(api_name,
                                                                  api_table,
                                                                  api_table_name)

        cpp_file_contents += generate_wrapper_enable_disable_functions(api_name,
                                                                       interceptor_name)

        with open(interceptor_cpp_path, "w") as f:
            f.writelines("// NOLINTBEGIN\n")
            f.writelines(cpp_file_contents)
            f.writelines("\n// NOLINTEND\n")


if __name__ == "__main__":
    main()
