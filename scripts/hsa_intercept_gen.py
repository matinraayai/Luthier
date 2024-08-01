# !/usr/bin/env python3
import os

import argparse

from cxxheaderparser.simple import ClassScope, ParsedData, Function
import cxxheaderparser.types as cxx_types
from header_preprocessor import parse_header_file
from typing import *


def parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HSA API interception generation script for Luthier")
    parser.add_argument("--hsa-include-dir", type=str,
                        default="/opt/rocm/include/hsa",
                        help="location of the HSA include directory")
    parser.add_argument("--cpp-callback-save-path", type=str,
                        default="../src/lib/hsa/hsa_intercept.cpp",
                        help="location of where the generated C++ callback file will be saved")
    parser.add_argument("--hpp-structs-save-path", type=str,
                        default="../include/luthier/hsa_trace_api.h",
                        help="location of where the generated C++ header file containing the callback args struct "
                             "and callback enumerators will be saved")
    args = parser.parse_args()
    return args


def parse_hsa_functions(header_files: Iterable[str], defines) -> dict[str, Function]:
    functions = {}
    for header in header_files:
        phf = parse_header_file(header, defines)
        for f in phf.namespace.functions:
            functions[f.name.segments[0].name] = f
    return functions


def get_api_tables(parsed_hsa_api_trace_header: ParsedData, api_table_names: List[str]) -> Dict[str, ClassScope]:
    api_tables = {}
    for cls in parsed_hsa_api_trace_header.namespace.classes:
        typename = cls.class_decl.typename
        if typename.classkey == "struct" and len(typename.segments) == 1 \
                and typename.segments[0].name in api_table_names:
            api_tables[typename.segments[0].name] = cls
    return api_tables


def get_api_table_container(parsed_hsa_api_trace_header: ParsedData, table_container_name: str) -> ClassScope:
    for cls in parsed_hsa_api_trace_header.namespace.classes:
        typename = cls.class_decl.typename
        if typename.classkey == "struct" and len(typename.segments) == 1 \
                and typename.segments[0].name == table_container_name:
            return cls


def main():
    # Name of macros that need to be defined when parsing the headers
    defines = ("__GNUC__", "LITTLEENDIAN_CPU", "_M_X64", "__inline__ __attribute__((always_inline))")
    # Name of the API tables to capture in HSA
    api_table_names = ["CoreApiTable", "AmdExtTable", "ImageExtTable", "FinalizerExtTable"]

    args = parse_and_validate_args()
    # All possible HSA API functions are located in these headers
    hsa_include_files = tuple(os.path.join(args.hsa_include_dir, header_file) for header_file in
                              ("hsa.h", "hsa_ext_amd.h", "hsa_ext_image.h", "hsa_ext_finalize.h", "hsa_api_trace.h"))
    # This returns a Dict that contains all HSA API functions and other functions not of interest to us.
    # It maps the name of the function (e.g. hsa_init) to its cxxheaderparser Function type
    hsa_functions = parse_hsa_functions(hsa_include_files, defines)

    parsed_api_trace_header = parse_header_file(os.path.join(args.hsa_include_dir, "hsa_api_trace.h"), defines)

    # Parse the API tables in hsa_api_trace.h
    api_tables = get_api_tables(parsed_api_trace_header, api_table_names)

    # Parse the HsaApiTableContainer.
    # HsaApiTableContainer is a master struct that contains all the API tables. It is used to save the original
    # APIs before installing the callbacks
    api_table_container = get_api_table_container(parsed_api_trace_header, "HsaApiTableContainer")

    # We need to know which field name in HsaApiTableContainer corresponds to which API table
    # struct type. Here we construct the following in a Dict:
    #  core -> CoreApiTable
    #  amd_ext -> AmdExtTable
    #  finalizer_ext -> FinalizerExtTable
    #  image_ext -> ImageExtTable
    api_table_container_fields = {}
    for table_container_field in api_table_container.fields:
        field_type = table_container_field.type.typename.segments[0].name
        if field_type in api_tables:
            api_table_container_fields[field_type] = table_container_field.name

    """
    The following loops could have been fused together in a single loop, but they're kept separate for 
    better readability
    """

    # Create the HSA_API_EVT_ID enums, enumerating all HSA APIs and EVTs used in Luthier
    callback_enums = [
        f"""/* Generated by {os.path.basename(__file__)}. DO NOT EDIT!*/
#ifndef LUTHIER_HSA_API_TRACE_H
#define LUTHIER_HSA_API_TRACE_H
#include <hsa/hsa_api_trace.h>
#include "luthier/types.h"

namespace luthier::hsa {{

enum ApiEvtID: unsigned int {{
  HSA_API_EVT_ID_FIRST = 0,
"""]

    enum_idx = 0
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        # Mark the beginning of the API Table in the enum
        callback_enums.append(
            f"""  // {api_name} API
""")
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                # Add the function to the enums list
                callback_enums.append(f"""  HSA_API_EVT_ID_{hsa_function_name} = {enum_idx}, 
""")
                enum_idx += 1
    # Handle the kernel launch event separately
    callback_enums.append(f"""  HSA_API_EVT_ID_hsa_queue_packet_submit = {enum_idx},
""")
    # Finalize the API EVT ID Enum
    callback_enums.append(f"""  HSA_API_EVT_ID_LAST = {enum_idx}, 
""")
    callback_enums.append("""};

""")

    # Create a union struct containing the arguments to all HSA APIs and EVTs used in Luthier
    callback_arguments_struct = ["""typedef union {
"""
                                 ]
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                # The hsa function representation parsed by cxxheaderparser
                hsa_function_cxx = hsa_functions[hsa_function_name]
                # Format the args for later
                formatted_params = [p.format() for p in hsa_function_cxx.parameters]
                # Generate the argument struct field
                callback_arguments_struct.append("""  struct {
""")
                for p in formatted_params:
                    callback_arguments_struct.append(f"""    {p};
""")
                callback_arguments_struct.append(f"""  }} {hsa_function_name};
""")
    # Handle the kernel launch event separately
    callback_arguments_struct.append("  struct {\n"
                                     "    luthier::HsaAqlPacket* packets;\n"
                                     "    uint64_t pkt_count;\n"
                                     "    uint64_t user_pkt_index;\n"
                                     "  } hsa_queue_packet_submit;\n")
    callback_arguments_struct.append("""} ApiEvtArgs;

};


#endif
""")

    # Generate the callback functions that will replace the original HSA functions
    callback_defs = [f"""/* Generated by {os.path.basename(__file__)}. DO NOT EDIT! */
#include <hsa/hsa_api_trace.h>
#include "luthier/hsa_trace_api.h"
#include "hsa/hsa_intercept.hpp"
#include "luthier/types.h"
                     
template <>
luthier::hsa::HsaRuntimeInterceptor
    *luthier::Singleton<luthier::hsa::HsaRuntimeInterceptor>::Instance{{nullptr}};

void queueSubmitWriteInterceptor(const void *Packets, uint64_t PktCount,
                                 uint64_t UserPktIndex, void *Data, 
                                 hsa_amd_queue_intercept_packet_writer Writer) {{
  auto &HsaInterceptor = luthier::hsa::HsaRuntimeInterceptor::instance();
  auto [HsaUserCallback, UserCallbackLock] = HsaInterceptor.getUserCallback();
  auto [HsaInternalCallback, InternalCallbackLock] = HsaInterceptor.getInternalCallback();
  auto ApiId = luthier::hsa::HSA_API_EVT_ID_hsa_queue_packet_submit;
  luthier::hsa::ApiEvtArgs Args;
  bool IsUserCallbackEnabled = HsaInterceptor.isUserCallbackEnabled(ApiId);
  bool IsInternalCallbackEnabled = HsaInterceptor.isInternalCallbackEnabled(ApiId);
  if (IsUserCallbackEnabled || IsInternalCallbackEnabled) {{
    // Copy the packets to a non-const buffer
    std::vector<luthier::HsaAqlPacket> ModifiedPackets(
      reinterpret_cast<const luthier::HsaAqlPacket*>(Packets),
      reinterpret_cast<const luthier::HsaAqlPacket*>(Packets) + PktCount);
    Args.hsa_queue_packet_submit.packets = ModifiedPackets.data();
    Args.hsa_queue_packet_submit.pkt_count = PktCount;
    Args.hsa_queue_packet_submit.user_pkt_index = UserPktIndex;
    if (IsUserCallbackEnabled)
      (*HsaUserCallback)(&Args, luthier::API_EVT_PHASE_BEFORE, ApiId);
    if (IsInternalCallbackEnabled)
      (*HsaInternalCallback)(&Args, luthier::API_EVT_PHASE_BEFORE, ApiId);
    // Write the packets to hardware queue
    // Even if the packets are not modified, this call has to be made to ensure
    // the packets are copied to the hardware queue
    Writer(Args.hsa_queue_packet_submit.packets, Args.hsa_queue_packet_submit.pkt_count);
  }} else {{
    Writer(Packets, PktCount);
  }}
}}

static hsa_status_t createInterceptQueue(hsa_agent_t agent, uint32_t size, 
                                         hsa_queue_type32_t type, 
                                         void (* callback)(hsa_status_t status, hsa_queue_t* source, void* data), 
                                         void* data, uint32_t private_segment_size, 
                                         uint32_t group_segment_size, hsa_queue_t** queue) {{
    const auto& AmdExtTable = luthier::hsa::HsaRuntimeInterceptor::instance().getSavedApiTableContainer().amd_ext;
    hsa_status_t Out = AmdExtTable.hsa_amd_queue_intercept_create_fn(agent, 
                                                                     size,
                                                                     type, 
                                                                     callback,
                                                                     data, 
                                                                     private_segment_size,
                                                                     group_segment_size, 
                                                                     queue);
    if (Out != HSA_STATUS_SUCCESS)
      return Out;
    Out = AmdExtTable.hsa_amd_queue_intercept_register_fn(*queue, queueSubmitWriteInterceptor, *queue);
    return Out;
}}
"""]

    for api_name in api_table_names:
        api_table = api_tables[api_name]
        # The field in the API Container this table corresponds to
        api_container_field_name = api_table_container_fields[api_name]
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                # The field of the API Table this function corresponds to (e.g.) hsa_init_fn
                api_table_function_name = f.name
                # The hsa function representation parsed by cxxheaderparser
                hsa_function_cxx = hsa_functions[hsa_function_name]
                # Format the args for later
                formatted_params = [p.format() for p in hsa_functions[hsa_function_name].parameters]
                # Generate the callback
                return_type = hsa_function_cxx.return_type.format()
                callback_defs.append(
                    f"""static {return_type} {hsa_function_name}_wrapper({", ".join(formatted_params)}) {{
  auto& HsaInterceptor = luthier::hsa::HsaRuntimeInterceptor::instance();
  HsaInterceptor.freezeRuntimeApiTable();
  auto ApiId = luthier::hsa::HSA_API_EVT_ID_{hsa_function_name};
  bool IsUserCallbackEnabled = HsaInterceptor.isUserCallbackEnabled(ApiId);
  bool IsInternalCallbackEnabled = HsaInterceptor.isInternalCallbackEnabled(ApiId);
  bool ShouldCallback = IsUserCallbackEnabled || IsInternalCallbackEnabled;
  if (ShouldCallback) {{
""")
                if return_type != "void":
                    callback_defs.append(f"""    {return_type} Out{{}};
""")
                callback_defs.append("""    auto [HsaUserCallback, UserCallbackLock] = HsaInterceptor.getUserCallback();
    auto [HsaInternalCallback, InternalCallbackLock] = HsaInterceptor.getInternalCallback();
    luthier::hsa::ApiEvtArgs Args;
""")
                for p in hsa_function_cxx.parameters:
                    callback_defs.append(f"""    Args.{hsa_function_name}.{p.name} = {p.name};
""")
                callback_defs.append(
                    """    if (IsUserCallbackEnabled)
      (*HsaUserCallback)(&Args, luthier::API_EVT_PHASE_BEFORE, ApiId);
    if (IsInternalCallbackEnabled)
      (*HsaInternalCallback)(&Args, luthier::API_EVT_PHASE_BEFORE, ApiId);
""")
                if hsa_function_name == "hsa_queue_create":  # Create intercept queues instead
                    callback_defs.append(
                        """    Out = createInterceptQueue(Args.hsa_queue_create.agent, 
                                                          Args.hsa_queue_create.size,
                                                          Args.hsa_queue_create.type, 
                                                          Args.hsa_queue_create.callback,
                                                          Args.hsa_queue_create.data, 
                                                          Args.hsa_queue_create.private_segment_size,
                                                          Args.hsa_queue_create.group_segment_size, 
                                                          Args.hsa_queue_create.queue);
""")
                else:
                    callback_defs.append(
                        f'      {"Out =" if return_type != "void" else ""} HsaInterceptor.getSavedApiTableContainer().'
                        f'{api_container_field_name}.{api_table_function_name}(')
                    for i, p in enumerate(hsa_function_cxx.parameters):
                        callback_defs.append(f'Args.{hsa_function_name}.{p.name}')
                        if i != len(hsa_function_cxx.parameters) - 1:
                            callback_defs.append(", ")
                    callback_defs.append(");\n")
                callback_defs.append(
                    f"""    if (IsUserCallbackEnabled)
      (*HsaUserCallback)(&Args, luthier::API_EVT_PHASE_AFTER, ApiId);
    if (IsInternalCallbackEnabled)
      (*HsaInternalCallback)(&Args, luthier::API_EVT_PHASE_AFTER, ApiId);
    {"return Out;" if return_type != "void" else ""}
  }}
  else {{
""")
                if hsa_function_name == "hsa_queue_create":  # Create intercept queues instead
                    callback_defs.append(
                        """    return createInterceptQueue(
                                                           agent, 
                                                           size,
                                                           type, 
                                                           callback,
                                                           data, 
                                                           private_segment_size,
                                                           group_segment_size, 
                                                           queue);
  }
""")
                else:
                    callback_defs.append(f'    return HsaInterceptor.getSavedApiTableContainer().'
                                         f'{api_container_field_name}.{api_table_function_name}(')
                    for i, p in enumerate(hsa_function_cxx.parameters):
                        callback_defs.append(f'{p.name}')
                        if i != len(hsa_function_cxx.parameters) - 1:
                            callback_defs.append(", ")
                    callback_defs.append(""");
  }
""")
                callback_defs.append("""}

""")

    # Generate wrapper switch functions, which can switch between the real and the wrapper version of each API function
    wrapper_switches = []
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        # The field in the API Container this table corresponds to
        api_container_field_name = api_table_container_fields[api_name]
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                # The field of the API Table this function corresponds to (e.g.) hsa_init_fn
                api_table_function_name = f.name
                # Handle special case for hsa_queue_create; All queues created by the application will be wrapped
                # inside an intercept queue
                real_function = "createInterceptQueue" if hsa_function_name == "hsa_queue_create" else \
                    f"SavedApiTable.{api_container_field_name}.{api_table_function_name}"
                wrapper_switches.append(
                    # @formatter:off
f"""static void switch_{hsa_function_name}_wrapper(HsaApiTable *RuntimeApiTable, 
        const HsaApiTableContainer &SavedApiTable, bool State) {{
  if (State)
    RuntimeApiTable->{api_container_field_name}_->{api_table_function_name} = {hsa_function_name}_wrapper;
  else
    RuntimeApiTable->{api_container_field_name}_->{api_table_function_name} = {real_function};
}}\n\n"""
                    # @formatter:off
                )

    # Generate wrapper installation functions, which checks whether the wrapper function or the real function
    # is currently installed over the API runtime table
    wrapper_checks = []
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        # The field in the API Container this table corresponds to
        api_container_field_name = api_table_container_fields[api_name]
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                # The field of the API Table this function corresponds to (e.g.) hsa_init_fn
                api_table_function_name = f.name
                wrapper_checks.append(
                    # @formatter:off
                    f"""static bool is_{hsa_function_name}_wrapper_installed(HsaApiTable *RuntimeApiTable) {{
  return RuntimeApiTable->{api_container_field_name}_->{api_table_function_name} == {hsa_function_name}_wrapper;
}}\n\n"""
                    # @formatter:off
                )

    # Generate a static const mapping between the API IDs and the switch wrapper functions
    # This is used to switch between wrappers and reals for each the API IDs
    switch_functions_map = ["static const llvm::DenseMap<luthier::hsa::ApiEvtID, std::function<void(HsaApiTable *, "
                            "const HsaApiTableContainer &, bool)>> HsaWrapperSwitchFunctionsMap {\n"]
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                switch_functions_map.append(f"  {{luthier::hsa::HSA_API_EVT_ID_{hsa_function_name}, "
                                            f"switch_{hsa_function_name}_wrapper}},\n")
    switch_functions_map.append("};")

    # Generate a static const mapping between the API IDs and the wrapper installation check functions
    # This is used to confirm wrapper installation status of each API ID
    wrapper_check_functions_map = ["static const llvm::DenseMap<luthier::hsa::ApiEvtID, "
                                   "std::function<bool(HsaApiTable *)>> "
                                   "HsaWrapperInstallationCheckFunctionsMap {\n"]
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                wrapper_check_functions_map.append(f"  {{luthier::hsa::HSA_API_EVT_ID_{hsa_function_name}, "
                                                   f"is_{hsa_function_name}_wrapper_installed}},\n")
    wrapper_check_functions_map.append("};")

    # Generate enable/disable callback functions for the interceptor
    enable_disable_funcs = f"""
bool luthier::hsa::HsaRuntimeInterceptor::enableUserCallback(ApiEvtID Op) {{
  if (!IsRuntimeApiTableFrozen)
    HsaWrapperSwitchFunctionsMap.at(Op)(RuntimeApiTable, SavedRuntimeApiTable, true);
  else if (!HsaWrapperInstallationCheckFunctionsMap.at(Op)(RuntimeApiTable))
    return false;
  EnabledUserOps.insert(Op);
  return true;
}}

void luthier::hsa::HsaRuntimeInterceptor::disableUserCallback(ApiEvtID Op) {{
  EnabledUserOps.erase(Op);
  if (!IsRuntimeApiTableFrozen && !EnabledInternalOps.contains(Op))
    HsaWrapperSwitchFunctionsMap.at(Op)(RuntimeApiTable, SavedRuntimeApiTable, false);
}}

bool luthier::hsa::HsaRuntimeInterceptor::enableInternalCallback(ApiEvtID Op) {{
  if (!IsRuntimeApiTableFrozen)
    HsaWrapperSwitchFunctionsMap.at(Op)(RuntimeApiTable, SavedRuntimeApiTable, true);
  else if (!HsaWrapperInstallationCheckFunctionsMap.at(Op)(RuntimeApiTable))
    return false;
  EnabledInternalOps.insert(Op);
  return true;
}}

void luthier::hsa::HsaRuntimeInterceptor::disableInternalCallback(ApiEvtID Op) {{
  EnabledInternalOps.erase(Op);
  if (!EnabledUserOps.contains(Op))
    HsaWrapperSwitchFunctionsMap.at(Op)(RuntimeApiTable, SavedRuntimeApiTable, false);
}}"""

    # Write the generated code into appropriate files
    with open(args.hpp_structs_save_path, "w") as f:
        f.writelines("// NOLINTBEGIN\n")
        f.writelines(callback_enums)
        f.write("\n")
        f.writelines(callback_arguments_struct)
        f.writelines("// NOLINTEND\n")

    with open(args.cpp_callback_save_path, "w") as f:
        f.writelines("// NOLINTBEGIN\n")
        f.writelines(callback_defs)
        f.write("\n")
        f.writelines(wrapper_switches)
        f.write("\n")
        f.writelines(wrapper_checks)
        f.write("\n")
        f.writelines(switch_functions_map)
        f.write("\n")
        f.writelines(wrapper_check_functions_map)
        f.write("\n")
        f.writelines(enable_disable_funcs)
        f.write("\n")
        f.writelines("// NOLINTEND\n")


if __name__ == "__main__":
    main()
