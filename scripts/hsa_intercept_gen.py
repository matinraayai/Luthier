# !/usr/bin/env python3
from __future__ import print_function

import os
import io
import re
import sys

import argparse

from cxxheaderparser.simple import parse_string, ClassScope, ParsedData, Function
import cxxheaderparser.types as cxx_types
import cxxheaderparser
from header_preprocessor import ROCmPreprocessor
from typing import *


def parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HSA API interception generation script for Luthier; Originally used by AMD in "
                                     "the Roctracer project")
    parser.add_argument("--hsa-include-dir", type=str,
                        default="/opt/rocm/include/hsa",
                        help="location of the HSA include directory")
    parser.add_argument("--cpp-callback-save-path", type=str,
                        default="../src/hsa_intercept.cpp",
                        help="location of where the generated C++ callback file will be saved")
    parser.add_argument("--hpp-structs-save-path", type=str,
                        default="../include/hsa_trace_api.hpp",
                        help="location of where the generated C++ header file containing the callback args struct "
                             "and callback enumerators will be saved")
    args = parser.parse_args()
    return args


def parse_header_file(header_file: str) -> ParsedData:
    preprocessor = ROCmPreprocessor()
    preprocessor.line_directive = None
    preprocessor.passthru_unfound_includes = True
    preprocessor.passthru_includes = re.compile(r".*")
    preprocessor.define("__GNUC__")
    preprocessor.define("LITTLEENDIAN_CPU")
    preprocessor.define("_M_X64")
    api_tables = {}
    with open(header_file, 'r') as hf:
        preprocessor.parse(hf)
        str_io = io.StringIO()
        preprocessor.write(str_io)
    preprocessed_header = str_io.getvalue()
    parsed = parse_string(preprocessed_header)
    str_io.close()
    return parsed


def parse_hsa_functions(header_files: Iterable[str]) -> dict[str, Function]:
    functions = {}
    for header in header_files:
        phf = parse_header_file(header)
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
    # Name of the API tables to capture in HSA
    api_table_names = ["CoreApiTable", "AmdExtTable", "ImageExtTable", "FinalizerExtTable"]

    args = parse_and_validate_args()
    # All possible HSA API functions are located in these headers
    hsa_include_files = tuple(os.path.join(args.hsa_include_dir, header_file) for header_file in
                              ("hsa.h", "hsa_ext_amd.h", "hsa_ext_image.h", "hsa_ext_finalize.h", "hsa_api_trace.h"))
    # This returns a Dict that contains all HSA API functions and other functions not of interest to us.
    # It maps the name of the function (e.g. hsa_init) to its cxxheaderparser Function type
    hsa_functions = parse_hsa_functions(hsa_include_files)

    parsed_api_trace_header = parse_header_file(os.path.join(args.hsa_include_dir, "hsa_api_trace.h"))

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
        f"""/* Generated by {os.path.basename(__file__)}. DO NOT EDIT!*/"
#include <hsa/hsa_api_trace.h>
#include "luthier_types.h"

enum hsa_api_evt_id_t: unsigned int {{
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
                # The field of the API Table this function corresponds to (e.g.) hsa_init_fn
                api_table_function_name = f.name
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
    callback_arguments_struct = ["typedef union {\n"]
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
                callback_arguments_struct.append("\tstruct {\n")
                for p in formatted_params:
                    callback_arguments_struct.append(f"\t\t{p};\n")
                callback_arguments_struct.append(f"\t}} {hsa_function_name};\n")
    # Handle the kernel launch event separately
    callback_arguments_struct.append("\tstruct {\n"
                                     "\t\tluthier::HsaAqlPacket* packets;\n"
                                     "\t\tuint64_t pkt_count;\n"
                                     "\t\tuint64_t user_pkt_index;\n"
                                     "\t} hsa_queue_packet_submit;\n")
    callback_arguments_struct.append("} hsa_api_evt_args_t;\n\n")

    # Generate functions that install the wrapper callbacks when HSA is loaded
    wrapper_install_defs = []
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        # First create the wrapper install function's prototype
        wrapper_install_defs.append(
            f'void luthier::hsa::Interceptor::install{api_name}Wrappers({api_name} *Table) {{\n')
        wrapper_install_defs.append(f'\tSavedTables.{api_table_container_fields[api_name]} = *Table;\n')
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                # Name of the hsa function (e.g.) hsa_init
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                # The field of the API Table this function corresponds to (e.g.) hsa_init_fn
                api_table_function_name = f.name
                # Install the callback
                wrapper_install_defs.append(f'\tTable->{api_table_function_name} = {hsa_function_name};\n')
        wrapper_install_defs.append("};\n")

    # Generate the callback functions that will replace the original HSA functions
    callback_defs = [f"/* Generated by {os.path.basename(__file__)}. DO NOT EDIT!*/\n",
                     "#include <hsa/hsa_api_trace.h>\n",
                     '#include "hsa_trace_api.hpp"\n',
                     '#include "hsa_intercept.hpp"\n',
                     '#include "luthier_types.h"\n\n']
    # Create the Queue Packet Submit Event first for later
    callback_defs.append("void queueSubmitWriteInterceptor(const void *Packets, uint64_t PktCount, "
                         "uint64_t UserPktIndex, void *Data, hsa_amd_queue_intercept_packet_writer Writer) {\n"
                         "\tauto &HsaInterceptor = luthier::hsa::Interceptor::instance();\n"
                         "\tauto &HsaUserCallback = HsaInterceptor.getUserCallback();\n"
                         "\tauto &HsaInternalCallback = HsaInterceptor.getInternalCallback();\n"
                         "\tauto ApiId = HSA_API_EVT_ID_hsa_queue_packet_submit;\n"
                         "\thsa_api_evt_args_t Args;\n"
                         "\tbool IsUserCallbackEnabled = HsaInterceptor.isUserCallbackEnabled(ApiId);\n"
                         "\tbool IsInternalCallbackEnabled = HsaInterceptor.isInternalCallbackEnabled(ApiId);\n"
                         "\tif (IsUserCallbackEnabled || IsInternalCallbackEnabled) {\n"
                         "\t// Copy the packets to a non-const buffer\n"
                         "\tstd::vector<luthier::HsaAqlPacket> ModifiedPackets(reinterpret_cast<const "
                         "luthier::HsaAqlPacket*>(Packets),"
                         "reinterpret_cast<const luthier::HsaAqlPacket*>(Packets) + PktCount);\n"
                         "\tArgs.hsa_queue_packet_submit.packets = ModifiedPackets.data();\n"
                         "\tArgs.hsa_queue_packet_submit.pkt_count = PktCount;\n"
                         "\tArgs.hsa_queue_packet_submit.user_pkt_index = UserPktIndex;\n"
                         "\tif (IsUserCallbackEnabled)\n"
                         "\t\tHsaUserCallback(&Args, luthier::API_EVT_PHASE_ENTER, ApiId);\n"
                         "\tif (IsInternalCallbackEnabled)\n"
                         "\t\tHsaInternalCallback(&Args, luthier::API_EVT_PHASE_ENTER, ApiId, nullptr);\n"
                         "\t// Write the packets to hardware queue\n"
                         "\t// Even if the packets are not modified, this call has to be made to ensure\n"
                         "\t// the packets are copied to the hardware queue\n"
                         "\tWriter(Args.hsa_queue_packet_submit.packets, Args.hsa_queue_packet_submit.pkt_count);\n"
                         "\t} else {\n"
                         "\t\tWriter(Packets, PktCount);\n"
                         "}\n"
                         "\n}")

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
                callback_defs.append(f'static {return_type} '
                                     f'{hsa_function_name}_callback({", ".join(formatted_params)}) {{\n'
                                     "\tauto& HsaInterceptor = luthier::hsa::Interceptor::instance();\n"
                                     f"\tauto ApiId = HSA_API_EVT_ID_{hsa_function_name};\n"
                                     "\tbool IsUserCallbackEnabled = HsaInterceptor.isUserCallbackEnabled(ApiId);\n"
                                     "\tbool IsInternalCallbackEnabled = HsaInterceptor.isInternalCallbackEnabled("
                                     "ApiId);\n"
                                     "\tbool ShouldCallback = IsUserCallbackEnabled || IsInternalCallbackEnabled;\n")
                if return_type != "void":
                    callback_defs.append(f'\t{return_type} Out{{}};\n')
                callback_defs.append("\tif (ShouldCallback) {\n"
                                     "\t\tauto& HsaUserCallback = HsaInterceptor.getUserCallback();\n"
                                     "\t\tauto& HsaInternalCallback = HsaInterceptor.getInternalCallback();\n"
                                     "\t\thsa_api_evt_args_t Args;\n"
                                     "\t\tbool SkipFunction{false};\n")
                for p in hsa_function_cxx.parameters:
                    callback_defs.append(f"\t\tArgs.{hsa_function_name}.{p.name} = {p.name};\n")
                callback_defs.append("\t\tif (IsUserCallbackEnabled)\n"
                                     "\t\t\tHsaUserCallback(&Args, luthier::API_EVT_PHASE_ENTER, ApiId);\n"
                                     "\t\tif (IsInternalCallbackEnabled)\n"
                                     "\t\t\tHsaInternalCallback(&Args, luthier::API_EVT_PHASE_ENTER, ApiId, "
                                     "&SkipFunction);\n"
                                     "\t\tif (!SkipFunction)\n")
                if hsa_function_name == "hsa_queue_create":  # Create intercept queues instead
                    callback_defs.append("{\t\t\tOut = HsaInterceptor.getSavedHsaTables("
                                         ").amd_ext.hsa_amd_queue_intercept_create_fn("
                                         "Args.hsa_queue_create.agent, Args.hsa_queue_create.size,"
                                         "Args.hsa_queue_create.type, Args.hsa_queue_create.callback,"
                                         "Args.hsa_queue_create.data, Args.hsa_queue_create.private_segment_size,"
                                         "Args.hsa_queue_create.group_segment_size, Args.hsa_queue_create.queue);\n"
                                         "\t\t\tif (Out != HSA_STATUS_SUCCESS)\n"
                                         "\t\t\t\treturn Out;\n"
                                         "\t\t\tOut = HsaInterceptor.getSavedHsaTables("
                                         ").amd_ext.hsa_amd_queue_intercept_register_fn("
                                         "*Args.hsa_queue_create.queue, queueSubmitWriteInterceptor, "
                                         "*Args.hsa_queue_create.queue);\n"
                                         "\t\t\tif (Out != HSA_STATUS_SUCCESS)}\n")
                else:
                    if return_type != "void":
                        callback_defs.append(f'\t\t\tOut = ')
                    else:
                        callback_defs.append("\t\t\t")
                    callback_defs.append(
                        f'HsaInterceptor.getSavedHsaTables().{api_container_field_name}.{api_table_function_name}(')
                    for i, p in enumerate(hsa_function_cxx.parameters):
                        callback_defs.append(f'Args.{hsa_function_name}.{p.name}')
                        if i != len(hsa_function_cxx.parameters) - 1:
                            callback_defs.append(", ")
                    callback_defs.append(");\n")
                callback_defs.append("\t\tif (IsUserCallbackEnabled)\n"
                                     "\t\t\tHsaUserCallback(&Args, luthier::API_EVT_PHASE_EXIT, ApiId);\n"
                                     "\t\tif (IsInternalCallbackEnabled)\n"
                                     "\t\t\tHsaInternalCallback(&Args, luthier::API_EVT_PHASE_EXIT, ApiId, "
                                     "&SkipFunction);\n")
                if return_type != "void":
                    callback_defs.append("\t\treturn Out;\n")
                callback_defs.append("\t}\n"
                                     "\telse {\n")
                callback_defs.append(f'\t\treturn HsaInterceptor.getSavedHsaTables().{api_container_field_name}' \
                                     f'.{api_table_function_name}(')
                for i, p in enumerate(hsa_function_cxx.parameters):
                    callback_defs.append(f'{p.name}')
                    if i != len(hsa_function_cxx.parameters) - 1:
                        callback_defs.append(", ")
                callback_defs.append(");\n")
                callback_defs.append("\n\t}\n}\n")

    # Write the generated code into appropriate files
    with open(args.hpp_structs_save_path, "w") as f:
        f.writelines(callback_enums)
        f.write("\n")
        f.writelines(callback_arguments_struct)

    with open(args.cpp_callback_save_path, "w") as f:
        f.writelines(callback_defs)
        f.write("\n")
        f.writelines(wrapper_install_defs)


if __name__ == "__main__":
    main()
