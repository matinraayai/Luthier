# Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import re
import warnings

import CppHeaderParser
from typing import *
import argparse


def parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HIP API interception generation script for Luthier; Originally used by AMD in "
                                     "HIP and roctracer projects.")
    parser.add_argument("--hipamd-src-dir", type=str,
                        help='where hipamd src directory is located')
    parser.add_argument("--hip-include-dir", type=str, default='/opt/rocm/include/hip/',
                        help="path to the include directory of hip installation")
    parser.add_argument("--hip-prof-str", type=str,
                        help='path to hip_prof_str.h')
    parser.add_argument("--output", type=str, default="./hip_intercept.cpp",
                        help="where to save the generated interception function for Luthier")
    args = parser.parse_args()

    assert os.path.isdir(args.hipamd_src_dir), f"input file {args.hipamd_src_dir} not found"
    assert os.path.isdir(args.hip_include_dir), f"input file {args.hip_include_dir} not found"
    assert os.path.isfile(args.hip_prof_str), f"src dir {args.hip_prof_str} not found"

    return args


# Checking for pointer non-void arg type
def pointer_ck(arg_type):
    ptr_type = ''
    m = re.match(r'(.*)\*$', arg_type)
    if m:
        ptr_type = m.group(1)
        n = re.match(r'(.*)\*\*$', arg_type)
        if not n:
            ptr_type = re.sub(r'const ', '', ptr_type)
        if ptr_type == 'void':
            ptr_type = ''
    return ptr_type


def convert_function_list_to_dict(func_list: List[CppHeaderParser.CppMethod]) -> Dict[str, CppHeaderParser.CppMethod]:
    out = {}
    for f in func_list:
        if f['name'] not in out:
            out[f['name']] = [f]
        else:
            out[f['name']].append(f)
    out_copy = out.copy()
    for f_name, f_list in out_copy.items():
        # if len(f_list) > 1:
        candidates = set(i for i in range(len(f_list)))
        for i, f in enumerate(f_list):
            # Check for templated functions
            if not f['template'] is False:
                candidates.remove(i)
                continue
            # Check for functions with default arguments
            for p in f['parameters']:
                if 'default' in p:
                    candidates.remove(i)
                    continue

        if len(candidates) == 0:
            warnings.warn(f"No candidates for {f_name} are eligible for interception.")
            del out[f_name]
        elif len(candidates) > 1:
            warnings.warn(f"two or more candidates are found for {f_name}.")
            out[f_name] = f_list[list(candidates)[0]]
        else:
            out[f_name] = f_list[0]
    return out


def parse_hipamd_src_functions(src_dir: str, hip_prof_api_enums: Dict[str, int]):
    function_list = []
    for root, dirs, files in os.walk(src_dir):
        for file_name in files:
            if ".cpp" not in file_name:
                continue
            extern_c_range = []
            with open(os.path.join(root, file_name), 'r') as f:
                file_content = f.readlines()
            # If there is a block of code annotated with "extern C", detect its start/end line number
            in_extern_c = False
            extern_c_start = 0
            bracket_level = 0
            for i, line in enumerate(file_content):
                if 'extern "C" {' in line:
                    in_extern_c = True
                    extern_c_start = i
                    bracket_level = 1
                elif '{' in line:
                    bracket_level += 1
                elif '}' in line:
                    bracket_level -= 1
                if bracket_level == 0 and in_extern_c is True:
                    extern_c_range.append((extern_c_start, i))
                    extern_c_start = 0
                    in_extern_c = False
            try:
                parsed_file = CppHeaderParser.CppHeader("".join(file_content), argType="string")
            except CppHeaderParser.CppHeaderParser.CppParseError:
                warnings.warn(f"Could not parse {os.path.join(root, file_name)}.")
                continue
            file_functions = parsed_file.functions
            for func in file_functions:
                # Ignore Windows Dll start function
                if "DllMain" in func['name']:
                    continue
                # Ensure capturing any function already in HIP profiling API
                # These usually have the HIP_API_INIT macro at their start
                if "HIP_API_ID_" + func['name'] in hip_prof_api_enums:
                    function_list.append(func)
                else:
                    is_in_extern_c_range = False
                    for r in extern_c_range:
                        if r[0] <= func['line_number'] <= r[1]:
                            is_in_extern_c_range = True
                    function_line = file_content[func['line_number'] - 1]
                    if is_in_extern_c_range or ('extern "C"' in function_line and func['name'] in function_line):
                        function_list.append(func)
    return function_list


def combine_private_and_public_api_functions(public_api: Dict[str, CppHeaderParser.CppMethod],
                                             private_api: Dict[str, CppHeaderParser.CppMethod]) -> \
        Dict[str, CppHeaderParser.CppMethod]:
    out = public_api.copy()
    for name, f in private_api.items():
        if name not in public_api:
            out[name] = f

    return out


def parse_hip_prof_str_enum_map(hip_prof_str_path: str) -> Dict[str, int]:
    hip_prof_header = CppHeaderParser.CppHeader(hip_prof_str_path)
    hip_prof_enums = hip_prof_header.enums[0]
    out = {}
    for e in hip_prof_enums['values']:
        if e['name'] not in out:
            out[e['name']] = e['value']
        else:
            raise RuntimeError("Found duplicate names in hip_prof_str.h.")
    return out


def generate_hip_private_api_enums(f: IO[Any], hip_runtime_api_map: Dict[str, CppHeaderParser.CppMethod],
                                   hip_api_id_enums: Dict[str, int]):
    f.write('#ifndef HIP_PRIVATE_API\n#define HIP_PRIVATE_API\n\n')
    f.write('enum hip_private_api_id_t {\n')
    f.write('\tHIP_PRIVATE_API_ID_NONE = 0,\n')
    api_id = 1000
    f.write(f'\tHIP_PRIVATE_API_ID_FIRST = {api_id},\n')
    for api_name in sorted(hip_runtime_api_map):
        if f'HIP_API_ID_{api_name}' not in hip_api_id_enums:
            f.write(f'\tHIP_PRIVATE_API_ID_{api_name} = {api_id},\n')
            api_id += 1
    f.write(f'\tHIP_PRIVATE_API_ID_LAST = {api_id - 1}\n')
    f.write("};\n\n")
    f.write('#endif')


def generate_hip_intercept_args(f: IO[Any], hip_runtime_api_map: Dict[str, CppHeaderParser.CppMethod]):
    f.write('#ifndef HIP_ARGS\n#define HIP_ARGS\n\n')
    f.write("#include <hip/hip_runtime_api.h>\n")
    f.write("namespace hip {\n")
    f.write("\tstruct FatBinaryInfo{};\n")
    f.write("}\n\n")

    for name in sorted(hip_runtime_api_map.keys()):
        f.write(f'typedef struct hip_{name}_api_args_s {{\n')

        output_type = hip_runtime_api_map[name]['rtnType']
        args = hip_runtime_api_map[name]['parameters']
        are_args_non_empty = len(args) != 0 and not (len(args) == 1 and args[0]['type'] == 'void')
        if are_args_non_empty:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                ptr_type = pointer_ck(arg_type)
                arg_name = arg['name']
                # Checking for enum type
                if arg_type == "hipLimit_t":
                    arg_type = 'enum ' + arg_type
                # Function arguments
                f.write(f"\t{arg_type} {arg_name};\n")
                # if i != len(args) - 1:
                #     f.write(';\n')
        f.write(f'}} hip_{name}_api_args_t;\n\n')
        f.write(f'typedef {output_type} {name}_return_t;\n\n\n')
    f.write("#endif")


def generate_hip_intercept_dlsym_functions(f: IO[Any], hip_runtime_api_map: Dict[str, CppHeaderParser.CppMethod],
                                           hip_api_id_enums: Dict[str, int]) -> None:
    for name in hip_api_id_enums:
        actual_name = name[11:]
        if actual_name not in hip_runtime_api_map and actual_name != "NONE" and actual_name != "FIRST" and \
                actual_name != "LAST" and "RESERVED" not in actual_name and \
                hip_api_id_enums[name] != "HIP_API_ID_NONE":
            raise RuntimeError(f"{actual_name} is in hip_prof_str.h but not in the captured function APIs.\n")
    # Create the constructor definition of HipInterceptor here to avoid clashing between link.h and ELFIO definitions
    f.write('#include <link.h>\n#include "hip_intercept.hpp"\n\n\n')
    f.write("""luthier::HipInterceptor::HipInterceptor() {
    // Iterate through the process' loaded shared objects and try to dlopen the first entry with a
    // file name starting with the given 'pattern'. This allows the loader to acquire a handle
    // to the target library iff it is already loaded. The handle is used to query symbols
    // exported by that library.
    auto callback = [this](dl_phdr_info *info) {
    if (handle_ == nullptr && fs::path(info->dlpi_name).filename().string().rfind("libamdhip64.so", 0) == 0)
        handle_ = ::dlopen(info->dlpi_name, RTLD_LAZY);
    };
    dl_iterate_phdr(
        [](dl_phdr_info *info, size_t size, void *data) {
            (*reinterpret_cast<decltype(callback) *>(data))(info);
            return 0;
        }, &callback);
};\n\n""")

    for name in sorted(hip_runtime_api_map.keys()):
        if name != "hipCreateSurfaceObject" and name != "hipDestroySurfaceObject":
            f.write('extern "C" ')
        f.write('__attribute__((visibility("default")))\n')
        output_type = hip_runtime_api_map[name]['rtnType']
        f.write(f'{output_type} {name}(')
        args = hip_runtime_api_map[name]['parameters']
        are_args_non_empty = len(args) != 0 and not (len(args) == 1 and args[0]['type'] == 'void')
        if are_args_non_empty:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                ptr_type = pointer_ck(arg_type)
                arg_name = arg['name']
                # Checking for enum type
                if arg_type == "hipLimit_t":
                    arg_type = 'enum ' + arg_type
                # Function arguments
                f.write(f"{arg_type} {arg_name}")
                if i != len(args) - 1:
                    f.write(', ')
        f.write(') {\n')
        f.write('\tauto& hipInterceptor = luthier::HipInterceptor::Instance();\n')
        f.write('\tauto& hipUserCallback = hipInterceptor.getUserCallback();\n')
        f.write('\tauto& hipInternalCallback = hipInterceptor.getInternalCallback();\n')
        if f"HIP_API_ID_{name}" not in hip_api_id_enums:
            f.write(f'\tauto api_id = HIP_PRIVATE_API_ID_{name};\n')
        else:
            f.write(f'\tauto api_id = HIP_API_ID_{name};\n')
        f.write('\t// Copy Arguments for PHASE_ENTER\n')

        f.write("\t// Flag to skip calling the original function\n")
        f.write("\tbool skipFunction{false};\n")
        f.write("\tstd::optional<std::any> out{std::nullopt};\n")
        if are_args_non_empty:
            f.write(f'\thip_{name}_api_args_t hip_func_args{{')
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(f"{arg_name}")
                if i != len(args) - 1:
                    f.write(', ')
            f.write("};\n")
        callback_args = "static_cast<void*>(&hip_func_args)" if are_args_non_empty else "nullptr"

        f.write('\tif (!hipInterceptor.IsEnabledOpsEmpty() && !hipInterceptor.IsEnabledOps(api_id)) {')
        f.write(f'\n\t\tstatic auto hip_func = hipInterceptor.GetHipFunction<{output_type}(*)(')
        if are_args_non_empty:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(arg_type)
                if i != len(args) - 1:
                    f.write(',')
        f.write(f')>("{name}");\n')
        f.write("\t\t")
        if output_type != "void":
            f.write(f"out = ")
        f.write('hip_func(')
        if are_args_non_empty:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(f"hip_func_args.{arg_name}")
                if i != len(args) - 1:
                    f.write(', ')
        f.write(");\n\t")
        f.write("\treturn")
        if (output_type != "void"):
            f.write(f' std::any_cast<{output_type}>(*out)')
        f.write(";\n")
        f.write("\t};\n")

        f.write(f"\thipUserCallback({callback_args}, LUTHIER_API_EVT_PHASE_ENTER, api_id);\n")
        f.write(f"\thipInternalCallback({callback_args}, LUTHIER_API_EVT_PHASE_ENTER, api_id, &skipFunction, &out);\n")
        f.write("\tif (!skipFunction && !out.has_value()) {\n")
        f.write(f"\t\tstatic auto hip_func = hipInterceptor.GetHipFunction<{output_type}(*)(")
        if are_args_non_empty:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(arg_type)
                if i != len(args) - 1:
                    f.write(',')
        f.write(f')>("{name}");\n')
        f.write("\t\t")
        if output_type != "void":
            f.write(f"out = ")
        f.write('hip_func(')
        if are_args_non_empty:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(f"hip_func_args.{arg_name}")
                if i != len(args) - 1:
                    f.write(', ')
        f.write(");\n\t};")
        f.write("\t// Exit Callback\n")
        f.write(f"\thipUserCallback({callback_args}, LUTHIER_API_EVT_PHASE_EXIT, api_id);\n")
        f.write(f"\thipInternalCallback({callback_args}, LUTHIER_API_EVT_PHASE_EXIT, api_id, &skipFunction, &out);\n")
        if are_args_non_empty:
            f.write("\t// Copy the modified arguments back to the original arguments (if non-const)\n")
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                if "const" not in arg_type:
                    f.write(f"\t{arg_name} = hip_func_args.{arg_name};\n")
        if output_type != "void":
            f.write(f"\n\treturn std::any_cast<{output_type}>(*out);\n")
        f.write("};\n\n")

########################################################################################################################


def main():
    args = parse_and_validate_args()

    assert os.path.isdir(args.hipamd_src_dir), f"input file {args.hipamd_src_dir} not found"
    assert os.path.isfile(args.hip_prof_str), f"src dir {args.hip_prof_str} not found"
    assert os.path.isdir(args.hip_include_dir), f"input file {args.hip_include_dir} not found"

    # Parse API header
    with open(os.path.join(args.hip_include_dir, "hip_runtime_api.h"), 'r') as f:
        hip_runtime_api_content = f.readlines()

    # Remove the "DEPRECATED" macro from the header content to ensure correct parsing
    hip_runtime_api_content = [line for line in hip_runtime_api_content if "DEPRECATED" not in line]

    hip_runtime_api_content = [re.sub(r"\s__dparm\([^)]*\)", r'', line) for line in hip_runtime_api_content]

    # Convert the content to a single string for CppHeader to parse
    hip_runtime_api_content = "".join(hip_runtime_api_content)

    hip_api_runtime_header = CppHeaderParser.CppHeader(hip_runtime_api_content, argType="string")
    hip_api_runtime_functions = convert_function_list_to_dict(hip_api_runtime_header.functions)

    hip_api_enums = parse_hip_prof_str_enum_map(args.hip_prof_str)

    hip_private_api_functions = parse_hipamd_src_functions(args.hipamd_src_dir, hip_api_enums)

    hip_private_api_functions = convert_function_list_to_dict(hip_private_api_functions)

    all_api_functions = combine_private_and_public_api_functions(hip_api_runtime_functions,
                                                                 hip_private_api_functions)
    # Generating output header file

    with open("hip_private_api.h", 'w') as f:
        generate_hip_private_api_enums(f, all_api_functions, hip_api_enums)

    with open("hip_arg_types.h", 'w') as f:
        generate_hip_intercept_args(f, all_api_functions)

    with open(args.output, 'w') as f:
        generate_hip_intercept_dlsym_functions(f, all_api_functions, hip_api_enums)


if __name__ == "__main__":
    main()
