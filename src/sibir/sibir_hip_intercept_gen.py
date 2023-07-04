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
import sys
import re
import warnings

import CppHeaderParser
from typing import *
import argparse
import filecmp


# Verbose message


#############################################################
# Normalizing API name
def filtr_api_name(name):
    name = re.sub(r'\s*$', r'', name);
    return name


def filtr_api_decl(record):
    record = re.sub(r"\s__dparm\([^)]*\)", r'', record);
    record = re.sub("\(void\*\)", r'', record);
    return record


# Normalizing API arguments
def filtr_api_args(args_str):
    args_str = re.sub(r'^\s*', r'', args_str)
    args_str = re.sub(r'\s*$', r'', args_str)
    args_str = re.sub(r'\s*,\s*', r',', args_str)
    args_str = re.sub(r'\s+', r' ', args_str)
    args_str = re.sub(r'\s*(\*+)\s*', r'\1 ', args_str)
    args_str = re.sub(r'(\benum|struct) ', '', args_str)
    return args_str


# Normalizing types
def norm_api_types(type_str):
    type_str = re.sub(r'uint32_t', r'unsigned int', type_str)
    type_str = re.sub(r'^unsigned$', r'unsigned int', type_str)
    return type_str


# Creating a list of arguments [(type, name), ...]
def list_api_args(args_str):
    args_str = filtr_api_args(args_str)
    args_list = []
    if args_str != '':
        for arg_pair in args_str.split(','):
            if arg_pair == 'void':
                continue
            arg_pair = re.sub(r'\s*=\s*\S+$', '', arg_pair)
            m = re.match("^(.*)\s(\S+)$", arg_pair)
            if m:
                arg_type = norm_api_types(m.group(1))
                arg_name = m.group(2)
                args_list.append((arg_type, arg_name))
            else:
                fatal("bad args: args_str: '" + args_str + "' arg_pair: '" + arg_pair + "'")
    return args_list


# Creating arguments string "type0, type1, ..."
def filtr_api_types(args_str):
    args_list = list_api_args(args_str)
    types_str = ''
    for arg_tuple in args_list:
        types_str += arg_tuple[0] + ', '
    return types_str


# Creating options list [opt0, opt1, ...]
def filtr_api_opts(args_str):
    args_list = list_api_args(args_str)
    opts_list = []
    for arg_tuple in args_list:
        opts_list.append(arg_tuple[1])
    return opts_list


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


#############################################################
# Parsing API header
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
def parse_api(inp_file_p, input_args_map, output_type_map):
    global inp_file
    global line_num
    inp_file = inp_file_p

    beg_pattern = re.compile(r"^(hipError_t|const char\s*\*)\s+([^(]+)\(")
    api_pattern = re.compile(r"^(hipError_t|const char\s*\*)\s+([^(]+)\(([^)]*)\)")
    end_pattern = re.compile("Texture")
    hidden_pattern = re.compile(r'__attribute__\(\(visibility\("hidden"\)\)\)')
    nms_open_pattern = re.compile(r'namespace hip_impl {')
    nms_close_pattern = re.compile(r'}')

    inp = open(inp_file, 'r')

    found = 0
    hidden = 0
    nms_level = 0
    record = ""
    line_num = -1

    for line in inp.readlines():
        record += re.sub(r'^\s+', r' ', line[:-1])
        line_num += 1

        if len(record) > REC_MAX_LEN:
            fatal("bad record \"" + record + "\"")

        m = beg_pattern.match(line)
        if m:
            name = m.group(2)
            if hidden != 0:
                message("api: " + name + " - hidden")
            elif nms_level != 0:
                message("api: " + name + " - hip_impl")
            else:
                message("api: " + name)
                found = 1

        if found != 0:
            record = re.sub(r"\s__dparm\([^)]*\)", '', record)
            m = api_pattern.match(record)
            if m:
                found = 0
                if end_pattern.search(record): continue
                api_name = filtr_api_name(m.group(2))
                api_args = m.group(3)
                out_type = m.group(1)
                if api_name not in input_args_map:
                    input_args_map[api_name] = api_args
                    output_type_map[api_name] = out_type
            else:
                continue

        hidden = 0
        if hidden_pattern.match(line):
            hidden = 1

        if nms_open_pattern.match(line):
            nms_level += 1
        if (nms_level > 0) and nms_close_pattern.match(line):
            nms_level -= 1
        if nms_level < 0:
            fatal("nms level < 0")

        record = ""

    inp.close()
    line_num = -1


#############################################################
# Parsing API implementation
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset) {
#    HIP_INIT_API(hipSetupArgument, arg, size, offset);
# inp_file - input implementation source file
# api_map - input public API map [<api name>] => <api args>
# out - output map  [<api name>] => [opt0, opt1, ...]
def parse_content(inp_file_p, api_map, out):
    global hip_patch_mode
    global types_check_mode
    global private_check_mode
    global inp_file
    global line_num
    inp_file = inp_file_p

    # API method begin pattern
    beg_pattern = re.compile(r"^(hipError_t|const char\s*\*)\s+[^(]+\(")
    # API declaration pattern
    decl_pattern = re.compile(r"^(hipError_t|const char\s*\*)\s+([^(]+)\(([^\)]*)\)\s*;")
    # API definition pattern
    api_pattern = re.compile(r"^(hipError_t|const char\s*\*)\s+([^(]+)\(([^)]*)\)\s*{")
    # API init macro pattern
    init_pattern = re.compile(r"(^\s*HIP_INIT_API\s*\s*)\((([^,]+)(,.*|)|)(\);|,)\s*$")

    # Open input file
    inp = open(inp_file, 'r')

    # API name
    api_name = ""
    # Valid public API found flag
    api_valid = 0
    # API overload (parameters mismatch)
    api_overload = 0

    # Input file patched content
    content = ''
    # Sub content for found API definition
    sub_content = ''
    # Current record, accumulating several API definition related lines
    record = ''
    # Current input file line number
    line_num = -1
    # API beginning found flag
    found = 0

    # Reading input file
    for line in inp.readlines():
        # Accumulating record
        record += re.sub(r'^\s+', r' ', line[:-1])
        line_num += 1

        if len(record) > REC_MAX_LEN:
            fatal("bad record \"" + record + "\"")
            break;

        # Looking for API begin
        if found == 0:
            record = re.sub(r'\s*extern\s+"C"\s+', r'', record);
            if beg_pattern.match(record):
                found = 1
                record = filtr_api_decl(record)

        # Matching API declaration
        if found == 1:
            if decl_pattern.match(record):
                found = 0

        # Matching API definition
        if found == 1:
            m = api_pattern.match(record)
            # Checking if complete API matched
            if m:
                found = 2
                api_valid = 0
                api_overload = 0
                api_name = filtr_api_name(m.group(2))
                # Checking if API name is in the API map
                if (private_check_mode == 0) or (api_name in api_map):
                    if not api_name in api_map:
                        api_map[api_name] = ''
                    # Getting API arguments
                    api_args = m.group(3)
                    # Getting etalon arguments from the API map
                    eta_args = api_map[api_name]
                    if eta_args == '':
                        eta_args = api_args
                        api_map[api_name] = eta_args
                    # Normalizing API arguments
                    api_types = filtr_api_types(api_args)
                    # Normalizing etalon arguments
                    eta_types = filtr_api_types(eta_args)
                    if (api_types == eta_types) or ((types_check_mode == 0) and (not api_name in out)):
                        # API is already found and not is mismatched
                        if (api_name in out):
                            fatal("API redefined \"" + api_name + "\", record \"" + record + "\"")
                        # Set valid public API found flag
                        api_valid = 1
                        # Set output API map with API arguments list
                        out[api_name] = filtr_api_opts(api_args)
                        # Register missmatched API methods
                    else:
                        api_overload = 1
                        # Warning about mismatched API, possible non public overloaded version
                        api_diff = '\t\t' + inp_file + " line(" + str(
                            line_num) + ")\n\t\tapi: " + api_types + "\n\t\teta: " + eta_types
                        message("\t" + api_name + ' args mismatch:\n' + api_diff + '\n')

        # API found action
        if found == 2:
            if hip_patch_mode != 0:
                # Looking for INIT macro
                m = init_pattern.match(line)
                if m:
                    init_name = api_name
                    if api_overload == 1:
                        init_name = 'NONE'
                    init_args = m.group(4)
                    line = m.group(1) + '(' + init_name + init_args + m.group(5) + '\n'

            m = init_pattern.match(line)
            if m:
                found = 0
                if api_valid == 1: message("\t" + api_name)
                # Ignore if it is initialized as NONE
                init_name = m.group(3)
                if init_name != 'NONE':
                    # Check if init name matching API name
                    # if init_name != api_name:
                    #   fatal("init name mismatch: '" + init_name +  "' <> '" + api_name + "'")
                    # Registering dummy API for non public API if the name in INIT is not NONE
                    if api_valid == 0:
                        # If init name is not in public API map then it is private API
                        # else it was not identified and will be checked on finish
                        if not init_name in api_map:
                            if init_name in out:
                                continue
                                fatal("API reinit \"" + api_name + "\", record \"" + record + "\"")
                            out[init_name] = []
            elif re.search('}', line):
                found = 0
                # Expect INIT macro for valid public API
                # Removing and registering non-conformant APIs with missing HIP_INIT macro
                if api_valid == 1:
                    if api_name in out:
                        del out[api_name]
                        del api_map[api_name]
                        # Registering non-conformant APIs
                        out['.' + api_name] = 1
                    else:
                        fatal("API is not in out \"" + api_name + "\", record \"" + record + "\"")

        if found != 1:
            record = ""
        content += line

    inp.close()
    line_num = -1

    if len(out) != 0:
        return content
    else:
        return ''


# src path walk
def parse_src(api_map, src_path, src_patt, out):
    global recursive_mode

    pattern = re.compile(src_patt)
    src_path = re.sub(r'\s', '', src_path)
    for src_dir in src_path.split(':'):
        message("Parsing " + src_dir + " for '" + src_patt + "'")
        for root, dirs, files in os.walk(src_dir):
            for fnm in files:
                if pattern.search(fnm):
                    file = root + '/' + fnm
                    message(file)
                    content = parse_content(file, api_map, out)
                    if (hip_patch_mode != 0) and (content != ''):
                        f = open(file, 'w')
                        f.write(content)
                        f.close()
            if recursive_mode == 0:
                break


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
    f.write(f'\tHIP_PRIVATE_API_ID_LAST = {api_id}\n')
    f.write("};\n\n")
    f.write('#endif')


def generate_hip_intercept_args(f: IO[Any], hip_runtime_api_map: Dict[str, CppHeaderParser.CppMethod]):
    f.write('#ifndef HIP_ARGS\n#define HIP_ARGS\n\n')
    f.write("#include <hip/hip_runtime_api.h>\n\n")
    for name in sorted(hip_runtime_api_map.keys()):
        f.write(f'typedef struct {name}_args_s {{\n')

        output_type = hip_runtime_api_map[name]['returns']
        args = hip_runtime_api_map[name]['parameters']
        if len(args) != 0:
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
        f.write(f'}} {name}_args_t;\n\n')
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

    f.write('#include "hip_intercept.h"\n\n\n')
    for name in sorted(hip_runtime_api_map.keys()):
        f.write('__attribute__((visibility("default")))\n')
        output_type = hip_runtime_api_map[name]['returns']
        f.write(f'{output_type} {name}(')
        args = hip_runtime_api_map[name]['parameters']
        if len(args) != 0:
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
        f.write('\tauto& hipInterceptor = SibirHipInterceptor::Instance();\n')
        f.write('\tauto& hipCallback = hipInterceptor.getCallback();\n')
        if f"HIP_API_ID_{name}" not in hip_api_id_enums:
            f.write(f'\tauto api_id = HIP_PRIVATE_API_ID_{name};\n')
        else:
            f.write(f'\tauto api_id = HIP_API_ID_{name};\n')
        f.write('\t// Copy Arguments for PHASE_ENTER\n')
        f.write('\thip_api_args_t hip_args{};\n')
        if len(args) != 0:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(f"\thip_args.{name}.{arg_name} = {arg_name};\n")
        f.write("\thipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);\n")
        f.write(f"    static auto hip_func = hipInterceptor.GetHipFunction<{output_type}(*)(")
        if len(args) != 0:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(arg_type)
                if i != len(args) - 1:
                    f.write(',')
        f.write(f')>("{name}");\n')
        f.write(f'    {output_type} out = hip_func(')
        if len(args) != 0:
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(f"hip_args.{name}.{arg_name}")
                if i != len(args) - 1:
                    f.write(', ')
        f.write(");\n")
        f.write("    // Exit Callback\n")
        f.write("    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);\n")
        if len(args) != 0:
            f.write("    // Copy the modified arguments back to the original arguments\n")
            for i, arg in enumerate(args):
                arg_type = arg['type']
                arg_name = arg['name']
                f.write(f"    {arg_name} = hip_args.{name}.{arg_name};\n")
        f.write("\n    return out;\n}\n\n")


def parse_and_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("HIP API interception generation script for Sibir; Originally used by AMD in "
                                     "HIP and roctracer projects.")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="enable verbose messages")
    parser.add_argument("-r", "--recursive", action='store_true',
                        help="process source directory recursively")
    parser.add_argument("-t", "--type-matching-check", action='store_true',
                        help="API types matching check")
    parser.add_argument("--private-api-check", action='store_true',
                        help="private API check")
    parser.add_argument("--macro-patching-mode", action='store_true',
                        help='Macro patching mode')
    parser.add_argument("-e", action='store_true',
                        help='on error exit mode')
    parser.add_argument("--hipamd-src-dir", type=str,
                        help='where hipamd src directory is located')
    parser.add_argument("--hip-include-dir", type=str, default='/opt/rocm/include/hip/',
                        help="path to the include directory of hip installation")
    parser.add_argument("--hip-prof-str", type=str,
                        help='path to hip_prof_str.h')
    parser.add_argument("--output", type=str, default="./hip_intercept.cpp",
                        help="where to save the generated interception function for Sibir")
    args = parser.parse_args()

    assert os.path.isdir(args.hipamd_src_dir), f"input file {args.hipamd_src_dir} not found"
    assert os.path.isdir(args.hip_include_dir), f"input file {args.hip_include_dir} not found"
    assert os.path.isfile(args.hip_prof_str), f"src dir {args.hip_prof_str} not found"

    return args


def convert_function_list_to_dict(func_list: List[CppHeaderParser.CppMethod]) -> Dict[str, CppHeaderParser.CppMethod]:
    out = {}
    for f in func_list:
        if f['name'] not in out:
            out[f['name']] = [f]
        else:
            out[f['name']].append(f)
    for f_name, f_list in out.items():
        if len(f_list) > 1:
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
            if len(candidates) > 1:
                print(f"two or more candidates are found for {f_name}.")
            if len(candidates) == 0:
                raise RuntimeError("All candidates lost")
            out[f_name] = f_list[list(candidates)[0]]
        else:
            out[f_name] = f_list[0]
    return out


# def remove_macro_annotated_code_from_header_content(content: List[str], macro: str) -> List[str]:
#     """
#     Removes the portion of the code annotated with the macro.
#     :param content: content of the header as a list of lines (strings)
#     :param macro: the macro to be removed (e.g. __cplusplus)
#     :return: content of the header with the annotated portion removed
#     """
#     curr_line_number = 0
#     while curr_line_number < len(content):
#         if macro in content[curr_line_number] and "#if" in content[curr_line_number]:
#             cur_ifdef_level = 1
#             while cur_ifdef_level:
#                 current_line = content[curr_line_number]
#                 del content[curr_line_number]
#                 if "#endif" not in current_line:
#                     if "#ifdef" in current_line and macro not in current_line:
#                         cur_ifdef_level += 1
#                 else:
#                     cur_ifdef_level -= 1
#         curr_line_number += 1
#     return content


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
        else:
            print(f"Function {name} was found in both private and public apis.\n")

    return out


def create_hip_public_api_enum_map(hip_api_str_path: str) -> Dict[str, int]:
    hip_prof_header = CppHeaderParser.CppHeader(hip_api_str_path)
    hip_prof_enums = hip_prof_header.enums[0]
    out = {}
    for e in hip_prof_enums['values']:
        if e['name'] not in out:
            out[e['name']] = e['value']
        else:
            raise RuntimeError("Found duplicate names in hip_api_str.h.")
    return out


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

    # Remove all the C++ code since dlsym doesn't work with it too well
    # remove_macro_annotated_code_from_header_content(hip_runtime_api_content, "__cplusplus")

    # Convert the content to a single string for CppHeader to parse
    hip_runtime_api_content = "".join(hip_runtime_api_content)

    hip_api_runtime_header = CppHeaderParser.CppHeader(hip_runtime_api_content, argType="string")
    hip_api_runtime_functions = convert_function_list_to_dict(hip_api_runtime_header.functions)

    hip_api_enums = create_hip_public_api_enum_map(args.hip_prof_str)

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
