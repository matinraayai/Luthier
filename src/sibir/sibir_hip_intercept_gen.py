#!/usr/bin/python

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

import os, sys, re
import CppHeaderParser
import filecmp

PROF_HEADER = "hip_prof_str.h"
OUTPUT = PROF_HEADER
REC_MAX_LEN = 1024

# Recursive sources processing
recursive_mode = 0
# HIP_INIT_API macro patching
hip_patch_mode = 0
# API matching types check
types_check_mode = 0
# Private API check
private_check_mode = 0

# Messages and errors controll
verbose = 0
errexit = 0
inp_file = 'none'
line_num = -1

# Verbose message
def message(msg):
    if verbose: sys.stdout.write(msg + '\n')

# Fatal error termination
def error(msg):
    if line_num != -1:
        msg += ", file '" + inp_file + "', line (" + str(line_num) + ")"
    if errexit:
        msg = " Error: " + msg
    else:
        msg = " Warning: " + msg

    sys.stdout.write(msg + '\n')
    sys.stderr.write(sys.argv[0] + msg +'\n')

def fatal(msg):
    error(msg)
    sys.exit(1)

#############################################################
# Normalizing API name
def filtr_api_name(name):
    name = re.sub(r'\s*$', r'', name);
    return name

def filtr_api_decl(record):
    record = re.sub("\s__dparm\([^\)]*\)", r'', record);
    record = re.sub("\(void\*\)", r'', record);
    return record

# Normalizing API arguments
def filtr_api_args(args_str):
    args_str = re.sub(r'^\s*', r'', args_str);
    args_str = re.sub(r'\s*$', r'', args_str);
    args_str = re.sub(r'\s*,\s*', r',', args_str);
    args_str = re.sub(r'\s+', r' ', args_str);
    args_str = re.sub(r'\s*(\*+)\s*', r'\1 ', args_str);
    args_str = re.sub(r'(\benum|struct) ', '', args_str);
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
            if arg_pair == 'void': continue
            arg_pair = re.sub(r'\s*=\s*\S+$','', arg_pair);
            m = re.match("^(.*)\s(\S+)$", arg_pair);
            if m:
                arg_type = norm_api_types(m.group(1))
                arg_name = m.group(2)
                args_list.append((arg_type, arg_name))
            else:
                fatal("bad args: args_str: '" + args_str + "' arg_pair: '" + arg_pair + "'")
    return args_list;

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
        if ptr_type == 'void': ptr_type = ''
    return ptr_type
#############################################################
# Parsing API header
# hipError_t hipSetupArgument(const void* arg, size_t size, size_t offset);
def parse_api(inp_file_p, input_args_map, output_type_map):
    global inp_file
    global line_num
    inp_file = inp_file_p

    beg_pattern = re.compile("^(hipError_t|const char\s*\*)\s+([^\(]+)\(");
    api_pattern = re.compile("^(hipError_t|const char\s*\*)\s+([^\(]+)\(([^\)]*)\)");
    end_pattern = re.compile("Texture");
    hidden_pattern = re.compile(r'__attribute__\(\(visibility\("hidden"\)\)\)')
    nms_open_pattern = re.compile(r'namespace hip_impl {')
    nms_close_pattern = re.compile(r'}')

    inp = open(inp_file, 'r')

    found = 0
    hidden = 0
    nms_level = 0;
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
            record = re.sub("\s__dparm\([^\)]*\)", '', record);
            m = api_pattern.match(record)
            if m:
                found = 0
                if end_pattern.search(record): continue
                api_name = filtr_api_name(m.group(2))
                api_args = m.group(3)
                out_type = m.group(1)
                if not api_name in input_args_map:
                    input_args_map[api_name] = api_args
                    output_type_map[api_name] = out_type
            else: continue

        hidden = 0
        if hidden_pattern.match(line): hidden = 1

        if nms_open_pattern.match(line): nms_level += 1
        if (nms_level > 0) and nms_close_pattern.match(line): nms_level -= 1
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
    beg_pattern = re.compile("^(hipError_t|const char\s*\*)\s+[^\(]+\(");
    # API declaration pattern
    decl_pattern = re.compile("^(hipError_t|const char\s*\*)\s+([^\(]+)\(([^\)]*)\)\s*;");
    # API definition pattern
    api_pattern = re.compile("^(hipError_t|const char\s*\*)\s+([^\(]+)\(([^\)]*)\)\s*{");
    # API init macro pattern
    init_pattern = re.compile("(^\s*HIP_INIT_API[^\s]*\s*)\((([^,]+)(,.*|)|)(\);|,)\s*$");

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
    # Sub content for found API defiition
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
                    if not api_name in api_map: api_map[api_name] = ''
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
                        api_diff = '\t\t' + inp_file + " line(" + str(line_num) + ")\n\t\tapi: " + api_types + "\n\t\teta: " + eta_types
                        message("\t" + api_name + ' args mismatch:\n' + api_diff + '\n')

        # API found action
        if found == 2:
            if hip_patch_mode != 0:
                # Looking for INIT macro
                m = init_pattern.match(line)
                if m:
                    init_name = api_name
                    if api_overload == 1: init_name = 'NONE'
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

        if found != 1: record = ""
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
                    content = parse_content(file, api_map, out);
                    if (hip_patch_mode != 0) and (content != ''):
                        f = open(file, 'w')
                        f.write(content)
                        f.close()
            if recursive_mode == 0: break
#############################################################
# Generating profiling primitives header
# api_map - public API map [<api name>] => [(type, name), ...]
# callback_ids - public API callback IDs list (name, callback_id)
# opts_map - opts map  [<api name>] => [opt0, opt1, ...]
def generate_hip_intercept(f, api_map, out_type_map, callback_ids, opts_map):
    f.write('#include "hip_intercept.h"\n\n\n')
    for name in sorted(api_map.keys()):
        if name not in out_type_map:
            continue
        f.write('__attribute__((visibility("default")))\n')
        f.write(f'{out_type_map[name]} {name}(')
        args = api_map[name]
        if len(args) != 0:
            for i, arg_tuple in enumerate(args):
                arg_type = arg_tuple[0]
                ptr_type = pointer_ck(arg_type)
                arg_name = arg_tuple[1]
                # Checking for enum type
                if arg_type == "hipLimit_t": arg_type = 'enum ' + arg_type
                # Function arguments
                f.write(f"{arg_type} {arg_name}")
                # if ptr_type != '':
                #     f.write('      ' + ptr_type + ' ' + arg_name + '__val;\n')
                if i != len(args) - 1:
                    f.write(', ')
            # f.write('    } ' + name + ';\n')
        f.write(') {\n')
        f.write('    auto& hipInterceptor = SibirHipInterceptor::Instance();\n')
        f.write('    auto& hipCallback = hipInterceptor.getCallback();\n')
        f.write(f'    auto api_id = HIP_API_ID_{name};\n')
        f.write('    // Copy Arguments for PHASE_ENTER\n')
        f.write('    hip_api_args_t hip_args{};\n')
        if len(args) != 0:
            for i, arg_tuple in enumerate(args):
                arg_type = arg_tuple[0]
                arg_name = arg_tuple[1]
                f.write(f"    hip_args.{name}.{arg_name} = {arg_name};\n")
        f.write("    hipCallback(&hip_args, SIBIR_API_PHASE_ENTER, api_id);\n")
        f.write(f"    static auto hip_func = hipInterceptor.GetHipFunction<{out_type_map[name]}(*)(")
        if len(args) != 0:
            for i, arg_tuple in enumerate(args):
                arg_type = arg_tuple[0]
                arg_name = arg_tuple[1]
                f.write(arg_type)
                if i != len(args) - 1:
                    f.write(',')
                # else:
                #     f.write(")")
        f.write(f')>("{name}");\n')
        f.write(f'    {out_type_map[name]} out = hip_func(')
        if len(args) != 0:
            for i, arg_tuple in enumerate(args):
                arg_type = arg_tuple[0]
                arg_name = arg_tuple[1]
                f.write(f"hip_args.{name}.{arg_name}")
                if i != len(args) - 1:
                    f.write(', ')
        f.write(");\n")
        f.write("    // Exit Callback\n")
        f.write("    hipCallback(&hip_args, SIBIR_API_PHASE_EXIT, api_id);\n")
        if len(args) != 0:
            f.write("    // Copy the modified arguments back to the original arguments\n")
            for i, arg_tuple in enumerate(args):
                arg_type = arg_tuple[0]
                arg_name = arg_tuple[1]
                f.write(f"    {arg_name} = hip_args.{name}.{arg_name};\n")
        f.write("\n    return out;\n}\n\n")


#############################################################
# main
while len(sys.argv) > 1:
    if not re.match(r'-', sys.argv[1]): break

    if (sys.argv[1] == '-v'):
        verbose = 1
        sys.argv.pop(1)

    if (sys.argv[1] == '-r'):
        recursive_mode = 1
        sys.argv.pop(1)

    if (sys.argv[1] == '-t'):
        types_check_mode = 1
        sys.argv.pop(1)

    if (sys.argv[1] == '--priv'):
        private_check_mode = 1
        sys.argv.pop(1)

    if (sys.argv[1] == '-e'):
        errexit = 1
        sys.argv.pop(1)

    if (sys.argv[1] == '-p'):
        hip_patch_mode = 1
        sys.argv.pop(1)

# Usage
if (len(sys.argv) < 4):
    fatal ("Usage: " + sys.argv[0] + " [-v] <input HIP API .h file> <patched srcs path> <previous output> [<output>]\n" +
           "  -v - verbose messages\n" +
           "  -r - process source directory recursively\n" +
           "  -t - API types matching check\n" +
           "  --priv - private API check\n" +
           "  -e - on error exit mode\n" +
           "  -p - HIP_INIT_API macro patching mode\n" +
           "\n" +
           "  Example:\n" +
           "  $ " + sys.argv[0] + " -v -p -t --priv ../hip/include/hip/hip_runtime_api.h" +
           " ./src ./include/hip/amd_detail/hip_prof_str.h ./include/hip/amd_detail/hip_prof_str.h.new");

# API header file given as an argument
src_pat = "\.cpp$"
api_hfile = sys.argv[1]
if not os.path.isfile(api_hfile):
    fatal("input file '" + api_hfile + "' not found")

# Srcs directory given as an argument
src_dir = sys.argv[2]
if not os.path.isdir(src_dir):
    fatal("src directory " + src_dir + "' not found")

# Current hip_prof_str include
INPUT = sys.argv[3]
if not os.path.isfile(INPUT):
    fatal("input file '" + INPUT + "' not found")

if len(sys.argv) > 4:
    OUTPUT = sys.argv[4]

# API declaration map
api_map = {
    'hipSetupArgument': '',
    'hipMalloc3DArray': '',
    'hipFuncGetAttribute': '',
    'hipMemset3DAsync': '',
    'hipKernelNameRef': '',
    'hipStreamGetPriority': '',
    'hipLaunchByPtr': '',
    'hipFreeHost': '',
    'hipGetErrorName': '',
    'hipMemcpy3DAsync': '',
    'hipMemcpyParam2DAsync': '',
    'hipArray3DCreate': '',
    'hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags': '',
    'hipOccupancyMaxPotentialBlockSize': '',
    'hipMallocManaged': '',
    'hipOccupancyMaxActiveBlocksPerMultiprocessor': '',
    'hipGetErrorString': '',
    'hipMallocHost': '',
    'hipModuleLoadDataEx': '',
    'hipGetDeviceProperties': '',
    'hipConfigureCall': '',
    'hipHccModuleLaunchKernel': '',
    'hipExtModuleLaunchKernel': '',
}
# API options map
opts_map = {}

out_type_map = {}
# Parsing API header
parse_api(api_hfile, api_map, out_type_map)

# Parsing sources
parse_src(api_map, src_dir, src_pat, opts_map)

try:
    cppHeader = CppHeaderParser.CppHeader(INPUT)
except CppHeaderParser.CppParseError as e:
    print(e)
    sys.exit(1)

# Callback IDs
api_callback_ids = []

for enum in cppHeader.enums:
    if enum['name'] == 'hip_api_id_t':
        for value in enum['values']:
            if value['name'] == 'HIP_API_ID_NONE' or value['name'] == 'HIP_API_ID_FIRST':
                continue
            if value['name'] == 'HIP_API_ID_LAST':
                break
            m = re.match(r'HIP_API_ID_(\S*)', value['name'])
            if m:
                api_callback_ids.append((m.group(1), value['value']))
        break

# Checking for non-conformant APIs with missing HIP_INIT macro
for name in list(opts_map.keys()):
    m = re.match(r'\.(\S*)', name)
    if m:
        message("Init missing: " + m.group(1))
        del opts_map[name]

# Converting api map to map of lists
# Checking for not found APIs
not_found = 0
if len(opts_map) != 0:
    for name in api_map.keys():
        args_str = api_map[name];
        api_map[name] = list_api_args(args_str)
        if not name in opts_map:
            error("implementation not found: " + name)
            not_found += 1
if not_found != 0:
    error(str(not_found) + " API calls missing in interception layer")

# The output subdirectory seems to exist or not depending on the
# version of cmake.
output_dir = os.path.dirname(OUTPUT)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generating output header file
with open(OUTPUT, 'w') as f:
    generate_hip_intercept(f, api_map, out_type_map, api_callback_ids, opts_map)

# if not filecmp.cmp(INPUT, OUTPUT):
#     fatal("\"" + INPUT + "\" needs to be re-generated and checked-in with the current changes")

# Successfull exit
sys.exit(0)
