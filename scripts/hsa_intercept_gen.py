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
                        default="./hsa_intercept.cpp",
                        help="location of where the generated C++ callback file will be saved")
    parser.add_argument("--hpp-structs-save-path", type=str,
                        default="../include/hsa_trace_api.hpp",
                        help="location of where the generated C++ header file containing the callback args struct "
                             "and callback enumerators will be saved")
    args = parser.parse_args()
    return args


H_OUT = 'hsa_prof_str.h'
CPP_OUT = 'hsa_intercept.cpp'

API_TABLES_H = 'hsa_api_trace.h'
API_HEADERS_H = (
    ('CoreApi', 'hsa.h'),
    ('AmdExt', 'hsa_ext_amd.h'),
    ('ImageExt', 'hsa_ext_image.h'),
    ('AmdExt', API_TABLES_H),
)
API_TABLE_NAMES = {'CoreApi': 'core', 'AmdExt': 'amd_ext', 'ImageExt': 'image_ext'}


#############################################################
# Error handler
def fatal(module, msg):
    print(module + ' Error: "' + msg + '"', file=sys.stderr)
    sys.exit(1)


# Get next text block
def NextBlock(pos, record):
    if len(record) == 0:
        return pos

    space_pattern = re.compile(r'(\s+)')
    word_pattern = re.compile(r'([\w\*]+)')
    if record[pos] != '(':
        m = space_pattern.match(record, pos)
        if not m:
            m = word_pattern.match(record, pos)
        if m:
            return pos + len(m.group(1))
        else:
            fatal('NextBlock', "bad record '" + record + "' pos(" + str(pos) + ")")
    else:
        count = 0
        for index in range(pos, len(record)):
            if record[index] == '(':
                count = count + 1
            elif record[index] == ')':
                count = count - 1
                if count == 0:
                    index = index + 1
                    break
        if count != 0:
            fatal('NextBlock', "count is not zero (" + str(count) + ")")
        if record[index - 1] != ')':
            fatal('NextBlock', "last char is not ')' '" + record[index - 1] + "'")
        return index


#############################################################
# API table parser class
class ApiTableParser:

    def __init__(self, header, name):
        self.name = name

        if not os.path.isfile(header):
            self.fatal("file '" + header + "' not found")

        self.inp = open(header, 'r')

        self.beg_pattern = re.compile('^\s*struct\s+' + name + 'Table\s*{\s*$')
        self.end_pattern = re.compile('^\s*};\s*$')
        self.array = []
        self.parse()

    # normalizing a line
    def norm_line(self, line):
        return re.sub(r'^\s+', r' ', line[:-1])

    # check for start record
    def is_start(self, record):
        return self.beg_pattern.match(record)

    # check for end record
    def is_end(self, record):
        return self.end_pattern.match(record)

    # check for declaration entry record
    def is_entry(self, record):
        return re.match(r'^\s*decltype\(([^\)]*)\)', record)

    # parse method
    def parse(self):
        active = 0
        for line in self.inp.readlines():
            record = self.norm_line(line)
            if self.is_start(record):
                active = 1
            if active != 0:
                if self.is_end(record): return
                m = self.is_entry(record)
                if m:
                    self.array.append(m.group(1))


#############################################################
# API declaration parser class
class ApiDeclParser:
    def fatal(self, msg):
        fatal('ApiDeclParser', msg)

    def __init__(self, header, array, data):
        if not os.path.isfile(header):
            self.fatal("file '" + header + "' not found")

        self.inp = open(header, 'r')

        self.end_pattern = re.compile('\);\s*$')
        self.data = data
        for call in array:
            if call in data:
                self.fatal(call + ' is already found')
            self.parse(call)

    # api record filter
    def api_filter(self, record):
        record = re.sub(r'\sHSA_API\s', r' ', record)
        record = re.sub(r'\sHSA_DEPRECATED\s', r' ', record)
        return record

    # check for start record
    def is_start(self, call, record):
        return re.search('\s' + call + '\s*\(', record)

    # check for API method record
    def is_api(self, call, record):
        record = self.api_filter(record)
        return re.match('\s+\S+\s+' + call + '\s*\(', record)

    # check for end record
    def is_end(self, record):
        return self.end_pattern.search(record)

    # parse method args
    def get_args(self, record):
        struct = {'ret': '', 'args': '', 'astr': {}, 'alst': [], 'tlst': []}
        record = re.sub(r'^\s+', r'', record)
        record = re.sub(r'\s*(\*+)\s*', r'\1 ', record)
        rind = NextBlock(0, record)
        struct['ret'] = record[0:rind]
        pos = record.find('(')
        end = NextBlock(pos, record)
        args = record[pos:end]
        args = re.sub(r'^\(\s*', r'', args)
        args = re.sub(r'\s*\)$', r'', args)
        args = re.sub(r'\s*,\s*', r',', args)
        struct['args'] = re.sub(r',', r', ', args)
        if len(args) == 0: return struct

        pos = 0
        args = args + ','
        while pos < len(args):
            ind1 = NextBlock(pos, args)  # type
            ind2 = NextBlock(ind1, args)  # space
            if args[ind2] != '(':
                while ind2 < len(args):
                    end = NextBlock(ind2, args)
                    if args[end] == ',':
                        break
                    else:
                        ind2 = end
                name = args[ind2:end]
            else:
                ind3 = NextBlock(ind2, args)  # field
                m = re.match(r'\(\s*\*\s*(\S+)\s*\)', args[ind2:ind3])
                if not m:
                    self.fatal("bad block3 '" + args + "' : '" + args[ind2:ind3] + "'")
                name = m.group(1)
                end = NextBlock(ind3, args)  # the rest
            item = args[pos:end]
            struct['astr'][name] = item
            struct['alst'].append(name)
            struct['tlst'].append(item)
            if args[end] != ',':
                self.fatal("no comma '" + args + "'")
            pos = end + 1

        return struct

    # parse given api
    def parse(self, call):
        record = ''
        active = 0
        found = 0
        api_name = ''
        prev_line = ''

        self.inp.seek(0)
        for line in self.inp.readlines():
            record += ' ' + line[:-1]
            record = re.sub(r'^\s*', r' ', record)

            if active == 0:
                if self.is_start(call, record):
                    active = 1
                    m = self.is_api(call, record)
                    if not m:
                        record = ' ' + prev_line + ' ' + record
                        m = self.is_api(call, record)
                        if not m:
                            self.fatal("bad api '" + line + "'")

            if active == 1:
                if self.is_end(record):
                    self.data[call] = self.get_args(record)
                    active = 0
                    found = 0

            if active == 0: record = ''
            prev_line = line


#############################################################
# API description parser class
class ApiDescrParser:
    def fatal(self, msg):
        fatal('ApiDescrParser', msg)

    def __init__(self, out_h_file, hsa_dir, api_table_h, api_headers, license):
        out_macro = re.sub(r'[\/\.]', r'_', out_h_file.upper()) + '_'

        self.h_content = ''
        self.cpp_content = ''
        self.api_names = []
        self.api_calls = {}
        self.api_rettypes = set()
        self.api_id = {}

        api_data = {}
        api_list = []
        ns_calls = []

        for i in range(0, len(api_headers)):
            (name, header) = api_headers[i]

            if i < len(api_headers) - 1:
                api = ApiTableParser(hsa_dir + api_table_h, name)
                api_list = api.array
                self.api_names.append(name)
                self.api_calls[name] = api_list
            else:
                api_list = ns_calls
                ns_calls = []

            for call in api_list:
                if call in api_data:
                    self.fatal("call '" + call + "' is already found")

            ApiDeclParser(hsa_dir + header, api_list, api_data)

            for call in api_list:
                if call not in api_data:
                    # Not-supported functions
                    ns_calls.append(call)
                else:
                    # API ID map
                    self.api_id[call] = 'HSA_API_ID_' + call
                    # Return types
                    self.api_rettypes.add(api_data[call]['ret'])

        self.api_rettypes.discard('void')
        self.api_data = api_data
        self.ns_calls = ns_calls

        self.cpp_content += f"/* Generated by {os.path.basename(__file__)} */\n{license}\n\n"

        self.cpp_content += '#include <hsa/hsa_api_trace.h>\n'
        self.cpp_content += '#include "hsa_intercept.hpp"\n'
        self.cpp_content += '#include "luthier_types.h"\n'

        self.cpp_content += self.add_section('API callback functions', '', self.gen_callbacks)
        self.cpp_content += self.add_section('API intercepting code', '', self.gen_intercept)
        self.cpp_content += '\n'

    # add code section
    def add_section(self, title: str, gap, fun):
        content = ''
        n = 0
        content += '\n/* section: ' + title + ' */\n\n'
        content += fun(-1, '-', '-', {})
        for index in range(len(self.api_names)):
            last = (index == len(self.api_names) - 1)
            name = self.api_names[index]
            if n != 0:
                if gap == '':
                    content += fun(n, name, '-', {})
                content += '\n'
            content += gap + '/* block: ' + name + ' API */\n'
            for call in self.api_calls[name]:
                content += fun(n, name, call, self.api_data[call])
                n += 1
            content += fun(n, name, '-', {})
        content += fun(n, '-', '-', {})
        return content

    # generate API callbacks
    def gen_callbacks(self, n, name, call, struct):
        content = ''
        bad_callbacks = ['hsa_amd_portable_close_dmabuf',
                         'hsa_amd_portable_export_dmabuf',
                         'hsa_amd_memory_async_copy_on_engine',
                         'hsa_amd_memory_copy_engine_status',
                         'hsa_amd_spm_acquire',
                         'hsa_amd_spm_release',
                         'hsa_amd_spm_set_dest_buffer']
        if n == -1:
            content += '/* section: Static declarations */\n'
            content += '\n'
        if call != '-' and call not in bad_callbacks:
            call_id = self.api_id[call]
            ret_type = struct['ret']
            content += f'static {ret_type} {call}_callback({struct["args"]}) {{\n'
            content += "\tauto& hsaInterceptor = luthier::HsaInterceptor::instance();\n" \
                       f"\tauto apiId = HSA_API_ID_{call};\n" \
                       "\tbool isUserCallbackEnabled = hsaInterceptor.isUserCallbackEnabled(apiId);\n" \
                       "\tbool isInternalCallbackEnabled = hsaInterceptor.isInternalCallbackEnabled(apiId);\n" \
                       "\tbool shouldCallback = isUserCallbackEnabled || isInternalCallbackEnabled;\n"
            if ret_type != 'void':
                content += f'\t{ret_type} out{{}};\n'
            content += "\tif (shouldCallback) {\n" \
                       "\t\tauto& hsaUserCallback = hsaInterceptor.getUserCallback();\n" \
                       "\t\tauto& hsaInternalCallback = hsaInterceptor.getInternalCallback();\n" \
                       "\t\thsa_api_evt_args_t args;\n" \
                       "\t\tbool skipFunction{false};\n"
            for var in struct['alst']:
                item = struct['astr'][var]
                content += f"\t\targs.api_args.{call}.{var} = {var};\n"
            content += "\t\tif (isUserCallbackEnabled)\n" \
                       "\t\t\thsaUserCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId);\n" \
                       "\t\tif (isInternalCallbackEnabled)\n" \
                       "\t\t\thsaInternalCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction);\n" \
                       "\t\tif (!skipFunction)\n"
            if ret_type != 'void':
                content += f'\t\t\tout = '
            else:
                content += "\t\t\t"
            content += f'hsaInterceptor.getSavedHsaTables().{API_TABLE_NAMES[name]}.{call}_fn('
            for i, var in enumerate(struct['alst']):
                content += f'args.api_args.{call}.{var}'
                if i != len(struct['alst']) - 1:
                    content += ", "
            content += ');\n'
            content += "\t\tif (isUserCallbackEnabled)\n" \
                       "\t\t\thsaUserCallback(&args, LUTHIER_API_EVT_PHASE_EXIT, apiId);\n" \
                       "\t\tif (isInternalCallbackEnabled)\n" \
                       "\t\t\thsaInternalCallback(&args, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction);\n"
            if ret_type != 'void':
                content += "\t\treturn out;\n"
            content += "\t}\n" \
                       "\telse {\n"
            if ret_type != 'void':
                content += f'\t\tout = '
            else:
                content += "\t\t"
            content += f'hsaInterceptor.getSavedHsaTables().{API_TABLE_NAMES[name]}.{call}_fn('
            for i, var in enumerate(struct['alst']):
                content += f'{var}'
                if i != len(struct['alst']) - 1:
                    content += ", "
            content += ');\n'
            if ret_type != 'void':
                content += '\t\treturn out;\n\t}\n}\n\n'
            else:
                content += "\n\t}\n}\n\n"

        return content

    # generate API intercepting code
    def gen_intercept(self, n, name, call, struct):
        content = ''
        if n > 0 and call == '-':
            # if name != '-':
            #     content += f'  interceptTables_.{API_TABLE_NAMES[name]} = *table;\n'
            # else:
            content += '};\n'
        if n == 0 or (call == '-' and name != '-'):
            content += 'void luthier::HsaInterceptor::install' + name + 'Wrappers(' + name + 'Table* table) {\n'
            content += f'  savedTables_.{API_TABLE_NAMES[name]}' + ' = *table;\n'
        if call != '-':
            if call != 'hsa_shut_down':
                content += '  table->' + call + '_fn = ' + call + '_callback;\n'
            else:
                content += '  { void* p = (void*)' + call + '_callback; (void)p; }\n'
        return content


#############################################################
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
    api_table_names = ["CoreApiTable", "AmdExtTable", "ImageExtTable", "FinalizerExtTable"]
    args = parse_and_validate_args()
    hsa_include_files = tuple(os.path.join(args.hsa_include_dir, header_file) for header_file in
                              ("hsa.h", "hsa_ext_amd.h", "hsa_ext_image.h", "hsa_ext_finalize.h", "hsa_api_trace.h"))
    hsa_functions = parse_hsa_functions(hsa_include_files)

    parsed_api_trace_header = parse_header_file(os.path.join(args.hsa_include_dir, "hsa_api_trace.h"))

    api_tables = get_api_tables(parsed_api_trace_header, api_table_names)

    api_table_container = get_api_table_container(parsed_api_trace_header, "HsaApiTableContainer")

    api_table_container_fields = {}
    for table_container_field in api_table_container.fields:
        field_type = table_container_field.type.typename.segments[0].name
        if field_type in api_tables:
            api_table_container_fields[field_type] = table_container_field.name

    callback_defs = [f"/* Generated by {os.path.basename(__file__)}. DO NOT EDIT!*/",
                     "#include <hsa/hsa_api_trace.h>",
                     '#include "hsa_intercept.hpp"',
                     '#include "luthier_types.h"']
    callback_arguments_struct = ["typedef union {"]
    callback_enums = ["enum hsa_api_evt_id_t: unsigned int {",
                      "\tHSA_API_EVT_ID_FIRST = 0,"]
    wrapper_install_defs = []

    enum_idx = 0
    for api_name in api_table_names:
        api_table = api_tables[api_name]
        # Mark the beginning of the API Table in the enum and the argument struct
        callback_enums.append(f"\t// {api_name} API")
        # First create the wrapper install function's prototype
        print(api_name)
        wrapper_install_defs.append(f'void luthier::hsa::Interceptor::install{api_name}Wrappers({api_name} *Table) {{')
        wrapper_install_defs.append(f'\tSavedTable.{api_table_container_fields[api_name]}')
        # content = ''
        # if n > 0 and call == '-':
        #     # if name != '-':
        #     #     content += f'  interceptTables_.{API_TABLE_NAMES[name]} = *table;\n'
        #     # else:
        #     content += '};\n'
        # if n == 0 or (call == '-' and name != '-'):
        #     content += 'void luthier::HsaInterceptor::install' + name + 'Wrappers(' + name + 'Table* table) {\n'
        #     content += f'  savedTables_.{API_TABLE_NAMES[name]}' + ' = *table;\n'
        # if call != '-':
        #     if call != 'hsa_shut_down':
        #         content += '  table->' + call + '_fn = ' + call + '_callback;\n'
        #     else:
        #         content += '  { void* p = (void*)' + call + '_callback; (void)p; }\n'
        # return content
        for f in api_table.fields:
            # look for functions in the API Tables, not fields
            # function fields in the API table are defined as pointers to the decltype of their target HSA function
            if isinstance(f.type, cxx_types.Pointer) and isinstance(f.type.ptr_to.typename.segments[0],
                                                                    cxx_types.DecltypeSpecifier):
                hsa_function_name = f.type.ptr_to.typename.segments[0].tokens[0].value
                api_table_function_name = f.name
                hsa_function = hsa_functions[hsa_function_name]
                print(hsa_functions[hsa_function_name].return_type.format())
                # Format the args for later
                formatted_params = [p.format for p in hsa_functions[hsa_function_name].parameters]
                # params = ""
                # p_type = p.type
                # if isinstance(p_type, cxx_types.Type):
                #     params = f"{p_type.typename.segments[0].name} {p.name}"
                #     formatted_params.append(params)
                # elif isinstance(p_type, cxx_types.Pointer):
                #     ptr = p_type.ptr_to
                #     if isinstance(ptr, cxx_types.FunctionType):
                #         print(ptr)
                #     elif isinstance(ptr, cxx_types.Type):
                #         print(p.format())
                #         params = f"{ptr.const} {ptr.typename.segments[0].name} *{p.name}"
                # # if hasattr(p_type, "type_name"):
                # #     print(len(p_type.typename.segments))
                # # elif hasattr(p_type, "ptr_to"):
                # #     print(p_type.ptr_to.)
                # print(params)
                # Add the function to the enums list
                callback_enums.append(f"\tHSA_API_EVT_ID_{hsa_function_name} = {enum_idx},")
                # Generate the argument struct field
                callback_arguments_struct.append("\tstruct {")
                for p in formatted_params:
                    callback_arguments_struct.append(f"\t\t{p},")
                callback_arguments_struct.append("\t},")
                # Generate the callback
                # callback_defs.append(f'static {hsa_function.return_type.format()}'
                #                      f'{hsa_function_name}_callback({', '.format(formatted_params)}) {{')
                # callback_defs.append("\tauto& HsaInterceptor = luthier::Hsa::Interceptor::instance();")
                # callback_defs.append(f"\tauto apiId = HSA_API_ID_{hsa_function_name};")
                # callback_defs.append("\tbool isUserCallbackEnabled = hsaInterceptor.isUserCallbackEnabled(apiId);\n"
                #                      "\tbool isInternalCallbackEnabled = hsaInterceptor.isInternalCallbackEnabled(apiId);\n"
                #                      "\tbool shouldCallback = isUserCallbackEnabled || isInternalCallbackEnabled;\n")
                # if ret_type != 'void':
                #     content += f'\t{ret_type} out{{}};\n'
                # content += "\tif (shouldCallback) {\n" \
                #            "\t\tauto& hsaUserCallback = hsaInterceptor.getUserCallback();\n" \
                #            "\t\tauto& hsaInternalCallback = hsaInterceptor.getInternalCallback();\n" \
                #            "\t\thsa_api_evt_args_t args;\n" \
                #            "\t\tbool skipFunction{false};\n"
                # for var in struct['alst']:
                #     item = struct['astr'][var]
                # content += f"\t\targs.api_args.{call}.{var} = {var};\n"
                # content += "\t\tif (isUserCallbackEnabled)\n" \
                #            "\t\t\thsaUserCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId);\n" \
                #            "\t\tif (isInternalCallbackEnabled)\n" \
                #            "\t\t\thsaInternalCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId, &skipFunction);\n" \
                #            "\t\tif (!skipFunction)\n"
                # if ret_type != 'void':
                #     content += f'\t\t\tout = '
                # else:
                #     content += "\t\t\t"
                # content += f'hsaInterceptor.getSavedHsaTables().{API_TABLE_NAMES[name]}.{call}_fn('
                # for i, var in enumerate(struct['alst']):
                #     content += f'args.api_args.{call}.{var}'
                # if i != len(struct['alst']) - 1:
                #     content += ", "
                # content += ');\n'
                # content += "\t\tif (isUserCallbackEnabled)\n" \
                #            "\t\t\thsaUserCallback(&args, LUTHIER_API_EVT_PHASE_EXIT, apiId);\n" \
                #            "\t\tif (isInternalCallbackEnabled)\n" \
                #            "\t\t\thsaInternalCallback(&args, LUTHIER_API_EVT_PHASE_EXIT, apiId, &skipFunction);\n"
                # if ret_type != 'void':
                #     content += "\t\treturn out;\n"
                # content += "\t}\n" \
                #            "\telse {\n"
                # if ret_type != 'void':
                #     content += f'\t\tout = '
                # else:
                #     content += "\t\t"
                # content += f'hsaInterceptor.getSavedHsaTables().{API_TABLE_NAMES[name]}.{call}_fn('
                # for i, var in enumerate(struct['alst']):
                #     content += f'{var}'
                # if i != len(struct['alst']) - 1:
                #     content += ", "
                # content += ');\n'
                # if ret_type != 'void':
                #     content += '\t\treturn out;\n\t}\n}\n\n'
                # else:
                #     content += "\n\t}\n}\n\n"

                # Install the callback
                wrapper_install_defs.append(f'\tTable.{api_table_function_name} = {hsa_function_name};')
                # print(api_table.fields[1].type.ptr_to.typename.segments[0].tokens[0].value)

                # descr = ApiDescrParser(H_OUT, HSA_DIR, API_TABLES_H, API_HEADERS_H, LICENSE)
                #
                # out_file = PREFIX + CPP_OUT
                # print('Generating "' + out_file + '"')
                # f = open(out_file, 'w')
                # f.write(descr.cpp_content[:-1])
                # f.close()
                enum_idx += 1
            wrapper_install_defs.append("};")
    # Handle the kernel launch event separately
    callback_enums.append(f"\tHSA_API_EVT_ID_hsa_queue_packet_submit = {enum_idx}")
    # Finalize the API EVT ID Enum
    callback_enums.append(f"\tHSA_API_EVT_ID_LAST = {enum_idx},")
    callback_enums.append("}")
    # print(callback_enums)
    print("\n".join(callback_enums))

if __name__ == "__main__":
    main()
