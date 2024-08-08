import argparse
import os
import subprocess

tmp_files = []


def assertion_gen(mir_file):
    print("Update assertions for", mir_file)
    update_assert_result = subprocess.Popen(
        [update_mir_assertions, "--llc-binary", llc_bin, mir_file])


def verify(mir_file):
    tmp = mir_file + ".tmp"
    print("Verify instructions in", mir_file)
    llc_result = subprocess.run(
        [llc_bin, "-mtriple=amdgcn", "-mcpu=gfx908",
         "--verify-machineinstrs", "-run-pass", "verify", "-o", tmp, mir_file])
    print(llc_result)

    filecheck_result = subprocess.run(
        [filecheck_bin, "-check-prefixes=GCN", "--input-file", tmp, mir_file])
    print(filecheck_result)
    tmp_files.append(tmp)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input_mir",
    default=None,
    help="Give a single MIR file as input. Please don't use this with -i-dir"
)
parser.add_argument(
    "-i-dir", "--input_dir",
    default=os.getenv('LUTHER_UNITTEST_OUTPUT_DIR', os.getcwd()),
    help="Directory containing MIR files. By default points to the current working directory. Can also be set using LUTHER_UNITTEST_OUTPUT_DIR"
)
parser.add_argument(
    "--llvm_bin_dir",
    default=os.getenv('LLVM_BIN_DIR', None),
    help="Path to llvm binaries."
)
parser.add_argument(
    "--llvm_util_dir",
    default=os.getenv('LLVM_UTIL_DIR', None),
    help="Path to llvm utils."
)

args = parser.parse_args()
llc_bin = args.llvm_bin_dir + "/llc"
filecheck_bin = args.llvm_bin_dir + "/FileCheck"
update_mir_assertions = args.llvm_util_dir + "/update_mir_test_checks.py"

print("==============================================")
print("Run MIR verification script")

if args.input_mir:
    mir_file = args.input_mir
    assertion_gen(mir_file)
    print()  # print a blank line
    verify(mir_file)
else:
    mir_dir = args.input_dir

    for file in os.listdir(mir_dir):
        if ".mir" not in file:
            continue
        assertion_gen(mir_dir + '/' + file)
        print()  # print a blank line

    print("----------------------------------------------")

    for file in os.listdir(mir_dir):
        if ".mir" not in file:
            continue
        verify(mir_dir + '/' + file)
        print()  # print a blank line

print("----------------------------------------------")
print("Cleaning temp files")
for tmp in tmp_files:
    os.remove(tmp)
print("Done")
print("==============================================")
