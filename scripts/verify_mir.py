import argparse
import os
import subprocess


def verify(mir_dir, mir_file):
    tmp = mir_dir + "/tmp"
    print("Verify instructions in", mir_file)
    print("Update assertions")
    update_assert_result = subprocess.Popen(
        [update_mir_assertions, "--llc-binary", llc_bin, mir_file])

    print("Run verification")
    llc_result = subprocess.run(
        [llc_bin, "-mtriple=amdgcn", "-mcpu=gfx908",
         "--verify-machineinstrs", "-run-pass", "verify", "-o", tmp, mir_file])
    print(llc_result)
    filecheck_result = subprocess.run(
        [filecheck_bin, "-check-prefixes=GCN", "--input-file", tmp, mir_file])
    print(filecheck_result)
    os.remove(tmp)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input_mir",
    default=None,
    help="Give a single MIR file as input. Please don't use this with -i-dir"
)
parser.add_argument(
    "-i-dir", "--input_dir",
    default=os.getenv('LUTHER_UNITTEST_OUTPUT_DIR', os.getcwd()),
    help="Directory containing MIR files. By default points to the output directory for the unittests"
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

if args.input_mir:
    mir_file = args.input_mir
    mir_dir = os.path.dirname(args.input_mir)
    verify(mir_dir, mir_file)
else:
    mir_dir = args.input_dir

    for file in os.listdir(mir_dir):
        if ".mir" not in file:
            continue
        mir_file = mir_dir + '/' + file
        verify(mir_dir, mir_file)

print("Done")
