import argparse
import os
import subprocess


def is_file(path):
    """Check if the given path to a file exists."""
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid file")


def get_code_objects(dir):
    """
        Recursively searches an the given directory for code object files.
        If another directory is found within the given path, it is also parsed
    """
    dir_contents = os.listdir(dir)
    code_obj = []
    for i in dir_contents:
        path = dir + i
        if os.path.isfile(path) and i.endswith(".out"):
            code_obj.append(path)
        elif os.path.isdir(path):
            for obj in get_code_objects(path + '/'):
                code_obj.append(obj)
    return code_obj


parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--unittest_executable",
    type=is_file,
    default=os.getenv('LUTHIER_UNITTEST_EXECUTABLE', None),
    help="Path to unittest executable. Can also be set with the environment var \'LUTHIER_UNITTEST_EXECUTABLE\'"
)
parser.add_argument(
    "-i", "--input_code_obj",
    type=is_file,
    help="Give a single code object file as input. Please don't use this with -i-dir"
)
parser.add_argument(
    "-i-dir", "--input_dir",
    default=None,
    help="Directory containing input code objects. May also be nested with other directories that have code object files. Please don't use this with -i"
)
parser.add_argument(
    "-o-dir", "--output_dir",
    default=os.getenv('LUTHIER_UNITTEST_OUTPUT_DIR', os.getcwd()),
    help="Directory for all unittest output files. Can also be set with the environment var \'LUTHIER_UNITTEST_OUTPUT_DIR\'"
)

args = parser.parse_args()
test_name = args.unittest_executable

code_objs = []
cmd_lst = []

print("==============================================")
print("Executing test found in:", test_name)
print("----------------------------------------------")

# If you are specifying an input dir, we are assuming you want to run this
# unit test on multiple code object files
if args.input_dir:
    code_objs = get_code_objects(args.input_dir)
    for obj in code_objs:
        cmd_lst.append([test_name, obj])
elif args.input_code_obj:
    cmd_lst.append([test_name, args.input_code_obj])
else:
    cmd_lst.append([test_name])

for cmd in cmd_lst:
    cmd.append("-o")
    cmd.append(args.output_dir)

for cmd in cmd_lst:
    print("Running command:", cmd)
    unittest_result = subprocess.run(cmd)
    print("----------------------------------------------")

print("Finished run_unittest script")
print("==============================================")
