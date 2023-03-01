#!/usr/bin/python3

import os
import argparse
import subprocess

parser = argparse.ArgumentParser(
    description="""Sibir is a binary instrumentation tool for AMD GPUs."""
)
parser.add_argument(
    "commands", metavar="commands", nargs="+", help="Commands to execute"
)
parser.add_argument(
    "-i", "--instrumentator", help="Path to instrumentation function"
)

if __name__ == "__main__":
    args = parser.parse_args()
    command = " ".join(args.commands)


    path = os.path.dirname(os.path.realpath(__file__))
    current_wd = os.getcwd()

    env = os.environ.copy()
    if "LD_LIBRARY_PATH" not in env:
        env["LD_LIBRARY_PATH"] = ""
    env["LD_LIBRARY_PATH"] = path + "/lib" + ":" + env["LD_LIBRARY_PATH"]
    env["LD_PRELOAD"] = path + "/lib/libsibir.so"

    if args.instrumentator != None:
        instrupath = os.path.abspath(args.instrumentator)
        env["INSTRU_FUNC"] = instrupath
    else:
        print("no instru func specified")

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=current_wd,
        env=env,
    )

    proc.communicate()
