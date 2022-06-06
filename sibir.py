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

if __name__ == "__main__":
    args = parser.parse_args()
    command = " ".join(args.commands)

    path = os.path.dirname(os.path.realpath(__file__))
    current_wd = os.getcwd()

    env = os.environ.copy()
    if "LD_LIBRARY_PATH" not in env:
        env["LD_LIBRARY_PATH"] = ""
    env["LD_LIBRARY_PATH"] = path + "/build/lib" + ":" + env["LD_LIBRARY_PATH"]
    env["LD_PRELOAD"] = path + "/build/lib/libsibir.so"

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=current_wd,
        env=env,
    )

    proc.communicate()
