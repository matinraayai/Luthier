import os

import lit.formats
import lit.util

config.name = "Luthier"
config.suffixes = {".s", ".luthier", ".ll", ".cpp"}
config.test_format = lit.formats.ShTest(True)

config.excludes = ["comgr", "Inputs"]

config.substitutions.append(
    ("%luthier-opt-plugin",
     os.path.join(config.my_obj_root, "..", "..", "..", "lib", "LuthierToolCodeGenPlugin.so"))
)

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.my_obj_root

# Opt-in capture of a RUN line's stdout to <build>/.../Output/<test>.tmp.out
# for post-mortem inspection. Tests that want the capture should pipe through
# the %tee_out substitution between their command and FileCheck, e.g.:
#
#     // RUN: luthier-llc ... | %tee_out FileCheck %s
#
# Enable by exporting LIT_SAVE_OUTPUT=1 (or any non-empty value) before
# running lit. When the env var is unset the substitution is empty, so the
# pipeline degrades to a plain `| FileCheck`.
if os.environ.get("LIT_SAVE_OUTPUT"):
    config.substitutions.append(("%tee_out", "tee %t.out |"))
else:
    config.substitutions.append(("%tee_out", ""))