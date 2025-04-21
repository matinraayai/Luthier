import os

import lit.formats
import lit.util

config.name = "Luthier"
config.suffixes = {".hip", ".cl", ".c", ".cpp"}
config.test_format = lit.formats.ShTest(True)

config.excludes = ["comgr"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.my_obj_root