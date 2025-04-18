import re
import io
from pcpp.preprocessor import Preprocessor, OutputDirective, Action
from cxxheaderparser.simple import parse_string, ParsedData
from typing import *

__all__ = ["ROCmPreprocessor", "parse_header_file"]


class ROCmPreprocessor(Preprocessor):
    def __init__(self, **argv):
        super(ROCmPreprocessor, self).__init__()
        self.line_directive = None
        self.passthru_unfound_includes = None

    def on_include_not_found(self, is_malformed, is_system_include, cur_dir, include_path):
        raise OutputDirective(Action.IgnoreAndPassThrough)


def parse_header_file(header_file: str, defines: Iterable[str]) -> ParsedData:
    """
    Parses the passed header file
    :param header_file: the location of the header file to be parsed
    :param defines: list of terms set to be defined before parsing the header
    :return: the Parsed header
    """
    preprocessor = ROCmPreprocessor()
    preprocessor.line_directive = None
    preprocessor.passthru_unfound_includes = True
    preprocessor.passthru_includes = re.compile(r".*")
    for define in defines:
        preprocessor.define(define)
    with open(header_file, 'r') as hf:
        preprocessor.parse(hf)
        str_io = io.StringIO()
        preprocessor.write(str_io)
    preprocessed_header = str_io.getvalue()
    parsed = parse_string(preprocessed_header)
    str_io.close()
    return parsed
