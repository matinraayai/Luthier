from pcpp.preprocessor import Preprocessor, OutputDirective, Action

__all__ = ["ROCmPreprocessor"]


class ROCmPreprocessor(Preprocessor):
    def __init__(self, **argv):
        super(ROCmPreprocessor, self).__init__()
        self.line_directive = None
        self.passthru_unfound_includes = None

    def on_include_not_found(self, is_malformed, is_system_include, cur_dir, include_path):
        raise OutputDirective(Action.IgnoreAndPassThrough)
