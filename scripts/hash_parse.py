import re
import os
from typing import Optional

def extract_llvm_revision(header_file: str) -> Optional[str]:
    """
    Reads a header file and extracts the LLVM_REVISION string if defined.
    """
    with open(header_file, 'r') as hf:
        for line in hf:
            match = re.match(r'#define LLVM_REVISION\s+"([^"]+)"', line)
            if match:
                return match.group(1)
    return None

def determine_src_dir(revision: Optional[str]) -> str:
    """
    Determines the source directory based on the LLVM_REVISION string.
    """
    base_dir = "/opt/luthier/src"
    return os.path.join(base_dir, revision) if revision else base_dir

if __name__ == "__main__":

    header_file = "/opt/luthier/llvm/include/llvm/Support/VCSRevision.h"  
    #Need to input proper header file path here
    revision = extract_llvm_revision(header_file)
    
    if revision:
        print("LLVM_REVISION:", revision)
    else:
        print("LLVM_REVISION not found")
    
    #src_dir = determine_src_dir(revision)
    #print("Source directory:", src_dir)
