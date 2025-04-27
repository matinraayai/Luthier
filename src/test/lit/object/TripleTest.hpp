#ifndef LUTHIER_TEST_LIT_OBJECT_TRIPLE_TEST_HPP
#define LUTHIER_TEST_LIT_OBJECT_TRIPLE_TEST_HPP
#include <luthier/common/ErrorCheck.h>
#include <luthier/object/ObjectUtils.h>

inline llvm::Error
performTargetTripleTest(const llvm::object::ObjectFile &ObjFile,
                        llvm::raw_ostream &OS) {
  llvm::Expected<std::string> TripleStrOrErr =
      luthier::object::getObjectFileTarget(ObjFile);
  LUTHIER_REPORT_FATAL_ON_ERROR(TripleStrOrErr.takeError());
  OS << "Target Triple: " << *TripleStrOrErr << "\n";
  return llvm::Error::success();
}

#endif