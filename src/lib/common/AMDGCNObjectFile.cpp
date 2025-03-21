#include <luthier/common/AMDGCNObjectFile.h>
#include <luthier/common/LuthierError.h>

namespace luthier {

bool AMDGCNElfSymbolRef::classof(const ELFSymbolRefWrapper *S) {
  return llvm::isa<AMDGCNObjectFile>(S->getObject());
}

const luthier::AMDGCNObjectFile *AMDGCNElfSymbolRef::getObject() const {
  return llvm::cast<luthier::AMDGCNObjectFile>(
      llvm::object::ELFSymbolRef::getObject());
}

llvm::Expected<bool> AMDGCNElfSymbolRef::isKernelDescriptor() const {
  llvm::Expected<llvm::StringRef> SymNameOrErr = getName();
  LUTHIER_RETURN_ON_ERROR(SymNameOrErr.takeError());
  uint8_t Binding = getBinding();
  uint64_t Size = getSize();
  return (Binding == llvm::ELF::STT_OBJECT && SymNameOrErr->ends_with(".kd") &&
          Size == 64) ||
         (Binding == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64);
}

llvm::Expected<bool> AMDGCNElfSymbolRef::isVariable() const {
  llvm::Expected<bool> IsKdOrErr = isKernelDescriptor();
  LUTHIER_RETURN_ON_ERROR(IsKdOrErr.takeError());
  return getBinding() == llvm::ELF::STT_OBJECT && !*IsKdOrErr;
}

llvm::Expected<bool> AMDGCNElfSymbolRef::isKernelFunction() const {
  if (getELFType() != llvm::ELF::STT_FUNC)
    return false;

  llvm::Expected<llvm::StringRef> SymbolNameOrErr = getName();
  LUTHIER_RETURN_ON_ERROR(SymbolNameOrErr.takeError());

  auto SymbolIfFoundOrError =
      getObject()->lookupSymbol((*getName() + ".kd").str());
  LUTHIER_RETURN_ON_ERROR(SymbolIfFoundOrError.takeError());

  return SymbolIfFoundOrError->has_value();
}

llvm::Expected<bool> AMDGCNElfSymbolRef::isDeviceFunction() const {
  if (getELFType() != llvm::ELF::STT_FUNC)
    return false;
  return !*isKernelFunction();
}

llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
AMDGCNObjectFile::createAMDGCNObjectFile(llvm::StringRef Elf) {
  std::unique_ptr<llvm::MemoryBuffer> Buffer =
      llvm::MemoryBuffer::getMemBuffer(Elf, "", false);
  llvm::Expected<std::unique_ptr<ObjectFile>> ObjectFile =
      ObjectFile::createELFObjectFile(*Buffer);
  LUTHIER_RETURN_ON_ERROR(ObjectFile.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      ObjectFile.get() != nullptr, "Created object file is nullptr."));
  return llvm::unique_dyn_cast<AMDGCNObjectFile>(std::move(*ObjectFile));
}

llvm::Expected<std::unique_ptr<AMDGCNObjectFile>>
AMDGCNObjectFile::createAMDGCNObjectFile(llvm::ArrayRef<uint8_t> Elf) {
  return createAMDGCNObjectFile(llvm::toStringRef(Elf));
}

bool luthier::AMDGCNObjectFile::classof(const llvm::object::Binary *v) {
  return llvm::isa<luthier::ELF64LEObjectFileWrapper>(v) &&
         llvm::cast<luthier::ELF64LEObjectFileWrapper>(v)
             ->makeTriple()
             .isAMDGCN();
}

} // namespace luthier