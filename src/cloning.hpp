//===-- cloning.hpp - IR and MIR Cloning Utilities ------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains utility functions used to clone LLVM IR and MIR
/// Modules and Functions, used frequently by Luthier components involved in the
/// code generation process.
//===----------------------------------------------------------------------===//

#ifndef CLONING_HPP
#define CLONING_HPP
#include <llvm/IR/Function.h>

namespace luthier {

/// Sets the attributes of the \p NewModule into the values found in
/// \p OldModule
/// \param OldModule \c llvm::Module whose attributes is to be cloned
/// \param NewModule the destination of the cloning operation
void cloneModuleAttributes(const llvm::Module &OldModule,
                                           llvm::Module &NewModule);

/// Performs a deep copy of the \c llvm::GlobalValue 's contained in
/// \p DeepClonedOldValues, and puts them in the \p NewModule
/// \n
/// A deep copy clones both the declaration and definition (only if present) of
/// the old value over to the new module; A shallow copy only clones the
/// declaration. Both types of copies will create a new value in the
/// \p NewModule regardless \n
/// For now, \p DeepClonedOldValues should all have the same parent
/// \c llvm::Module; This requirement can be changed in the future if a
/// legitimate use case is found.
/// \n
/// Besides the Global Values, the named Metadata of the Old Module will also
/// be copied over to the new Module, with old values pointing to new ones
/// \n
/// If any of values in \p DeepClonedOldValues relate to other global values
/// in their parent module (e.g. a function that calls another function, an
/// alias referring to a variable), unless specified explicitly in
/// \p DeepClonedOldValues, only their declaration will be cloned. Below are
/// a few exceptions:
/// \n
/// 1. If the related value is of type \c llvm::Function and it doesn't have
/// an external-facing linkage. This includes device functions in HIP.
/// \n
/// 2. If the related value is of type \c llvm::GlobalAlias, and its underlying
/// value is an \c llvm::Function of non-external linkage type or the underlying
/// value itself is deeply cloned. Otherwise the Global Alias will be replaced
/// with a Value of its underlying type in the target Module.
/// \n
/// 3. If the related value is of type \c llvm::GlobalIFunc, then its resolver
/// function doesn't have an external-facing linkage or is deeply cloned.
/// Keep in mind that IFuncs are not implemented yet in the HSA standard so it
/// is not likely for this type of Value to be encountered for now.
/// \n
/// If
/// \param [in] DeepCloneOldValues A list of \c llvm::GlobalValue's to be cloned
/// that belong to the same \c llvm::Module
/// \param [in, out] NewModule The destination of the cloned Values
/// \return an \p llvm::Error if the passed values don't have the same parent, or
/// if the list is empty.
/// Module; Otherwise an \p llvm::Error::success()
llvm::Error cloneGlobalValuesIntoModule(
    llvm::ArrayRef<llvm::GlobalValue *> DeepCloneOldValues,
    llvm::Module &NewModule);

} // namespace luthier

#endif