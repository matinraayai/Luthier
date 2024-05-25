#ifndef CLONING_HPP
#define CLONING_HPP
#include <llvm/IR/Function.h>

namespace luthier {

/**
 * Clones the prototype and the content of the \c llvm::Function
 * \p F into the \p Module, as well as any related Values (e.g.
 * other \c llvm::Function s called, global variables used, etc).
 *
 * \note this function behaves similarly to \c llvm::cloneFunction, in
 * \file <llvm/Transforms/Utils/Cloning.h>; The main difference is that it
 * performs a deep copy of \p F into
 * \param F the original \c llvm::Function to be cloned
 * \param Module the target Module; Should not be the parent of \p F
 * \return a pointer to the cloned function, residing in \p Module
 */
llvm::Function *cloneFunctionIntoModule(const llvm::Function &F,
                                        llvm::Module &Module);

} // namespace luthier

#endif