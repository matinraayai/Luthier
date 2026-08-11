//===-- RegValueMetadata.h --------------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file RegValueMetadata.h
/// Helpers for attaching/reading register-provenance metadata produced by
/// \c TraceFunctionTranslator. Each translated IR value that represents the
/// content of a (slice of a) physical register is tagged so downstream
/// passes can trace how a value was constructed.
///
/// Two attachment sites are supported:
///
/// 1. Per-instruction MD kind \c "luthier.reg" attached to
///    <tt>llvm::Instruction</tt>s. The MDNode is a list of register
///    descriptors (a single instruction may represent more than one
///    register slice after simplification folds).
///
///    \code
///    !1 = !{!2, !3}
///    !2 = !{!"vgpr3",      i32 <MCRegEnum>, i32 <HalfWordOff>, i32 <NumHalves>}
///    !3 = !{!"sgpr[12:14]", i32 <BaseEnum>,  i32 12,            i32 4}
///    \endcode
///
/// 2. Function-level MD kind \c "luthier.entry_reg_map" attached to the
///    translated <tt>llvm::Function</tt>. Lets us tag values that are not
///    Instructions (function arguments, constants seeded at kernel/device
///    entry) with the register slice they represent.
///
///    \code
///    !4 = !{!5, !6}
///    !5 = !{Value %arg0, !"sgpr0", i32 <BaseEnum>, i32 0, i32 2}
///    !6 = !{i32 42,      !"src_scc", i32 <BaseEnum>, i32 0, i32 2}
///    \endcode
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_REG_VALUE_METADATA_H
#define LUTHIER_TOOL_CODE_GEN_REG_VALUE_METADATA_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/MC/MCRegister.h>

namespace llvm {
class Function;
class Instruction;
class LLVMContext;
class MDNode;
class Value;
} // namespace llvm

namespace luthier {

/// Identifies a half-word-aligned slice of a register file.
/// Round-trips losslessly to/from the translator's \c RegFileKey.
struct RegValueDesc {
  llvm::MCRegister BaseReg;     ///< Base register of the file (e.g. VGPR0)
  unsigned HalfWordOffset;      ///< Offset from \c BaseReg in 16-bit halves
  unsigned NumHalves;           ///< Slice width in 16-bit halves

  friend bool operator==(const RegValueDesc &A, const RegValueDesc &B) {
    return A.BaseReg == B.BaseReg && A.HalfWordOffset == B.HalfWordOffset &&
           A.NumHalves == B.NumHalves;
  }
};

/// Metadata kind name attached to translated IR \c Instructions.
inline constexpr llvm::StringLiteral RegValueMDKindName = "luthier.reg";

/// Metadata kind name attached to the translated \c Function for
/// non-Instruction value tags (Arguments / Constants).
inline constexpr llvm::StringLiteral EntryRegMapMDKindName =
    "luthier.entry_reg_map";

/// Build a single descriptor MDNode of the form
/// <tt>!{!"<name>", i32 BaseEnum, i32 Off, i32 N}</tt>.
llvm::MDNode *buildRegValueDescMD(llvm::LLVMContext &Ctx,
                                  const RegValueDesc &D,
                                  llvm::StringRef Name);

/// Attach the descriptor to \p I's \c luthier.reg list. If \p I already
/// has a tag containing \p D, this is a no-op. Otherwise the descriptor
/// is appended (preserving prior entries).
void attachRegValue(llvm::Instruction &I, const RegValueDesc &D,
                    llvm::StringRef Name);

/// Union the \c luthier.reg list of \p Src into \p Dst. Used when
/// \c simplifyInstruction folds one tagged instruction into another:
/// the surviving instruction must inherit the doomed one's reg tags.
void mergeRegValues(llvm::Instruction &Dst, const llvm::Instruction &Src);

/// Read back the descriptors attached to \p I (empty result if none).
/// Names are not returned — callers that care about display strings can
/// inspect the MDNode directly via \c I.getMetadata(RegValueMDKindName).
void getRegValues(const llvm::Instruction &I,
                  llvm::SmallVectorImpl<RegValueDesc> &Out);

/// Append a \c (Value, RegValueDesc, Name) entry to \p F's
/// \c luthier.entry_reg_map. Use for tagging Arguments and Constants
/// seeded as the initial value of a register.
void addEntryRegMapping(llvm::Function &F, llvm::Value *V,
                        const RegValueDesc &D, llvm::StringRef Name);

/// Convenience: builds the display name from \p D using a default scheme
/// then delegates. The default scheme is:
/// - "<prefix>N"        for a single 32-bit GPR slot (prefix = v/s/a)
/// - "<prefix>[i:j]"    for a multi-DWORD GPR slot
/// - "<basename>"       for a slice that exactly covers BaseReg
/// - "<basename>+h<o>:<n>" otherwise
/// The name is advisory; programmatic readers must use \c getRegValues.
std::string formatRegValueDescName(const RegValueDesc &D,
                                   llvm::StringRef BaseRegName);

} // namespace luthier

#endif // LUTHIER_TOOL_CODE_GEN_REG_VALUE_METADATA_H
