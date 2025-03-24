//===-- Disassembler.hpp - Luthier's Disassembler Singleton  --------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
///
/// \file
/// This file describes Luthier's Disassembler, a singleton in charge of
/// disassembling object file function symbols into MC representation, as
/// well as caching the disassembled instructions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_DISASSEMBLER_HPP
#define LUTHIER_TOOLING_COMMON_DISASSEMBLER_HPP
#include "common/Singleton.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/SubtargetFeature.h>
#include <luthier/lift/Instr.h>
#include <luthier/llvm/DenseMapInfo.h>
#include <shared_mutex>

namespace luthier {

class LLVMManager;

class Disassembler : public Singleton<Disassembler> {
private:
  /// Mutex protecting internal components of Disassembler
  std::shared_mutex Mutex;

  /// Reference to the LLVM Manager; Ensures the disassembler is initialized
  /// After \c LLVMManager
  LLVMManager &LLVMManager;

  /// Struct used to hold information required for disassembling instructions
  /// for each \c ISAInfo
  struct DisassemblyInfo {
    const llvm::Target *Target{nullptr};
    std::unique_ptr<llvm::MCRegisterInfo> MRI{nullptr};
    std::unique_ptr<llvm::MCAsmInfo> MAI{nullptr};
    std::unique_ptr<llvm::MCInstrInfo> MII{nullptr};
    std::unique_ptr<llvm::MCInstrAnalysis> MIA{nullptr};
    std::unique_ptr<llvm::MCSubtargetInfo> STI{nullptr};
    std::unique_ptr<llvm::MCInstPrinter> IP{nullptr};
    std::unique_ptr<llvm::TargetOptions> TargetOptions{nullptr};
    std::unique_ptr<llvm::MCContext> Ctx{nullptr};
    std::unique_ptr<llvm::MCDisassembler> DisAsm{nullptr};
  };

  /// Mapping between each ISA string and the \c DisassemblyInfo used
  /// to disassemble its instructions
  llvm::StringMap<std::unique_ptr<DisassemblyInfo>> DisassemblyInfoMap{};


  /// On success, returns a reference to the \c DisassemblyInfo associated with
  /// the ISA of the \p ObjFile
  /// \note A unique lock over the \c Disassembler mutex must be acquired before
  /// running this function
  /// \return on success, a reference to the \c DisassemblyInfo associated with
  /// the given \p ObjFile, on failure, an \c llvm::Error if an issue is
  /// encountered
  llvm::Expected<DisassemblyInfo &>
  getDisassemblyInfo(const llvm::object::ObjectFile &ObjFile);

  /// On success, returns a reference to the \c DisassemblyInfo associated with
  /// the given ISA parameters. Creates the info if not already present in the
  /// \c DisassemblyInfoMap
  /// \note A unique lock over the \c Disassembler mutex must be acquired before
  /// running this function
  /// \param TT the target triple of the ISA
  /// \param CPUName the name of the target CPU; <tt>"unknown"</tt> must be
  /// passed if no CPU is specified
  /// \param STF the sub-target features of the ISA
  /// \return on success, a reference to the \c DisassemblyInfo associated with
  /// the specified ISA, on failure, an \c llvm::Error describing the issue
  /// encountered
  llvm::Expected<DisassemblyInfo &>
  getDisassemblyInfo(const llvm::Triple &TT, llvm::StringRef CPUName,
                     const llvm::SubtargetFeatures &STF);

  /// Cache of function symbols already disassembled
  /// The vector handles themselves are allocated as a unique pointer to
  /// stop the map from calling its destructor prematurely.\n
  /// TODO: Invalidate entries in this map when object cache is reworked
  llvm::DenseMap<llvm::object::SymbolRef,
                 std::unique_ptr<llvm::SmallVector<luthier::Instr>>>
      MCDisassembledSymbols{};

public:
  explicit Disassembler(luthier::LLVMManager &LLVMManager)
      : LLVMManager(LLVMManager), Singleton<Disassembler>() {};


  /// Disassembles the contents of the \p Symbol and returns a list
  /// of instructions for inspection
  /// \param Symbol the ELF symbol of type \c STT_FUNC to be disassembled
  /// \param LoadedContents Encapsulates the loaded image of the ELF, if
  /// exists. If passed, the disassembler will disassemble the loaded contents
  /// instead of the contents of the ELF object file. Must be accessible from
  /// the host
  /// \return on success, returns a reference to a cached list of
  /// <tt>Instr</tt>s; an \c llvm::Error on failure
  llvm::Expected<llvm::ArrayRef<luthier::Instr>>
  disassemble(const llvm::object::ELFSymbolRef &Symbol,
              llvm::ArrayRef<uint8_t> LoadedContents = {});



};

} // namespace luthier

#endif