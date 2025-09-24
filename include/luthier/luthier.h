//===-- luthier.h - Luthier High-level Interface  ---------------*- C++ -*-===//
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
/// This file defines the public-facing interface of Luthier.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_H
#define LUTHIER_H
/// HIP_ENABLE_WARP_SYNC_BUILTINS enables the warp sync built-ins in HIP
/// Must be defined before HIP runtime headers are included
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <llvm/Support/Error.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/consts.h>
/// Undef the ill-defined \c ICMP_NE in HIP headers
#undef ICMP_NE
#include <luthier/hsa/Instr.h>
#include <luthier/hsa/KernelDescriptor.h>
#include <luthier/hsa/LoadedCodeObjectKernel.h>
#include <luthier/hsa/LoadedCodeObjectSymbol.h>
#include <luthier/intrinsic/Intrinsics.h>
#include <luthier/tooling/InstrumentationTask.h>
#include <luthier/tooling/LiftedRepresentation.h>
#include <luthier/types.h>

namespace luthier {

//===----------------------------------------------------------------------===//
//  Inspection APIs
//===----------------------------------------------------------------------===//

/// Disassembles the \p Kernel into a list of <tt>hsa::Instr</tt>\n
/// Disassembly only occurs on the first time this function is invoked on
/// the \p Kernel. Subsequent calls will use a result cached internally\n
/// \note This function only provides a raw LLVM MC view of the instructions;
/// For instrumentation, use <tt>lift</tt> instead
/// \param Kernel the kernel symbol to be disassembled
/// \return an \c llvm::ArrayRef to an internally cached vector of
/// <tt>hsa::Instr</tt>s, or an \c llvm::Error if an issue was encountered
/// during the process
llvm::Expected<llvm::ArrayRef<hsa::Instr>>
disassemble(const hsa::LoadedCodeObjectKernel &Kernel);

/// Disassembles the \p Func into a list of <tt>hsa::Instr</tt>.\n
/// Disassembly only occurs on the first time this function is invoked on
/// \p Func. Subsequent calls will use a result cached internally.\n
/// \note This function only provides a raw LLVM MC view of the instructions;
/// For instrumentation, use <tt>lift</tt> instead
/// \param Func the device function to be disassembled
/// \return an \c llvm::ArrayRef to an internally cached vector of
/// <tt>hsa::Instr</tt>s, or an \c llvm::Error if an issue was encountered
/// during the process
llvm::Expected<llvm::ArrayRef<hsa::Instr>>
disassemble(const hsa::LoadedCodeObjectDeviceFunction &Func);

/// Lifts the given \p Kernel and return a reference to its
/// <tt>LiftedRepresentation</tt>.\n
/// The lifted result gets cached internally on the first invocation.
/// \param [in] Kernel an \c hsa::LoadedCodeObjectKernel to be lifted
/// \return a reference to the internally-cached <tt>LiftedRepresentation</tt>
/// if successful, or an \c llvm::Error describing the issue encountered.
/// \sa LiftedRepresentation, \sa lift
llvm::Expected<const luthier::LiftedRepresentation &>
lift(const hsa::LoadedCodeObjectKernel &Kernel);

//===----------------------------------------------------------------------===//
//  Instrumentation API
//===----------------------------------------------------------------------===//

/// Instruments the \p LR by applying the \p Mutator to it
/// \param LR the \c LiftedRepresentation about to be instrumented
/// \param Mutator a function that instruments and modifies the \p LR
/// \return returns a new \c LiftedRepresentation containing the instrumented
/// code, or an \c llvm::Error if an issue was encountered during
/// instrumentation
llvm::Expected<std::unique_ptr<LiftedRepresentation>>
instrument(const LiftedRepresentation &LR,
           llvm::function_ref<llvm::Error(InstrumentationTask &,
                                          LiftedRepresentation &)>
               Mutator);

/// Applies the assembly printer pass on the \p LR to generate object files or
/// assembly files for each of its <tt>llvm::Module</tt>s and
/// <tt>llvm::MachineModuleInfo</tt>s
/// \note After printing, and all of the <tt>LR</tt>'s
/// <tt>llvm::MachineModuleInfo</tt>s will be deleted; This is due to an LLVM
/// design shortcoming which is being worked on
/// \param [in] LR the \c LiftedRepresentation to be printed into an assembly
/// file; Its \c llvm::TargetMachine can be used to control the
/// \c llvm::TargetOptions of the compilation process
/// \param [out] CompiledObjectFiles the printed assembly file
/// \param [in] FileType Type of the assembly file printed; Can be either
/// \c llvm::CodeGenFileType::AssemblyFile or
/// \c llvm::CodeGenFileType::ObjectFile
/// \return an \c llvm::Error in case of any issues encountered during the
/// process
llvm::Error printLiftedRepresentation(
    LiftedRepresentation &LR, llvm::SmallVectorImpl<char> &CompiledObjectFile,
    llvm::CodeGenFileType FileType = llvm::CodeGenFileType::ObjectFile);

// TODO: Implement link to executable, and load methods individually +
//  update the instrumentAndLoad docs

/// Instruments the <tt>Kernel</tt>'s lifted representation \p LR by
/// applying the instrumentation task <tt>ITask</tt> to it.\n After
/// instrumentation, loads the instrumented code onto the same device as the
/// \p Kernel
/// \param Kernel the kernel that's about to be instrumented
/// \param LR the lifted representation of the \p Kernel
/// \param ITask the instrumentation task, describing the instrumentation to
/// be performed on the <tt>kernel</tt>'s <tt>LR</t>
/// \return an \c llvm::Error describing if the operation succeeded or
/// failed
llvm::Error
instrumentAndLoad(const hsa::LoadedCodeObjectKernel &Kernel,
                  const LiftedRepresentation &LR,
                  llvm::function_ref<llvm::Error(InstrumentationTask &,
                                                 LiftedRepresentation &)>
                      Mutator,
                  llvm::StringRef Preset);

/// Checks if the \p Kernel is instrumented under the given \p Preset or not
/// \param [in] Kernel the \c hsa::LoadedCodeObjectKernel of the app
/// \param [in] Preset the preset name the kernel was instrumented under
/// \return on success, returns \c true if the \p Kernel is instrumented, \c
/// false otherwise. Returns an \c llvm::Error if the \p Kernel HSA symbol
/// handle is invalid
llvm::Expected<bool>
isKernelInstrumented(const hsa::LoadedCodeObjectKernel &Kernel,
                     llvm::StringRef Preset);

/// Overrides the kernel object field of the Packet with its instrumented
/// version under the given \p Preset, forcing HSA to launch the
/// instrumented version instead\n Modifies the rest of the launch
/// configuration (e.g. private segment size) if needed\n Note that this
/// function should be called every time an instrumented kernel needs to be
/// launched, since the content of the dispatch packet will always be set by
/// the target application to the original, un-instrumented version\n To
/// launch the original version of the kernel, simply refrain from calling
/// this function
/// \param Packet Packet the HSA dispatch packet intercepted from an HSA
/// queue,
// containing the kernel launch parameters/configuration
/// \param Preset the preset the kernel was instrumented under
/// \return an \c llvm::Error reporting
llvm::Error overrideWithInstrumented(hsa_kernel_dispatch_packet_t &Packet,
                                     llvm::StringRef Preset);

/// \brief If a tool contains an instrumentation hook it \b must
/// use this macro once. Luthier hooks are annotated via the the
/// \p LUTHIER_HOOK_CREATE macro. \n
///
/// \p MARK_LUTHIER_DEVICE_MODULE macro defines a managed variable of
/// type \p char named \p __luthier_reserved in the tool device code.
/// This managed variable ensures that: \n
/// 1. <b>The HIP runtime is forced to load the tool code object before the
/// first HIP kernel is launched on the device, without requiring eager binary
/// loading to be enabled</b>: The Clang compiler embeds the device code of a
/// Luthier tool and its bitcode into a static HIP FAT binary bundled within the
/// tool's shared object. During runtime, the tool's FAT binary gets
/// registered with the HIP runtime; However, by default, the HIP runtime loads
/// FAT binaries in a lazy fashion, only loading it onto a device if:
/// a. a kernel is launched from it on the said device, or
/// b. it contains a managed variable. \n
/// Including a managed variable is the only way to ensure the tool's FAT binary
/// is loaded in time without interfering with the loading mechanism of HIP
/// runtime.
/// \n
/// 2. <b>Luthier can easily identify a tool's code object by a constant time
/// symbol hash lookup</b>.
/// \n
/// If the target application is not using the HIP runtime, then no kernel is
/// launched by the HIP runtime, meaning that the tool FAT binary does not ever
/// get loaded. In that scenario, as the HIP runtime is present solely for
/// Luthier's function, the `HIP_ENABLE_DEFERRED_LOADING` environment
/// variable must be set to zero to ensure Luthier tool code objects get loaded
/// right away on all devices.
/// \sa LUTHIER_HOOK_ANNOTATE
#define MARK_LUTHIER_DEVICE_MODULE                                             \
  __attribute__((managed, used)) char LUTHIER_RESERVED_MANAGED_VAR = 0;

#define LUTHIER_HOOK_ANNOTATE                                                  \
  __attribute__((                                                              \
      device, used,                                                            \
      annotate(LUTHIER_STRINGIFY(LUTHIER_HOOK_ATTRIBUTE)))) extern "C" void

#define LUTHIER_EXPORT_HOOK_HANDLE(HookName)                                   \
  __attribute__((global, used)) extern "C" void LUTHIER_CAT(                   \
      LUTHIER_HOOK_HANDLE_PREFIX, HookName)(){};

#define LUTHIER_GET_HOOK_HANDLE(HookName)                                      \
  reinterpret_cast<const void *>(                                              \
      LUTHIER_CAT(LUTHIER_HOOK_HANDLE_PREFIX, HookName))
} // namespace luthier

#endif
