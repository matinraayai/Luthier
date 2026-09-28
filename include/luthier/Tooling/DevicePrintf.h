//===-- DevicePrintf.h ------------------------------------------*- C++ -*-===//
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
/// \file
/// Renders device-side \c printf output on the host.
///
/// The AMDGPU backend lowers a device-side \c printf one of two ways, and
/// Luthier has to service both because it does not control how the code
/// object it loads was compiled:
///
/// \li <b>hostcall printf</b> (<tt>-mprintf-kind=hostcall</tt>, the default
///     for HIP on Linux). The kernel streams the format string and the
///     arguments to the host through the hostcall buffer, one
///     \c SERVICE_PRINTF message per call, and blocks until the host has
///     rendered it. See \c HostcallHandler.h.
/// \li <b>buffered printf</b> (<tt>-mprintf-kind=buffered</tt>). The kernel
///     bump-allocates a record inside the buffer passed as the
///     \c hidden_printf_buffer argument and moves on; the host drains the
///     buffer once the dispatch has completed. Constant format strings are
///     not written into the record — the record carries a hash, and the
///     string itself is recovered from the code object's \c amdhsa.printf
///     metadata.
///
/// Both paths converge on the same message encoding, so they share
/// \c formatDevicePrintfMessage. The rendering follows ROCclr's
/// <tt>rocclr/device/devhcprintf.cpp</tt> so that Luthier's output matches
/// what HIP would have produced for the same kernel.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_DEVICE_PRINTF_H
#define LUTHIER_HSA_TOOLING_DEVICE_PRINTF_H

#include <cstdint>
#include <cstdio>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <string>
#include <vector>

namespace luthier {

/// Renders one device \c printf message onto \p Stream.
///
/// \p Message is a sequence of 64-bit words holding, in order:
/// \li the format string, NUL-terminated and zero-padded up to an 8-byte
///     boundary;
/// \li one word per integer, floating-point or pointer argument;
/// \li each string argument inline, NUL-terminated and padded to 8 bytes.
///
/// The format string is split at its conversion specifiers and handed to the
/// C library one specifier at a time, so the host's \c printf does the actual
/// formatting. A \c \%n specifier renders nothing and consumes its argument,
/// matching HIP — it is never passed through to the C library.
///
/// \return the number of characters written, or a negative value if the
/// underlying \c vfprintf failed. Behaviour matches the host \c printf return
/// value.
int formatDevicePrintfMessage(std::FILE *Stream,
                              llvm::ArrayRef<uint64_t> Message);

/// Renders a device \c printf message that is prefixed by a control word,
/// which is how the hostcall \c SERVICE_PRINTF service delivers it.
///
/// Bit 0 of <tt>Message[0]</tt> selects the output stream (clear: \c stdout,
/// set: \c stderr); every other bit is reserved and must be zero. The rest of
/// \p Message is the message body \c formatDevicePrintfMessage expects.
///
/// \param [out] Output receives the value returned to the device: the
/// character count on success, or \c -1 if the control word was malformed.
void handleDevicePrintfHostcall(uint64_t *Output,
                                llvm::ArrayRef<uint64_t> Message);

//===----------------------------------------------------------------------===//
// Buffered printf (the hidden_printf_buffer argument)
//===----------------------------------------------------------------------===//

/// Constant format strings a code object's \c amdhsa.printf metadata carries,
/// keyed by the hash the device code writes in their place.
using PrintfFormatStringMap = llvm::DenseMap<uint64_t, std::string>;

/// Builds the hash-to-format-string map a buffered-printf drain needs out of
/// a code object's \c amdhsa.printf metadata entries.
///
/// Each entry follows the AMDGPU convention
/// <tt>"<id>:<num-args>:<arg-size>...:<format-string>"</tt>. For the buffered
/// printf scheme clang emits no argument sizes and encodes the string as
/// <tt>"<hash-in-hex>,<format-string>"</tt>, which is what this reads.
/// Entries that do not carry a hash are skipped rather than rejected — a
/// code object may mix in OpenCL-style entries this map has no use for.
llvm::Expected<PrintfFormatStringMap>
parsePrintfFormatStrings(llvm::ArrayRef<std::string> PrintfMetadata);

/// Bytes of \c hidden_printf_buffer Luthier hands to a kernel it dispatches
/// itself. Records that no longer fit are dropped by the device, so this only
/// bounds how much output a single dispatch can produce.
constexpr size_t DefaultPrintfBufferSize = 1U << 20;

/// Number of bytes at the base of a printf buffer that hold its two control
/// words: the running write offset, then the number of bytes available for
/// records.
constexpr size_t PrintfBufferHeaderSize = 2 * sizeof(uint32_t);

/// Prepares \p Buffer for a dispatch: zeroes it and writes the control words
/// the device bump-allocates against. \p Buffer must be at least
/// \c PrintfBufferHeaderSize bytes.
llvm::Error initializePrintfBuffer(llvm::MutableArrayRef<uint8_t> Buffer);

/// Renders every record the device wrote into \p Buffer, which must have been
/// prepared by \c initializePrintfBuffer and must not be concurrently written
/// — call this only once the dispatch has completed.
///
/// \param FormatStrings the map from \c parsePrintfFormatStrings, used to
/// resolve records that reference a constant format string by hash.
llvm::Error drainPrintfBuffer(llvm::ArrayRef<uint8_t> Buffer,
                              const PrintfFormatStringMap &FormatStrings);

} // namespace luthier

#endif // LUTHIER_HSA_TOOLING_DEVICE_PRINTF_H
