//===-- DevicePrintf.cpp --------------------------------------------------===//
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
#include "luthier/HSATooling/DevicePrintf.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"

#include <cstdarg>
#include <cstring>
#include <llvm/Support/Endian.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

namespace {

/// Forwards \p Fmt and its arguments to the C library and accumulates the
/// character count the way \c printf reports it: once a call fails, the count
/// stays negative.
///
/// The format string necessarily comes from the device code rather than from
/// a literal, which is the entire point of servicing \c printf, so the
/// non-literal-format diagnostics are silenced here rather than at every call
/// site.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#pragma GCC diagnostic ignored "-Wformat-security"
#endif
void printfAndCount(std::FILE *Stream, int &OutCount, const char *Fmt, ...) {
  va_list Args;
  va_start(Args, Fmt);
  const int Written = std::vfprintf(Stream, Fmt, Args);
  va_end(Args);
  OutCount = Written < 0 ? Written : OutCount + Written;
}
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

/// Number of \c '*' placeholders (dynamic field width / precision) in the
/// conversion specifier \p Spec.
unsigned countStars(llvm::StringRef Spec) { return Spec.count('*'); }

template <typename... ArgTypes>
const uint64_t *consumeInteger(std::FILE *Stream, int &OutCount,
                               const std::string &Spec, const uint64_t *Ptr,
                               ArgTypes... Args) {
  printfAndCount(Stream, OutCount, Spec.c_str(), Args..., Ptr[0]);
  return Ptr + 1;
}

template <typename... ArgTypes>
const uint64_t *consumeFloatingPoint(std::FILE *Stream, int &OutCount,
                                     const std::string &Spec,
                                     const uint64_t *Ptr, ArgTypes... Args) {
  double D;
  std::memcpy(&D, Ptr, sizeof(D));
  printfAndCount(Stream, OutCount, Spec.c_str(), Args..., D);
  return Ptr + 1;
}

/// String arguments travel inline in the message rather than as a pointer
/// into device memory, so the specifier consumes as many words as the string
/// occupies once padded out to an 8-byte boundary.
template <typename... ArgTypes>
const uint64_t *consumeCString(std::FILE *Stream, int &OutCount,
                               const std::string &Spec, const uint64_t *Ptr,
                               const uint64_t *End, ArgTypes... Args) {
  const auto *Str = reinterpret_cast<const char *>(Ptr);
  const size_t Available = (End - Ptr) * sizeof(uint64_t);
  // A message whose trailing string is not NUL-terminated is malformed; stop
  // rather than walk off the end of the buffer.
  const size_t Length = ::strnlen(Str, Available);
  if (Length == Available)
    return End;
  printfAndCount(Stream, OutCount, Spec.c_str(), Args..., Str);
  return Ptr + (Length + 1 + sizeof(uint64_t) - 1) / sizeof(uint64_t);
}

template <typename... ArgTypes>
const uint64_t *consumePointer(std::FILE *Stream, int &OutCount,
                               const std::string &Spec, const uint64_t *Ptr,
                               ArgTypes... Args) {
  auto *VPtr = reinterpret_cast<void *>(static_cast<uintptr_t>(Ptr[0]));
  printfAndCount(Stream, OutCount, Spec.c_str(), Args..., VPtr);
  return Ptr + 1;
}

/// Renders the single conversion specifier \p Spec against the argument at
/// \p Ptr, with any leading \c '*' placeholders already peeled off into
/// \p Args.
/// \return a pointer to the first word past the consumed argument, or \p End
/// if the specifier is one this cannot render.
template <typename... ArgTypes>
const uint64_t *consumeArgument(std::FILE *Stream, int &OutCount,
                                const std::string &Spec, const uint64_t *Ptr,
                                const uint64_t *End, ArgTypes... Args) {
  switch (Spec.back()) {
  case 'd':
  case 'i':
  case 'o':
  case 'u':
  case 'x':
  case 'X':
  case 'c':
    return consumeInteger(Stream, OutCount, Spec, Ptr, Args...);
  case 'f':
  case 'F':
  case 'e':
  case 'E':
  case 'g':
  case 'G':
  case 'a':
  case 'A':
    return consumeFloatingPoint(Stream, OutCount, Spec, Ptr, Args...);
  case 's':
    return consumeCString(Stream, OutCount, Spec, Ptr, End, Args...);
  case 'p':
    return consumePointer(Stream, OutCount, Spec, Ptr, Args...);
  case 'n':
    // Writing back through a device pointer is meaningless here, and handing
    // %n to the C library would let device code scribble over host memory.
    // HIP skips it too.
    return Ptr + 1;
  default:
    // An unrecognized conversion means the rest of the message can no longer
    // be located; give up on it rather than guess.
    return End;
  }
}

/// Peels the dynamic field-width / precision arguments off \p Ptr and renders
/// \p Spec with them.
const uint64_t *processSpecifier(std::FILE *Stream, int &OutCount,
                                 const std::string &Spec, const uint64_t *Ptr,
                                 const uint64_t *End) {
  switch (countStars(Spec)) {
  case 0:
    return consumeArgument(Stream, OutCount, Spec, Ptr, End);
  case 1:
    if (End - Ptr < 2)
      return End;
    return consumeArgument(Stream, OutCount, Spec, Ptr + 1, End, Ptr[0]);
  case 2:
    if (End - Ptr < 3)
      return End;
    return consumeArgument(Stream, OutCount, Spec, Ptr + 2, End, Ptr[0],
                           Ptr[1]);
  default:
    // A specifier cannot carry more than a width and a precision.
    return End;
  }
}

/// The AMDGPU metadata escape for a colon inside a format string, which would
/// otherwise be taken for a field separator.
constexpr llvm::StringLiteral ColonEscape = "\\72";

/// Undoes the escaping \c parsePrintfFormatStrings' input applies to colons.
std::string unescapeMetadataString(llvm::StringRef Escaped) {
  std::string Out;
  Out.reserve(Escaped.size());
  while (!Escaped.empty()) {
    if (Escaped.starts_with(ColonEscape)) {
      Out += ':';
      Escaped = Escaped.drop_front(ColonEscape.size());
      continue;
    }
    Out += Escaped.front();
    Escaped = Escaped.drop_front();
  }
  return Out;
}

} // namespace

//===----------------------------------------------------------------------===//
// formatDevicePrintfMessage
//===----------------------------------------------------------------------===//

int formatDevicePrintfMessage(std::FILE *Stream,
                              llvm::ArrayRef<uint64_t> Message) {
  if (Message.empty())
    return 0;

  const uint64_t *Ptr = Message.data();
  const uint64_t *const End = Message.data() + Message.size();

  // The format string leads the message, NUL-terminated and padded out to a
  // word boundary. Bound the search by the message so a device that forgot
  // the terminator cannot walk off the end.
  const auto *FmtStart = reinterpret_cast<const char *>(Ptr);
  const size_t FmtAvailable = Message.size() * sizeof(uint64_t);
  const size_t FmtLength = ::strnlen(FmtStart, FmtAvailable);
  if (FmtLength == FmtAvailable)
    return 0;
  const std::string Fmt(FmtStart, FmtLength);
  Ptr += (FmtLength + 1 + sizeof(uint64_t) - 1) / sizeof(uint64_t);

  static constexpr char ConversionSpecifiers[] = "diouxXfFeEgGaAcspn";

  int OutCount = 0;
  size_t Point = 0;
  while (true) {
    // Everything between two specifiers is literal text and goes out as-is.
    size_t Mark = Point;
    Point = Fmt.find('%', Point);
    if (Point == std::string::npos) {
      printfAndCount(Stream, OutCount, "%s", Fmt.c_str() + Mark);
      return OutCount;
    }
    printfAndCount(Stream, OutCount, "%.*s", static_cast<int>(Point - Mark),
                   Fmt.c_str() + Mark);
    if (OutCount < 0)
      return OutCount;

    Mark = Point;
    ++Point;

    if (Point < Fmt.size() && Fmt[Point] == '%') {
      printfAndCount(Stream, OutCount, "%%");
      if (OutCount < 0)
        return OutCount;
      ++Point;
      continue;
    }

    // A specifier with no argument left to render ends the message.
    if (Ptr == End)
      return OutCount;

    Point = Fmt.find_first_of(ConversionSpecifiers, Point);
    if (Point == std::string::npos)
      return OutCount;
    ++Point;

    // [Mark, Point) now spans one complete conversion specifier.
    Ptr = processSpecifier(Stream, OutCount, Fmt.substr(Mark, Point - Mark),
                           Ptr, End);
    if (OutCount < 0)
      return OutCount;
  }
}

void handleDevicePrintfHostcall(uint64_t *Output,
                                llvm::ArrayRef<uint64_t> Message) {
  if (Message.empty()) {
    *Output = static_cast<uint64_t>(-1);
    return;
  }

  /// Only bit 0 of the control word is defined; it selects the stream.
  static constexpr uint64_t ControlMask = 1;
  const uint64_t Control = Message.front();
  if ((Control & ~ControlMask) != 0) {
    *Output = static_cast<uint64_t>(-1);
    return;
  }

  std::FILE *Stream = (Control & ControlMask) ? stderr : stdout;
  *Output = static_cast<uint64_t>(
      formatDevicePrintfMessage(Stream, Message.drop_front()));
}

//===----------------------------------------------------------------------===//
// parsePrintfFormatStrings
//===----------------------------------------------------------------------===//

llvm::Expected<PrintfFormatStringMap>
parsePrintfFormatStrings(llvm::ArrayRef<std::string> PrintfMetadata) {
  PrintfFormatStringMap Out;
  for (const std::string &Entry : PrintfMetadata) {
    // "<id>:<num-args>:<arg-size>...:<format-string>". Only the argument
    // count matters, to know how many size fields to step over; the format
    // string is whatever remains, colons and all.
    // Step over the format-string id, which nothing here keys on.
    llvm::StringRef Rest = llvm::StringRef(Entry).split(':').second;
    if (Rest.empty())
      continue;
    llvm::StringRef NumArgsField;
    std::tie(NumArgsField, Rest) = Rest.split(':');

    unsigned NumArgs = 0;
    if (NumArgsField.getAsInteger(10, NumArgs))
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "amdhsa.printf entry '{0}' does not begin with an argument count",
          Entry));
    for (unsigned I = 0; I < NumArgs; ++I) {
      if (Rest.empty())
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "amdhsa.printf entry '{0}' declares {1} argument(s) but carries "
            "fewer argument sizes",
            Entry, NumArgs));
      Rest = Rest.split(':').second;
    }

    // Buffered printf replaces a constant format string with its hash, and
    // records the pairing here as "<hash>,<string>". Anything else is an
    // OpenCL-style entry with no hash to key on.
    const size_t Comma = Rest.find(',');
    if (Comma == llvm::StringRef::npos)
      continue;
    uint64_t Hash = 0;
    if (Rest.substr(0, Comma).getAsInteger(16, Hash))
      continue;

    std::string Format = unescapeMetadataString(Rest.substr(Comma + 1));
    auto [It, Inserted] = Out.try_emplace(Hash, Format);
    if (!Inserted && It->second != Format)
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "amdhsa.printf carries two different format strings for hash {0:x}; "
          "the printf buffer cannot be decoded unambiguously",
          Hash));
  }
  return Out;
}

//===----------------------------------------------------------------------===//
// Buffered printf
//===----------------------------------------------------------------------===//

llvm::Error initializePrintfBuffer(llvm::MutableArrayRef<uint8_t> Buffer) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Buffer.size() >= PrintfBufferHeaderSize,
      llvm::formatv("A printf buffer must be at least {0} bytes to hold its "
                    "control words; got {1}",
                    PrintfBufferHeaderSize, Buffer.size())));
  std::memset(Buffer.data(), 0, Buffer.size());
  // Word 0 is the running write offset, which starts at zero. Word 1 is how
  // many bytes the device may bump-allocate, i.e. everything past the header.
  const auto Available =
      static_cast<uint32_t>(Buffer.size() - PrintfBufferHeaderSize);
  llvm::support::endian::write32le(Buffer.data() + sizeof(uint32_t), Available);
  return llvm::Error::success();
}

llvm::Error drainPrintfBuffer(llvm::ArrayRef<uint8_t> Buffer,
                              const PrintfFormatStringMap &FormatStrings) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Buffer.size() >= PrintfBufferHeaderSize,
      llvm::formatv("A printf buffer must be at least {0} bytes to hold its "
                    "control words; got {1}",
                    PrintfBufferHeaderSize, Buffer.size())));

  const uint32_t Written = llvm::support::endian::read32le(Buffer.data());
  if (Written == 0)
    return llvm::Error::success();

  llvm::ArrayRef<uint8_t> Records = Buffer.drop_front(PrintfBufferHeaderSize);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Written <= Records.size(),
      llvm::formatv("The device reported writing {0} bytes into a printf "
                    "buffer with room for {1}",
                    Written, Records.size())));
  Records = Records.take_front(Written);

  while (!Records.empty()) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        Records.size() >= sizeof(uint32_t),
        "A printf buffer record is too short to hold its control word"));
    const uint32_t Control = llvm::support::endian::read32le(Records.data());
    // Bit 0 selects the stream, bit 1 flags a constant (hashed) format
    // string, and the rest is the total record length in bytes. The control
    // word is not padded, so everything after it — the hash or the
    // word-padded format string, then the arguments — is a whole number of
    // 64-bit words offset four bytes into the record.
    const uint32_t RecordSize = Control >> 2;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        RecordSize > sizeof(uint32_t) && RecordSize <= Records.size() &&
            (RecordSize - sizeof(uint32_t)) % sizeof(uint64_t) == 0,
        llvm::formatv("A printf buffer record declares a length of {0} bytes, "
                      "which is not a well-formed record inside the {1} bytes "
                      "remaining",
                      RecordSize, Records.size())));

    llvm::ArrayRef<uint8_t> Body =
        Records.slice(sizeof(uint32_t), RecordSize - sizeof(uint32_t));

    // Reassemble the record into the layout formatDevicePrintfMessage wants:
    // the format string, padded to a word, followed by the arguments.
    std::vector<uint64_t> Message;
    if ((Control & 2U) != 0) {
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          Body.size() >= sizeof(uint64_t),
          "A printf buffer record flags a constant format string but is too "
          "short to hold its hash"));
      const uint64_t Hash = llvm::support::endian::read64le(Body.data());
      llvm::ArrayRef<uint8_t> Args = Body.drop_front(sizeof(uint64_t));

      auto FmtIt = FormatStrings.find(Hash);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          FmtIt != FormatStrings.end(),
          llvm::formatv("A printf buffer record references format string hash "
                        "{0:x}, which the code object's amdhsa.printf metadata "
                        "does not define",
                        Hash)));
      const std::string &Fmt = FmtIt->second;

      const size_t FmtWords =
          (Fmt.size() + 1 + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      Message.assign(FmtWords + Args.size() / sizeof(uint64_t), 0);
      std::memcpy(Message.data(), Fmt.data(), Fmt.size());
      std::memcpy(Message.data() + FmtWords, Args.data(), Args.size());
    } else {
      // The record already carries the format string inline.
      Message.assign(Body.size() / sizeof(uint64_t), 0);
      std::memcpy(Message.data(), Body.data(), Message.size() * sizeof(uint64_t));
    }

    std::FILE *Stream = (Control & 1U) ? stderr : stdout;
    formatDevicePrintfMessage(Stream, Message);

    Records = Records.drop_front(RecordSize);
  }
  return llvm::Error::success();
}

} // namespace luthier
