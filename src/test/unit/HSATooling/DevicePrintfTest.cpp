//===-- DevicePrintfTest.cpp ----------------------------------------------===//
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
///
/// \file
/// Tests the two device \c printf paths against hand-built messages: the
/// message encoding both paths share, and the buffered-\c printf record log a
/// kernel writes into its \c hidden_printf_buffer. Neither needs a GPU — the
/// encodings are fixed by the AMDGPU ABI, so a test can produce them directly.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/DevicePrintf.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <llvm/Support/Endian.h>
#include <llvm/Support/Error.h>
#include <string>
#include <vector>

using namespace luthier;

namespace {

//===----------------------------------------------------------------------===//
// Message construction
//===----------------------------------------------------------------------===//

/// Encodes a \c double into the single word a floating-point argument
/// occupies.
uint64_t asDoubleWord(double D) {
  uint64_t Word;
  std::memcpy(&Word, &D, sizeof(Word));
  return Word;
}

/// Packs \p Format (NUL-terminated, padded to a word) followed by \p Args into
/// a message body.
std::vector<uint64_t> makeMessage(llvm::StringRef Format,
                                  llvm::ArrayRef<uint64_t> Args) {
  const size_t FormatWords = (Format.size() + 1 + 7) / 8;
  std::vector<uint64_t> Message(FormatWords, 0);
  std::memcpy(Message.data(), Format.data(), Format.size());
  Message.insert(Message.end(), Args.begin(), Args.end());
  return Message;
}

/// Appends a string argument, which travels inline in the message rather than
/// as a pointer into device memory.
void appendInlineString(std::vector<uint64_t> &Message, llvm::StringRef Str) {
  const size_t Words = (Str.size() + 1 + 7) / 8;
  const size_t Start = Message.size();
  Message.resize(Start + Words, 0);
  std::memcpy(Message.data() + Start, Str.data(), Str.size());
}

std::string render(llvm::ArrayRef<uint64_t> Message) {
  testing::internal::CaptureStdout();
  formatDevicePrintfMessage(stdout, Message);
  std::fflush(stdout);
  return testing::internal::GetCapturedStdout();
}

//===----------------------------------------------------------------------===//
// Format rendering
//===----------------------------------------------------------------------===//

TEST(DevicePrintfFormat, RendersLiteralTextWithNoArguments) {
  EXPECT_EQ(render(makeMessage("hello, device\n", {})), "hello, device\n");
}

TEST(DevicePrintfFormat, RendersIntegerConversions) {
  EXPECT_EQ(render(makeMessage("%d", {uint64_t{42}})), "42");
  EXPECT_EQ(render(makeMessage("%u", {uint64_t{7}})), "7");
  EXPECT_EQ(render(makeMessage("%x", {uint64_t{255}})), "ff");
  EXPECT_EQ(render(makeMessage("%X", {uint64_t{255}})), "FF");
  EXPECT_EQ(render(makeMessage("%o", {uint64_t{8}})), "10");
  EXPECT_EQ(render(makeMessage("%c", {uint64_t{'Z'}})), "Z");
}

TEST(DevicePrintfFormat, RendersFloatingPointConversions) {
  EXPECT_EQ(render(makeMessage("%.2f", {asDoubleWord(1.5)})), "1.50");
  EXPECT_EQ(render(makeMessage("%.1e", {asDoubleWord(1234.0)})), "1.2e+03");
}

TEST(DevicePrintfFormat, RendersInlineStringArguments) {
  std::vector<uint64_t> Message = makeMessage("[%s]", {});
  appendInlineString(Message, "inline");
  EXPECT_EQ(render(Message), "[inline]");
}

// The string occupies as many words as it needs once padded, so an argument
// after it has to be found past that padding rather than one word on.
TEST(DevicePrintfFormat, ResumesAfterAPaddedStringArgument) {
  std::vector<uint64_t> Message = makeMessage("%s=%d", {});
  appendInlineString(Message, "a-fairly-long-key");
  Message.push_back(9);
  EXPECT_EQ(render(Message), "a-fairly-long-key=9");
}

TEST(DevicePrintfFormat, RendersEscapedPercent) {
  EXPECT_EQ(render(makeMessage("100%% done\n", {})), "100% done\n");
}

TEST(DevicePrintfFormat, MixesLiteralsAndConversions) {
  EXPECT_EQ(render(makeMessage("a=%d b=%d done", {uint64_t{1}, uint64_t{2}})),
            "a=1 b=2 done");
}

TEST(DevicePrintfFormat, HonoursDynamicWidthAndPrecision) {
  EXPECT_EQ(render(makeMessage("%*d", {uint64_t{5}, uint64_t{42}})), "   42");
  EXPECT_EQ(
      render(makeMessage("%.*f", {uint64_t{3}, asDoubleWord(2.0)})),
      "2.000");
}

// %n would let a device-supplied format string write through a pointer, so it
// is never handed to the C library; it renders nothing and eats its argument.
TEST(DevicePrintfFormat, SkipsPercentNAndStillConsumesItsArgument) {
  EXPECT_EQ(render(makeMessage("[%n%d]", {uint64_t{0}, uint64_t{5}})), "[5]");
}

TEST(DevicePrintfFormat, ReportsTheCharacterCount) {
  EXPECT_EQ(formatDevicePrintfMessage(stdout, {}), 0);

  testing::internal::CaptureStdout();
  const int Count = formatDevicePrintfMessage(stdout, makeMessage("abcde", {}));
  std::fflush(stdout);
  (void)testing::internal::GetCapturedStdout();
  EXPECT_EQ(Count, 5);
}

// A message whose format string has no terminator inside it would otherwise
// walk off the end of the buffer.
TEST(DevicePrintfFormat, RefusesAnUnterminatedFormatString) {
  std::vector<uint64_t> Message(2, ~uint64_t{0});
  EXPECT_EQ(render(Message), "");
}

TEST(DevicePrintfFormat, StopsWhenTheArgumentsRunOut) {
  // Two conversions, one argument: the second cannot be rendered.
  EXPECT_EQ(render(makeMessage("%d,%d", {uint64_t{1}})), "1,");
}

//===----------------------------------------------------------------------===//
// The hostcall control word
//===----------------------------------------------------------------------===//

TEST(DevicePrintfHostcall, ControlWordZeroGoesToStdout) {
  std::vector<uint64_t> Message{0};
  const std::vector<uint64_t> Body = makeMessage("out", {});
  Message.insert(Message.end(), Body.begin(), Body.end());

  uint64_t Output = 0;
  testing::internal::CaptureStdout();
  handleDevicePrintfHostcall(&Output, Message);
  std::fflush(stdout);
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "out");
  EXPECT_EQ(Output, 3u);
}

TEST(DevicePrintfHostcall, ControlWordBitZeroSelectsStderr) {
  std::vector<uint64_t> Message{1};
  const std::vector<uint64_t> Body = makeMessage("err", {});
  Message.insert(Message.end(), Body.begin(), Body.end());

  uint64_t Output = 0;
  testing::internal::CaptureStderr();
  handleDevicePrintfHostcall(&Output, Message);
  std::fflush(stderr);
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "err");
  EXPECT_EQ(Output, 3u);
}

TEST(DevicePrintfHostcall, RejectsReservedControlBits) {
  std::vector<uint64_t> Message{0b10};
  const std::vector<uint64_t> Body = makeMessage("nope", {});
  Message.insert(Message.end(), Body.begin(), Body.end());

  uint64_t Output = 0;
  handleDevicePrintfHostcall(&Output, Message);
  EXPECT_EQ(Output, static_cast<uint64_t>(-1));
}

TEST(DevicePrintfHostcall, RejectsAnEmptyMessage) {
  uint64_t Output = 0;
  handleDevicePrintfHostcall(&Output, {});
  EXPECT_EQ(Output, static_cast<uint64_t>(-1));
}

//===----------------------------------------------------------------------===//
// amdhsa.printf metadata
//===----------------------------------------------------------------------===//

TEST(PrintfMetadata, ParsesHashedConstantFormatStrings) {
  auto MapOrErr = parsePrintfFormatStrings(
      {"0:0:1f2e3d4c5b6a7988,hello %d\n", "0:0:00000000000000ff,bye\n"});
  ASSERT_TRUE(static_cast<bool>(MapOrErr))
      << llvm::toString(MapOrErr.takeError());

  ASSERT_EQ(MapOrErr->count(0x1f2e3d4c5b6a7988ULL), 1u);
  EXPECT_EQ(MapOrErr->lookup(0x1f2e3d4c5b6a7988ULL), "hello %d\n");
  EXPECT_EQ(MapOrErr->lookup(0xffULL), "bye\n");
}

// A colon inside a format string is escaped in the metadata so it cannot be
// mistaken for a field separator.
TEST(PrintfMetadata, UnescapesColonsInsideFormatStrings) {
  auto MapOrErr = parsePrintfFormatStrings({"0:0:10,a\\72b\n"});
  ASSERT_TRUE(static_cast<bool>(MapOrErr))
      << llvm::toString(MapOrErr.takeError());
  EXPECT_EQ(MapOrErr->lookup(0x10ULL), "a:b\n");
}

// OpenCL-style entries carry argument sizes and no hash; the buffered-printf
// decoder has no use for them, so they are skipped rather than rejected.
TEST(PrintfMetadata, SkipsEntriesWithoutAHash) {
  auto MapOrErr = parsePrintfFormatStrings({"1:2:4:4:%d %d\n"});
  ASSERT_TRUE(static_cast<bool>(MapOrErr))
      << llvm::toString(MapOrErr.takeError());
  EXPECT_TRUE(MapOrErr->empty());
}

TEST(PrintfMetadata, RejectsAnEntryWhoseArgumentCountIsUnreadable) {
  auto MapOrErr = parsePrintfFormatStrings({"0:not-a-number:ff,x"});
  EXPECT_FALSE(static_cast<bool>(MapOrErr));
  llvm::consumeError(MapOrErr.takeError());
}

TEST(PrintfMetadata, RejectsTwoFormatStringsForTheSameHash) {
  auto MapOrErr =
      parsePrintfFormatStrings({"0:0:ab,first\n", "0:0:ab,second\n"});
  EXPECT_FALSE(static_cast<bool>(MapOrErr));
  llvm::consumeError(MapOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// The buffered printf record log
//===----------------------------------------------------------------------===//

/// Builds one record of the log a buffered-printf kernel bump-allocates.
/// \p Body is everything after the control word: either a format-string hash
/// followed by arguments, or the format string inline followed by arguments.
std::vector<uint8_t> makeRecord(llvm::ArrayRef<uint64_t> Body, bool Hashed,
                                bool ToStderr) {
  const uint32_t RecordSize =
      static_cast<uint32_t>(sizeof(uint32_t) + Body.size() * sizeof(uint64_t));
  const uint32_t Control =
      (RecordSize << 2) | (Hashed ? 0b10u : 0u) | (ToStderr ? 1u : 0u);

  std::vector<uint8_t> Record(RecordSize, 0);
  llvm::support::endian::write32le(Record.data(), Control);
  std::memcpy(Record.data() + sizeof(uint32_t), Body.data(),
              Body.size() * sizeof(uint64_t));
  return Record;
}

/// Lays out a printf buffer holding \p Records, prepared the way the loader
/// hands one to a kernel and then filled in as the kernel would.
std::vector<uint8_t> makePrintfBuffer(llvm::ArrayRef<std::vector<uint8_t>> Records,
                                      size_t Capacity = 4096) {
  std::vector<uint8_t> Buffer(Capacity, 0);
  llvm::Error Err = initializePrintfBuffer(Buffer);
  EXPECT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));

  size_t Offset = PrintfBufferHeaderSize;
  for (const std::vector<uint8_t> &Record : Records) {
    std::memcpy(Buffer.data() + Offset, Record.data(), Record.size());
    Offset += Record.size();
  }
  // Word 0 is the running write offset the device bumps as it appends.
  llvm::support::endian::write32le(
      Buffer.data(),
      static_cast<uint32_t>(Offset - PrintfBufferHeaderSize));
  return Buffer;
}

TEST(PrintfBuffer, InitializesTheControlWords) {
  std::vector<uint8_t> Buffer(4096, 0xAB);
  ASSERT_FALSE(static_cast<bool>(initializePrintfBuffer(Buffer)));
  EXPECT_EQ(llvm::support::endian::read32le(Buffer.data()), 0u)
      << "nothing has been written yet";
  EXPECT_EQ(llvm::support::endian::read32le(Buffer.data() + 4),
            4096u - PrintfBufferHeaderSize)
      << "the device may fill everything past the header";
  EXPECT_EQ(Buffer[PrintfBufferHeaderSize], 0u) << "the log starts cleared";
}

TEST(PrintfBuffer, RejectsABufferTooSmallForItsHeader) {
  std::vector<uint8_t> Tiny(4, 0);
  llvm::Error Err = initializePrintfBuffer(Tiny);
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

TEST(PrintfBuffer, AnUntouchedBufferDrainsToNothing) {
  std::vector<uint8_t> Buffer(4096, 0);
  ASSERT_FALSE(static_cast<bool>(initializePrintfBuffer(Buffer)));

  testing::internal::CaptureStdout();
  llvm::Error Err = drainPrintfBuffer(Buffer, {});
  std::fflush(stdout);
  EXPECT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "");
}

TEST(PrintfBuffer, DrainsARecordCarryingItsFormatStringInline) {
  const std::vector<uint64_t> Body = makeMessage("inline %d\n", {uint64_t{7}});
  const std::vector<uint8_t> Buffer =
      makePrintfBuffer({makeRecord(Body, /*Hashed=*/false, /*ToStderr=*/false)});

  testing::internal::CaptureStdout();
  llvm::Error Err = drainPrintfBuffer(Buffer, {});
  std::fflush(stdout);
  EXPECT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "inline 7\n");
}

TEST(PrintfBuffer, DrainsARecordReferencingAFormatStringByHash) {
  PrintfFormatStringMap Formats;
  Formats.try_emplace(0xCAFEULL, "hashed %d\n");

  const std::vector<uint64_t> Body{0xCAFEULL, 11};
  const std::vector<uint8_t> Buffer =
      makePrintfBuffer({makeRecord(Body, /*Hashed=*/true, /*ToStderr=*/false)});

  testing::internal::CaptureStdout();
  llvm::Error Err = drainPrintfBuffer(Buffer, Formats);
  std::fflush(stdout);
  EXPECT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "hashed 11\n");
}

TEST(PrintfBuffer, DrainsSeveralRecordsInOrder) {
  const std::vector<uint8_t> Buffer = makePrintfBuffer(
      {makeRecord(makeMessage("one\n", {}), false, false),
       makeRecord(makeMessage("two %d\n", {uint64_t{2}}), false, false),
       makeRecord(makeMessage("three\n", {}), false, false)});

  testing::internal::CaptureStdout();
  llvm::Error Err = drainPrintfBuffer(Buffer, {});
  std::fflush(stdout);
  EXPECT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "one\ntwo 2\nthree\n");
}

TEST(PrintfBuffer, RoutesRecordsToStderrWhenTheControlWordSaysSo) {
  const std::vector<uint8_t> Buffer = makePrintfBuffer(
      {makeRecord(makeMessage("to stderr\n", {}), false, /*ToStderr=*/true)});

  testing::internal::CaptureStderr();
  llvm::Error Err = drainPrintfBuffer(Buffer, {});
  std::fflush(stderr);
  EXPECT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(testing::internal::GetCapturedStderr(), "to stderr\n");
}

TEST(PrintfBuffer, RejectsAHashTheMetadataDoesNotDefine) {
  const std::vector<uint64_t> Body{0xDEADULL, 1};
  const std::vector<uint8_t> Buffer =
      makePrintfBuffer({makeRecord(Body, /*Hashed=*/true, false)});

  llvm::Error Err = drainPrintfBuffer(Buffer, {});
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

// A record whose declared length does not fit what remains would otherwise
// walk the decoder off the end of the buffer.
TEST(PrintfBuffer, RejectsARecordLongerThanTheBytesWritten) {
  std::vector<uint8_t> Buffer(4096, 0);
  ASSERT_FALSE(static_cast<bool>(initializePrintfBuffer(Buffer)));
  // Claim a 64-byte record but only account for 16 bytes of output.
  llvm::support::endian::write32le(Buffer.data() + PrintfBufferHeaderSize,
                                   (64u << 2));
  llvm::support::endian::write32le(Buffer.data(), 16u);

  llvm::Error Err = drainPrintfBuffer(Buffer, {});
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

TEST(PrintfBuffer, RejectsAWriteOffsetPastTheBuffer) {
  std::vector<uint8_t> Buffer(256, 0);
  ASSERT_FALSE(static_cast<bool>(initializePrintfBuffer(Buffer)));
  llvm::support::endian::write32le(Buffer.data(), 1u << 20);

  llvm::Error Err = drainPrintfBuffer(Buffer, {});
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

} // namespace
