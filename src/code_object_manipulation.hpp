/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef AMDGPU_CODE_OBJECT_MANIPULATION
#define AMDGPU_CODE_OBJECT_MANIPULATION

#include "luthier_types.hpp"
#include <amd_comgr/amd_comgr.h>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <elfio/elfio.hpp>
#include <map>
#include <optional>
#include <utility>

namespace boost_ios = boost::iostreams;

namespace luthier::co_manip {

/**
 * \brief a non-owning view of memory portion that contains device code
 * Can be passed by value or reference and can be returned. Is trivially copyable.
 */
typedef std::basic_string_view<std::byte> code_view_t;

/**
 * \brief owns the memory portion that contains device code
 * Can be passed by reference only. Cannot be returned. Is not trivially copyable.
 */
typedef std::basic_string<std::byte> code_t;

/**
 * \briefs a non-owning read-only view of an AMDGPU ELF Code object located on the host
 */
class ElfView {
 public:
    ElfView() = delete;
    explicit ElfView(code_view_t elf) : data_(elf),
                                        // Convert the code_view_t to a string_view first, and then take its iterators to construct the dataStringStream_
                                        dataStringStream_(std::make_unique<boost_ios::stream<boost_ios::basic_array_source<char>>>(
                                            std::string_view(reinterpret_cast<const char *>(data_.data()), data_.size()).begin(),
                                            std::string_view(reinterpret_cast<const char *>(data_.data()), data_.size()).end())) {}

    ELFIO::elfio &get_elfio() const {
        if (io_ == std::nullopt) {
            io_ = ELFIO::elfio();
            io_->load(*dataStringStream_, false);
        }
        return io_.value();
    }

    code_view_t get_data() const {
        return data_;
    }

 private:
    mutable std::optional<ELFIO::elfio> io_{std::nullopt};
    const code_view_t data_;
    const std::unique_ptr<boost_ios::stream<boost_ios::basic_array_source<char>>> dataStringStream_;//! Used to construct the elfio object;
                                                                                                    //! Without keeping a reference to this stream,
                                                                                                    //! we cannot use the elfio in lazy mode
};

class SymbolInfo {
 private:
    const ELFIO::elfio *elfio;    //!   section's parent elfio class
    const ELFIO::section *section;//!   symbol's section
    std::string name;             //!   symbol name
    luthier_address_t address;    //!   symbol's offset from the beginning of the ELF
    uint64_t size;                //!   size of data corresponding to symbol
    size_t value;                 //!   value of the symbol
    unsigned char type;           //!   type of the symbol
 public:
    SymbolInfo() = delete;

    SymbolInfo(const ELFIO::elfio *symELFIo, unsigned int symIndex);

    [[nodiscard]] const ELFIO::elfio *get_elfio() const;
    [[nodiscard]] const ELFIO::section *get_section() const;
    [[nodiscard]] const std::string &get_name() const;
    [[nodiscard]] luthier_address_t get_address() const;
    [[nodiscard]] uint64_t get_size() const;
    [[nodiscard]] size_t get_value() const;
    [[nodiscard]] unsigned char get_type() const;

    [[nodiscard]] const char *get_data() const {
        return section->get_data() + (size_t) value - (size_t) section->get_offset();
    }
};

/**
 * Returns the number of symbols
 * @param io
 * @return
 */
unsigned int getSymbolNum(const ELFIO::elfio &io);

std::string getDemangledName(const char *mangledName);

amd_comgr_status_t getCodeObjectElfsFromFatBinary(const void *data, std::vector<ELFIO::elfio> &fatBinaryElfs);

code_view_t getFunctionFromSymbol(ELFIO::elfio &elfio, const std::string &functionName);

std::vector<code_view_t> getDeviceLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

std::vector<code_view_t> getHostLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

//std::shared_ptr<ELFIO::elfio> elfioFromMemory(const co_manip::code_view_t &elf, bool lazy = true);

void printRSR1(const kernel_descriptor_t *kd);

void printRSR2(const kernel_descriptor_t *kd);

void printCodeProperties(const kernel_descriptor_t *kd);

ELFIO::elfio createAuxilaryInstrumentationElf();

}// namespace luthier::co_manip

#endif
