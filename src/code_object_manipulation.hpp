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
class ElfViewImpl : public std::enable_shared_from_this<ElfViewImpl> {
 public:
    ElfViewImpl() = delete;

    static std::shared_ptr<ElfViewImpl> make_view(code_view_t elf) {
        return std::shared_ptr<ElfViewImpl>(new ElfViewImpl(elf));
    }

    ELFIO::elfio &getElfIo() const {
        if (io_ == std::nullopt) {
            io_ = ELFIO::elfio();
            // All elfio objects are loaded with lazy=true in ElfViewImpl to prevent additional memory copy
            io_->load(*dataStringStream_, true);
        }
        return io_.value();
    }

    code_view_t getView() const {
        return data_;
    }

 private:
    explicit ElfViewImpl(code_view_t elf) : data_(elf),
                                            // Convert the code_view_t to a string_view first, and then take its iterators to construct the dataStringStream_
                                            dataStringStream_(std::make_unique<boost_ios::stream<boost_ios::basic_array_source<char>>>(
                                                std::string_view(reinterpret_cast<const char *>(data_.data()), data_.size()).begin(),
                                                std::string_view(reinterpret_cast<const char *>(data_.data()), data_.size()).end())) {}

    mutable std::optional<ELFIO::elfio> io_{std::nullopt};
    const code_view_t data_;
    const std::unique_ptr<boost_ios::stream<boost_ios::basic_array_source<char>>> dataStringStream_;//! Used to construct the elfio object;
                                                                                                    //! Without keeping a reference to this stream,
                                                                                                    //! we cannot use the elfio in lazy mode
};

typedef std::shared_ptr<ElfViewImpl> ElfView;

/**
 * Factory method to construct an ElfView
 * To ensure correct passing of arguments of ElfView between different functions and scope management,
 * only shared pointers of ElfViewImpls are allowed to be constructed
 * \return an ElfView object, which is a std::shared_ptr of an ElfViewImpl
 */
ElfView makeElfView(code_view_t elf) {
    return ElfViewImpl::make_view(elf);
}

class SymbolView {
 private:
    const ElfView elf_;//!   section's parent elfio class
    const ELFIO::section *section_;         //!   symbol's section
    std::string name_;                      //!   symbol name
    code_view_t data_;                      //!   symbol's raw data
    size_t value_;                          //!   value of the symbol
    unsigned char type_;                    //!   type of the symbol
 public:
    SymbolView() = delete;

    SymbolView(const ElfView &elf, unsigned int symIndex);

    [[nodiscard]] ElfView getElfview() const {
        return elf_;
    };

    [[nodiscard]] const ELFIO::section *getSection() const {
        return section_;
    };

    [[nodiscard]] const std::string &getName() const {
        return name_;
    };
    [[nodiscard]] code_view_t getView() const {
        return data_;
    }

    [[nodiscard]] size_t getValue() const {
        return value_;
    };

    [[nodiscard]] unsigned char getType() const {
        return type_;
    };

    [[nodiscard]] const std::byte *getData() const {
        return reinterpret_cast<const std::byte *>(section_->get_data() + (size_t) value_ - (size_t) section_->get_offset());
    }
};

/**
 * Returns the number of symbols in the ElfView
 * \param elfView a view of the AMDGPU ELF in host memory
 * \return number of symbols in the ELF
 */
unsigned int getSymbolNum(const ElfView &elfView);

/**
 * Returns the demangled name of the input symbol name
 * @param mangledName mangled name string
 * @return demangled name as std::string
 */
std::string getDemangledName(const std::string &mangledName);

amd_comgr_status_t getCodeObjectElfsFromFatBinary(const void *data, std::vector<ELFIO::elfio> &fatBinaryElfs);

code_view_t getFunctionFromSymbol(ELFIO::elfio &elfio, const std::string &functionName);

std::vector<code_view_t> getDeviceLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

std::vector<code_view_t> getHostLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

void printRSR1(const kernel_descriptor_t *kd);

void printRSR2(const kernel_descriptor_t *kd);

void printCodeProperties(const kernel_descriptor_t *kd);

ELFIO::elfio createAuxilaryInstrumentationElf();

}// namespace luthier::co_manip

#endif
