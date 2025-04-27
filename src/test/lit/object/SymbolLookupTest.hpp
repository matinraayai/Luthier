#ifndef LUTHIER_TEST_LIT_OBJECT_SYMBOL_LOOKUP_TEST_HPP
#define LUTHIER_TEST_LIT_OBJECT_SYMBOL_LOOKUP_TEST_HPP
#include <llvm/ADT/StringMap.h>
#include <luthier/object/ELFObjectUtils.h>
#include <random>

static constexpr char Letters[] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
    'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
static constexpr std::size_t LettersSize = sizeof(Letters);

inline std::default_random_engine &getRandomEngine() {
  static std::default_random_engine RandEngine(std::random_device{}());
  return RandEngine;
}

inline char getRandomChar() {
  std::uniform_int_distribution<> LettersDist(0, LettersSize - 1);
  return Letters[LettersDist(getRandomEngine())];
}

template <class IntT> IntT getRandomInteger(IntT Min, IntT Max) {
  std::uniform_int_distribution<unsigned long long> dist(Min, Max);
  return static_cast<IntT>(dist(getRandomEngine()));
}

inline std::string getRandomString(std::size_t Len) {
  std::string str(Len, 0);
  std::generate_n(str.begin(), Len, &getRandomChar);
  return str;
}

inline bool
hasSymbolLookupHashTable(const llvm::object::ELFObjectFileBase &ObjFile) {
  return llvm::any_of(ObjFile.sections(),
                      [](const llvm::object::ELFSectionRef &Sec) {
                        return Sec.getType() == llvm::ELF::SHT_HASH &&
                               Sec.getType() == llvm::ELF::SHT_GNU_HASH;
                      });
}

template <typename ELFT>
static llvm::Error
symbolLookupTest(const llvm::object::ELFObjectFile<ELFT> &ElfObjFile,
                 llvm::raw_ostream &OS) {
  // Check if the ELF has a hash section to begin with; If not, skip tests
  // that relies on it being present
  // Mapping between dynamic symbol names and their symbol refs inside the
  // dynamic symbol table; We will use this later for testing if symbol lookup
  // fails for symbols not present in the ELF
  if (hasSymbolLookupHashTable(ElfObjFile)) {
    llvm::StringMap<llvm::object::ELFSymbolRef> NameToDynSymMap;
    for (const auto &DynSym :
         llvm::make_range(ElfObjFile.dynamic_symbol_begin(),
                          ElfObjFile.dynamic_symbol_end())) {
      llvm::Expected<llvm::StringRef> SymNameOrErr = DynSym.getName();
      LUTHIER_RETURN_ON_ERROR(SymNameOrErr.takeError());
      NameToDynSymMap.insert({*SymNameOrErr, DynSym});
      // Test if we can find this symbol using its name
      llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
          LookedUpSymbolUsingNameOrErr = luthier::object::lookupSymbolByName(
              ElfObjFile, *SymNameOrErr, false);
      LUTHIER_RETURN_ON_ERROR(LookedUpSymbolUsingNameOrErr.takeError());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          LookedUpSymbolUsingNameOrErr->has_value(),
          "Could not find dynamic symbol {0} by its name.", *SymNameOrErr));
      // The symbol refs should be equal
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_ERROR_CHECK((**LookedUpSymbolUsingNameOrErr) == DynSym,
                              "Looked up symbol with name {0} does not match"
                              "the original symbol found using iteration",
                              *SymNameOrErr));
    }
    // Iterate over all the symbols of the ELF object; If they are not
    // inside the dynsym table, then symbol lookup using the hash table should
    // fail
    for (const auto &Sym : ElfObjFile.symbols()) {
      llvm::Expected<llvm::StringRef> SymNameOrErr = Sym.getName();
      LUTHIER_REPORT_FATAL_ON_ERROR(SymNameOrErr.takeError());
      if (!NameToDynSymMap.contains(*SymNameOrErr)) {
        // Test if we can find this symbol using its name
        llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
            LookedUpSymbolUsingNameOrErr = luthier::object::lookupSymbolByName(
                ElfObjFile, *SymNameOrErr, false);
        LUTHIER_REPORT_FATAL_ON_ERROR(LookedUpSymbolUsingNameOrErr.takeError());
        LUTHIER_REPORT_FATAL_ON_ERROR(
            LUTHIER_ERROR_CHECK(!LookedUpSymbolUsingNameOrErr->has_value(),
                                "Found the non-dynamic {0} symbol using hash "
                                "lookup using its name.",
                                *SymNameOrErr));
      }
    }
  }

  llvm::StringMap<llvm::object::ELFSymbolRef> NameToSymMap;
  // Iterate over all the symbols of the ELF object, and check if we can find
  // them
  for (const auto &Sym : ElfObjFile.symbols()) {
    llvm::Expected<llvm::StringRef> SymNameOrErr = Sym.getName();
    LUTHIER_REPORT_FATAL_ON_ERROR(SymNameOrErr.takeError());
    NameToSymMap.insert({*SymNameOrErr, Sym});
    // Test if we can find this symbol using its name
    llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
        LookedUpSymbolUsingNameOrErr =
            luthier::object::lookupSymbolByName(ElfObjFile, *SymNameOrErr);
    LUTHIER_REPORT_FATAL_ON_ERROR(LookedUpSymbolUsingNameOrErr.takeError());
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        LookedUpSymbolUsingNameOrErr->has_value(),
        "Could not find symbol {0} by its name.", *SymNameOrErr));
    // The symbol refs might not be equal, but the contents of the symbols
    // should be the same
    llvm::Expected<bool> SymsAreEqualOrErr = luthier::object::areSymbolsEqual(
        ElfObjFile, (*LookedUpSymbolUsingNameOrErr)->getRawDataRefImpl(),
        Sym.getRawDataRefImpl());
    LUTHIER_RETURN_ON_ERROR(SymsAreEqualOrErr.takeError());
    LUTHIER_REPORT_FATAL_ON_ERROR(
        LUTHIER_ERROR_CHECK(*SymsAreEqualOrErr,
                            "Looked up symbol with name {0} does not match "
                            "the original symbol found using iteration",
                            *SymNameOrErr));
  }

  // Generate a random symbol name that's not inside the ELF, and look it up
  // inside the ELF to make sure we can't find it
  std::string RandomSymbolName;
  do {
    RandomSymbolName = getRandomString(10);
  } while (NameToSymMap.contains(RandomSymbolName));
  llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
      RandomSymbolLookupRes =
          luthier::object::lookupSymbolByName(ElfObjFile, RandomSymbolName);
  LUTHIER_REPORT_FATAL_ON_ERROR(RandomSymbolLookupRes.takeError());
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !RandomSymbolLookupRes->has_value(),
      "Found a symbol associated with a randomly generated string."));
  OS << "Passed symbol name lookup test.\n";
  return llvm::Error::success();
}

static llvm::Error symbolLookupTest(const llvm::object::ObjectFile &ObjFile,
                                    llvm::raw_ostream &OS) {
  if (const auto *ELF64LE =
          llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(&ObjFile))
    return symbolLookupTest(*ELF64LE, OS);
  else if (const auto *ELF64BE =
               llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(&ObjFile))
    return symbolLookupTest(*ELF64BE, OS);
  else if (const auto *ELF32LE =
               llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(&ObjFile))
    return symbolLookupTest(*ELF32LE, OS);
  else if (const auto *ELF32BE =
               llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(&ObjFile)) {
    return symbolLookupTest(*ELF32BE, OS);
  } else {
    OS << "Skipped symbol name lookup test: Object file is not an ELF.\n";
    return llvm::Error::success();
  }
}

#endif