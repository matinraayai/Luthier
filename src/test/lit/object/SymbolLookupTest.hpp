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

template <typename ELFT>
static void
symbolLookupTest(const llvm::object::ELFObjectFile<ELFT> &ElfObjFile,
                 llvm::raw_ostream &OS) {
  // A mapping between symbol names inside ElfObjFile and their symbol ref;
  // We will use this later for testing if symbol lookup fails for symbols
  // not present in the ELF
  llvm::StringMap<llvm::object::ELFSymbolRef> NameToSymbolMap;
  // Iterate over the symbols of the ELF object, and check if we can find them
  for (const auto &Sym : ElfObjFile.symbols()) {
    llvm::Expected<llvm::StringRef> SymNameOrErr = Sym.getName();
    LUTHIER_REPORT_FATAL_ON_ERROR(SymNameOrErr.takeError());
    NameToSymbolMap.insert({*SymNameOrErr, Sym});
    // Test if we can find this symbol using its name
    llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
        LookedUpSymbolUsingNameOrErr =
            luthier::object::lookupSymbolByName(ElfObjFile, *SymNameOrErr);
    LUTHIER_REPORT_FATAL_ON_ERROR(LookedUpSymbolUsingNameOrErr.takeError());
    if (LookedUpSymbolUsingNameOrErr->has_value()) {
      OS << "Found the symbol.\n";
    }
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        LookedUpSymbolUsingNameOrErr->has_value(),
        "Could not find symbol {0} by its name.", *SymNameOrErr));
    if (**LookedUpSymbolUsingNameOrErr != Sym) {
      OS << "Does not match!\n";
      OS << "Original: " << *SymNameOrErr << "\n";
      OS << "Found: "
         << llvm::cantFail((*LookedUpSymbolUsingNameOrErr)->getName()) << "\n";
      OS << "Does comparing the symbol struct work? ";
      auto FoundSym = *llvm::cantFail(ElfObjFile.getSymbol(
          (*LookedUpSymbolUsingNameOrErr)->getRawDataRefImpl()));
      auto OrigSymThing =
          *llvm::cantFail(ElfObjFile.getSymbol(Sym.getRawDataRefImpl()));
      OS << (FoundSym.st_name == OrigSymThing.st_name) << "\n";
      OS << (FoundSym.st_value == OrigSymThing.st_value) << "\n";
      OS << (FoundSym.st_size == OrigSymThing.st_size) << "\n";
      OS << (FoundSym.st_info == OrigSymThing.st_info) << "\n";
      OS << (FoundSym.st_other == OrigSymThing.st_other) << "\n";
      OS << (FoundSym.st_shndx == OrigSymThing.st_shndx) << "\n";
    }
    // LUTHIER_REPORT_FATAL_ON_ERROR(
    //     LUTHIER_ERROR_CHECK((**LookedUpSymbolUsingNameOrErr) == Sym,
    //                         "Looked up symbol with name {0} does not match
    //                         the " "original symbol found using iteration",
    //                         *SymNameOrErr));
  }
  // Generate a random symbol name that's not inside the ELF, and look it up
  // inside the ELF to make sure we can't find it
  std::string RandomSymbolName;
  do {
    RandomSymbolName = getRandomString(10);
  } while (NameToSymbolMap.contains(RandomSymbolName));
  llvm::Expected<std::optional<llvm::object::ELFSymbolRef>>
      RandomSymbolLookupRes =
          luthier::object::lookupSymbolByName(ElfObjFile, RandomSymbolName);
  LUTHIER_REPORT_FATAL_ON_ERROR(RandomSymbolLookupRes.takeError());
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
      !RandomSymbolLookupRes->has_value(),
      "Found a symbol associated with a randomly generated string."));
  OS << "Passed symbol name lookup test.\n";
}

static void symbolLookupTest(const llvm::object::ObjectFile &ObjFile,
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
  }
}

#endif