//#include "llvm_mc.hpp"
//#include <llvm/MC/MCAsmBackend.h>
//#include <llvm/MC/MCAsmInfo.h>
//#include <llvm/MC/MCCodeEmitter.h>
//#include <llvm/MC/MCContext.h>
//#include <llvm/MC/MCDisassembler/MCDisassembler.h>
//#include <llvm/MC/MCInstPrinter.h>
//#include <llvm/MC/MCInstrAnalysis.h>
//#include <llvm/MC/MCInstrDesc.h>
//#include <llvm/MC/MCInstrInfo.h>
//#include <llvm/MC/MCObjectWriter.h>
//#include <llvm/MC/MCParser/AsmLexer.h>
//#include <llvm/MC/MCParser/MCAsmParser.h>
//#include <llvm/MC/MCParser/MCParsedAsmOperand.h>
//#include <llvm/MC/MCParser/MCTargetAsmParser.h>
//#include <llvm/MC/MCRegisterInfo.h>
//#include <llvm/MC/MCStreamer.h>
//#include <llvm/MC/MCSubtargetInfo.h>
//#include <llvm/MC/TargetRegistry.h>
//#include <llvm/Support/SourceMgr.h>
//#include <optional>
//
//namespace luthier::code {
//class InstructionAssemblyInfo {
// private:
//    llvm::MCContext context_;
//    llvm::AsmToken lexer_;
//
//};
//
//
//}
//
//
//
//struct ParseStatementInfo {
//    /// The parsed operands from the last parsed statement.
//    llvm::SmallVector<std::unique_ptr<llvm::MCParsedAsmOperand>, 8> ParsedOperands;
//
//    /// The opcode from the last parsed instruction.
//    unsigned Opcode = ~0U;
//
//    /// Was there an error parsing the inline assembly?
//    bool ParseError = false;
//
//    /// The value associated with a macro exit.
//    std::optional<std::string> ExitValue;
//
//    llvm::SmallVectorImpl<llvm::AsmRewrite> *AsmRewrites = nullptr;
//
//    ParseStatementInfo() = delete;
//    ParseStatementInfo(llvm::SmallVectorImpl<llvm::AsmRewrite> *rewrites)
//        : AsmRewrites(rewrites) {}
//};
//
//const llvm::AsmToken &Lex(llvm::AsmLexer &lexer) {
//    if (lexer.getTok().is(llvm::AsmToken::Error))
//        throw std::runtime_error(lexer.getErr());//TODO: fix this!
//
//    // if it's an end of statement with a comment in it
//    if (lexer.getTok().is(llvm::AsmToken::EndOfStatement)) {
//        // if this is a line comment output it.
//        if (!lexer.getTok().getString().empty() && lexer.getTok().getString().front() != '\n' && getTok().getString().front() != '\r' && MAI.preserveAsmComments())
//            Out.addExplicitComment(Twine(getTok().getString()));
//    }
//
//    const llvm::AsmToken *tok = &lexer.Lex();
//
//    // Parse comments here to be deferred until end of next statement.
//    while (tok->is(llvm::AsmToken::Comment)) {
//        if (MAI.preserveAsmComments())
//            Out.addExplicitComment(Twine(tok->getString()));
//        tok = &Lexer.Lex();
//    }
//
//    if (tok->is(llvm::AsmToken::Eof)) {
//        // If this is the end of an included file, pop the parent file off the
//        // include stack.
//        llvm::SMLoc ParentIncludeLoc = SrcMgr.getParentIncludeLoc(CurBuffer);
//        if (ParentIncludeLoc != SMLoc()) {
//            jumpToLoc(ParentIncludeLoc);
//            return Lex();
//        }
//    }
//
//    return *tok;
//}
//
//
//bool parseStatement(ParseStatementInfo &Info,
//                    llvm::MCAsmParserSemaCallback *SI,
//                    llvm::AsmLexer &lexer) {
//    assert(!hasPendingError() && "parseStatement started with pending error");
//    // Eat initial spaces and comments.
//    while (lexer.is(llvm::AsmToken::Space))
//        Lex(lexer);
//    if (lexer.is(llvm::AsmToken::EndOfStatement)) {
//        // If this is a line comment we can drop it safely.
//        if (lexer.getTok().getString().empty() || lexer.getTok().getString().front() == '\r' ||
//            lexer.getTok().getString().front() == '\n')
//            Out.addBlankLine();
//        Lex(lexer);
//        return false;
//    }
//
//    // Statements always start with an identifier, unless we're dealing with a
//    // processor directive (.386, .686, etc.) that lexes as a real.
//    llvm::AsmToken ID = lexer.getTok();
//    llvm::SMLoc IDLoc = ID.getLoc();
//    llvm::StringRef IDVal;
//    if (lexer.is(llvm::AsmToken::Dot)) {
//        // Treat '.' as a valid identifier in this context.
//        Lex(lexer);
//        IDVal = ".";
//    } else if (lexer.is(llvm::AsmToken::Real)) {
//        // Treat ".<number>" as a valid identifier in this context.
//        IDVal = lexer.getTok().getString();
//        Lex(); // always eat a token
//        if (!IDVal.startswith("."))
//            return Error(IDLoc, "unexpected token at start of statement");
//    } else if (parseIdentifier(IDVal, StartOfStatement)) {
//        if (!TheCondState.Ignore) {
//            Lex(); // always eat a token
//            return Error(IDLoc, "unexpected token at start of statement");
//        }
//        IDVal = "";
//    }
//
//    // Ignore the statement if in the middle of inactive conditional
//    // (e.g. ".if 0").
//    if (TheCondState.Ignore) {
//        eatToEndOfStatement();
//        return false;
//    }
//
//    // We also check if this is allocating memory with user-defined type.
//    auto IDIt = Structs.find(IDVal.lower());
//    if (IDIt != Structs.end())
//        return parseDirectiveStructValue(/*Structure=*/IDIt->getValue(), IDVal,
//                                         IDLoc);
//
//    // Non-conditional Microsoft directives sometimes follow their first argument.
//    const AsmToken nextTok = getTok();
//    const StringRef nextVal = nextTok.getString();
//    const SMLoc nextLoc = nextTok.getLoc();
//
//    const AsmToken afterNextTok = peekTok();
//
//    // There are several entities interested in parsing infix directives:
//    //
//    // 1. Asm parser extensions. For example, platform-specific parsers
//    //    (like the ELF parser) register themselves as extensions.
//    // 2. The generic directive parser implemented by this class. These are
//    //    all the directives that behave in a target and platform independent
//    //    manner, or at least have a default behavior that's shared between
//    //    all targets and platforms.
//
//    getTargetParser().flushPendingInstructions(getStreamer());
//
//
//    // __asm _emit or __asm __emit
//    if (ParsingMSInlineAsm && (IDVal == "_emit" || IDVal == "__emit" ||
//                               IDVal == "_EMIT" || IDVal == "__EMIT"))
//        return parseDirectiveMSEmit(IDLoc, Info, IDVal.size());
//
//    // __asm align
//    if (ParsingMSInlineAsm && (IDVal == "align" || IDVal == "ALIGN"))
//        return parseDirectiveMSAlign(IDLoc, Info);
//
//    if (ParsingMSInlineAsm && (IDVal == "even" || IDVal == "EVEN"))
//        Info.AsmRewrites->emplace_back(AOK_EVEN, IDLoc, 4);
//    if (checkForValidSection())
//        return true;
//
//    // Canonicalize the opcode to lower case.
//    std::string OpcodeStr = IDVal.lower();
//    ParseInstructionInfo IInfo(Info.AsmRewrites);
//    bool ParseHadError = getTargetParser().ParseInstruction(IInfo, OpcodeStr, ID,
//                                                            Info.ParsedOperands);
//    Info.ParseError = ParseHadError;
//
//    // Dump the parsed representation, if requested.
//    if (getShowParsedOperands()) {
//        SmallString<256> Str;
//        raw_svector_ostream OS(Str);
//        OS << "parsed instruction: [";
//        for (unsigned i = 0; i != Info.ParsedOperands.size(); ++i) {
//            if (i != 0)
//                OS << ", ";
//            Info.ParsedOperands[i]->print(OS);
//        }
//        OS << "]";
//
//        printMessage(IDLoc, SourceMgr::DK_Note, OS.str());
//    }
//
//    // Fail even if ParseInstruction erroneously returns false.
//    if (hasPendingError() || ParseHadError)
//        return true;
//
//    // If parsing succeeded, match the instruction.
//    if (!ParseHadError) {
//        uint64_t ErrorInfo;
//        if (getTargetParser().MatchAndEmitInstruction(
//                IDLoc, Info.Opcode, Info.ParsedOperands, Out, ErrorInfo,
//                getTargetParser().isParsingMSInlineAsm()))
//            return true;
//    }
//    return false;
//}
//
//llvm::MCInst luthier::code::InstrParser::makeInstruction(const std::string &instr, const luthier::hsa::Isa &isa) {
//    auto isaName = isa.getName();
//
//    std::string Error;
//    const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget(isaName, Error);
//    assert(TheTarget);
//
//    std::unique_ptr<const llvm::MCRegisterInfo> MRI(TheTarget->createMCRegInfo(llvm::StringRef(isaName)));
//    assert(MRI);
//
//    llvm::MCTargetOptions MCOptions;
//    std::unique_ptr<const llvm::MCAsmInfo> MAI(
//        TheTarget->createMCAsmInfo(*MRI, isaName, MCOptions));
//
//    assert(MAI);
//
//    std::unique_ptr<const llvm::MCInstrInfo> MII(TheTarget->createMCInstrInfo());
//    assert(MII);
//
//    std::unique_ptr<const llvm::MCSubtargetInfo> STI(
//        TheTarget->createMCSubtargetInfo(isaName, "gfx908", "+sramecc-xnack"));
//    assert(STI);
//
//    // MatchAndEmitInstruction in MCTargetAsmParser.h
//
//    // Now that GetTarget() has (potentially) replaced TripleName, it's safe to
//    // construct the Triple object.
//    llvm::Triple TheTriple(isaName);
//
//    std::unique_ptr<llvm::MemoryBuffer> BufferPtr = llvm::MemoryBuffer::getMemBuffer(instr);
//
//    llvm::SourceMgr SrcMgr;
//
//    // Tell SrcMgr about this buffer, which is what the parser will pick up.
//    SrcMgr.AddNewSourceBuffer(std::move(BufferPtr), llvm::SMLoc());
//
//    // Package up features to be passed to target/subtarget
//    std::string FeaturesStr;
//    //    if (MAttrs.size()) {
//    //        SubtargetFeatures Features;
//    //        for (unsigned i = 0; i != MAttrs.size(); ++i)
//    //            Features.AddFeature(MAttrs[i]);
//    //        FeaturesStr = Features.getString();
//    //    }
//
//    //    std::unique_ptr<llvm::MCContext> Ctx(new (std::nothrow)
//    //                                             llvm::MCContext(llvm::Triple(isaName), MAI.get(), MRI.get(),
//    //                                                             &SrcMgr,
//    //                                                             &MCOptions,
//    //                                                             STI.get()));
//    //    assert(Ctx);
//
//    // Prime the lexer.
//    llvm::AsmLexer lexer(*MAI);
//    Lex(lexer);
//
//    bool HadError = false;
//    llvm::AsmCond StartingCondState = TheCondState;
//    llvm::SmallVector<llvm::AsmRewrite, 4> AsmStrRewrites;
//
//
//    getTargetParser().onBeginOfFile();
//
//    // While we have input, parse each statement.
//    while (lexer.isNot(llvm::AsmToken::Eof) || SrcMgr.getParentIncludeLoc(CurBuffer) != llvm::SMLoc()) {
//        // Skip through the EOF at the end of an inclusion.
//        if (lexer.is(llvm::AsmToken::Eof))
//            Lex(lexer);
//
//        ParseStatementInfo Info(&AsmStrRewrites);
//        bool Parsed = parseStatement(Info, nullptr);
//
//        // If we have a Lexer Error we are on an Error Token. Load in Lexer Error
//        // for printing ErrMsg via Lex() only if no (presumably better) parser error
//        // exists.
//        if (Parsed && !hasPendingError() && Lexer.getTok().is(AsmToken::Error)) {
//            Lex();
//        }
//
//        // parseStatement returned true so may need to emit an error.
//        printPendingErrors();
//
//        // Skipping to the next line if needed.
//        if (Parsed && !getLexer().isAtStartOfStatement())
//            eatToEndOfStatement();
//    }
//
//    getTargetParser().onEndOfFile();
//    printPendingErrors();
//
//    // All errors should have been emitted.
//    assert(!hasPendingError() && "unexpected error from parseStatement");
//
//    getTargetParser().flushPendingInstructions(getStreamer());
//
//    if (TheCondState.TheCond != StartingCondState.TheCond || TheCondState.Ignore != StartingCondState.Ignore)
//        printError(getTok().getLoc(), "unmatched .ifs or .elses");
//    // Check to see there are no empty DwarfFile slots.
//    const auto &LineTables = getContext().getMCDwarfLineTables();
//    if (!LineTables.empty()) {
//        unsigned Index = 0;
//        for (const auto &File: LineTables.begin()->second.getMCDwarfFiles()) {
//            if (File.Name.empty() && Index != 0)
//                printError(getTok().getLoc(), "unassigned file number: " + Twine(Index) + " for .file directives");
//            ++Index;
//        }
//    }
//
//    // Check to see that all assembler local symbols were actually defined.
//    // Targets that don't do subsections via symbols may not want this, though,
//    // so conservatively exclude them. Only do this if we're finalizing, though,
//    // as otherwise we won't necessarilly have seen everything yet.
//    if (!NoFinalize) {
//        if (MAI.hasSubsectionsViaSymbols()) {
//            for (const auto &TableEntry: getContext().getSymbols()) {
//                MCSymbol *Sym = TableEntry.getValue();
//                // Variable symbols may not be marked as defined, so check those
//                // explicitly. If we know it's a variable, we have a definition for
//                // the purposes of this check.
//                if (Sym->isTemporary() && !Sym->isVariable() && !Sym->isDefined())
//                    // FIXME: We would really like to refer back to where the symbol was
//                    // first referenced for a source location. We need to add something
//                    // to track that. Currently, we just point to the end of the file.
//                    printError(getTok().getLoc(), "assembler local symbol '" + Sym->getName() + "' not defined");
//            }
//        }
//
//        // Temporary symbols like the ones for directional jumps don't go in the
//        // symbol table. They also need to be diagnosed in all (final) cases.
//        for (std::tuple<SMLoc, CppHashInfoTy, MCSymbol *> &LocSym: DirLabels) {
//            if (std::get<2>(LocSym)->isUndefined()) {
//                // Reset the state of any "# line file" directives we've seen to the
//                // context as it was at the diagnostic site.
//                CppHashInfo = std::get<1>(LocSym);
//                printError(std::get<0>(LocSym), "directional label undefined");
//            }
//        }
//    }
//
//    // Finalize the output stream if there are no errors and if the client wants
//    // us to.
//    if (!HadError && !NoFinalize)
//        Out.finish(Lexer.getLoc());
//
//    return HadError || getContext().hadError();
//}
