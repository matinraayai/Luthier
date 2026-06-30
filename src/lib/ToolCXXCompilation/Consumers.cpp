//===-- Consumers.cpp -----------------------------------------------------===//
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
#include "luthier/ToolCXXCompilation/Consumers.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/Cuda.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Sema/Sema.h>
#include <clang/Sema/SemaCUDA.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/IOSandbox.h>

namespace luthier {

namespace {

/// Provides the printed canonical declaration location of a function (or, for
/// a template specialization, its primary template's location) printed as
/// \c file:line:col. As Identical source parsed twice yields identical
/// locations, this can be used as a ASTContext-independent identifier for a
/// function
std::string keyOf(const clang::SourceManager &SM,
                  const clang::FunctionDecl *FD) {
  const clang::FunctionDecl *K = FD;
  if (clang::FunctionTemplateDecl *PT = FD->getPrimaryTemplate())
    K = PT->getTemplatedDecl();
  return K->getCanonicalDecl()->getLocation().printToString(SM);
}

/// Annotates \p FD with the \c ExportFunctionHandleMarker if not already
/// annotated. The check is on \p FD alone, not its re-declarations.
void annotateExportHandle(clang::ASTContext &Ctx, clang::FunctionDecl *FD) {
  for (const auto *A : FD->specific_attrs<clang::AnnotateAttr>())
    if (A->getAnnotation() == ExportFunctionHandleMarker)
      return;
  FD->addAttr(clang::AnnotateAttr::Create(Ctx, ExportFunctionHandleMarker,
                                          /*Args=*/nullptr, /*NumArgs=*/0,
                                          FD->getLocation()));
}

/// Attributes that are only meaningful on device code and must not be copied
/// onto a \c __host__ handle.
bool isDeviceOnlyAttr(const clang::Attr *A) {
  switch (A->getKind()) {
  case clang::attr::CUDADevice:
  case clang::attr::CUDAGlobal:
  case clang::attr::CUDAShared:
  case clang::attr::CUDAConstant:
  case clang::attr::CUDALaunchBounds:
    return true;
  default:
    return false;
  }
}

/// Returns a pre-existing \c __host__ / \c __host__ \c __device__ overload of
/// \p Dev with the same signature (a user-written sibling, or the host half of
/// a standard-library overload set), or nullptr if none exists.
clang::FunctionDecl *findHostOverload(clang::Sema &S,
                                      const clang::FunctionDecl *Dev) {
  clang::ASTContext &Ctx = S.Context;
  const clang::FunctionDecl *Canon = Dev->getCanonicalDecl();
  for (clang::NamedDecl *ND :
       Dev->getDeclContext()->lookup(Dev->getDeclName())) {
    auto *Other = llvm::dyn_cast<clang::FunctionDecl>(ND);
    if (!Other)
      if (auto *FTD = llvm::dyn_cast<clang::FunctionTemplateDecl>(ND))
        Other = FTD->getTemplatedDecl();
    if (!Other || Other->getCanonicalDecl() == Canon)
      continue;
    /// Ignore the exception specification: a __device__ function and its host
    /// peer routinely differ there (e.g. glibc's `malloc` is `__THROW` while
    /// HIP's __device__ `malloc` is not), yet they are the same overload.
    if (!Ctx.hasSameFunctionTypeIgnoringExceptionSpec(Other->getType(),
                                                      Dev->getType()))
      continue;
    clang::CUDAFunctionTarget T = S.CUDA().IdentifyTarget(Other);
    if (T == clang::CUDAFunctionTarget::Host ||
        T == clang::CUDAFunctionTarget::HostDevice)
      return Other;
  }
  return nullptr;
}

/// Deep-clones \p Src's parameters and sets them as \p Dst's parameters.
void cloneParams(clang::ASTContext &Ctx, const clang::FunctionDecl *Src,
                 clang::FunctionDecl *Dst) {
  llvm::SmallVector<clang::ParmVarDecl *, 4> Params;
  for (const clang::ParmVarDecl *P : Src->parameters()) {
    auto *NP = clang::ParmVarDecl::Create(
        Ctx, Dst, P->getBeginLoc(), P->getLocation(), P->getIdentifier(),
        P->getType(), P->getTypeSourceInfo(), P->getStorageClass(),
        /*DefArg=*/nullptr);
    NP->setScopeInfo(/*scopeDepth=*/0, /*parameterIndex=*/Params.size());
    Params.push_back(NP);
  }
  Dst->setParams(Params);
}

/// Copies every non-device-only attribute of \p Src onto \p Dst (this carries
/// \c used, visibility, etc.), then forces \p Dst \c __host__. Also skips
/// \c ExportFunctionHandleMarker annotations if present in \p Src.
void copyHostAttrs(clang::ASTContext &Ctx, const clang::FunctionDecl *Src,
                   clang::FunctionDecl *Dst) {
  for (const clang::Attr *A : Src->attrs()) {
    if (isDeviceOnlyAttr(A))
      continue;
    if (const auto *Ann = llvm::dyn_cast<clang::AnnotateAttr>(A);
        Ann && Ann->getAnnotation() == ExportFunctionHandleMarker)
      continue;
    Dst->addAttr(A->clone(Ctx));
  }
  if (!Dst->hasAttr<clang::CUDAHostAttr>())
    Dst->addAttr(clang::CUDAHostAttr::CreateImplicit(Ctx));
}

/// Synthesizes a \c __host__ handle for the \c __device__ function \p Dev in
/// \p DC.
clang::FunctionDecl *makeHostHandle(clang::Sema &S, clang::FunctionDecl *Dev,
                                    clang::DeclContext *DC) {
  /// Clone the canonical declaration, since a non-canonical redeclarations
  /// doesn't have all the attributes of the canonical.
  Dev = Dev->getCanonicalDecl();
  clang::ASTContext &Ctx = S.Context;
  clang::FunctionDecl *Host;
  if (auto *MD = llvm::dyn_cast<clang::CXXMethodDecl>(Dev)) {
    Host = clang::CXXMethodDecl::Create(
        Ctx, llvm::cast<clang::CXXRecordDecl>(DC), MD->getBeginLoc(),
        MD->getNameInfo(), MD->getType(), MD->getTypeSourceInfo(),
        MD->getStorageClass(), /*UsesFPIntrin=*/false, MD->isInlineSpecified(),
        MD->getConstexprKind(), MD->getEndLoc());
  } else {
    Host = clang::FunctionDecl::Create(
        Ctx, DC, Dev->getBeginLoc(), Dev->getLocation(), Dev->getDeclName(),
        Dev->getType(), Dev->getTypeSourceInfo(), Dev->getStorageClass(),
        /*UsesFPIntrin=*/false, Dev->isInlineSpecified(),
        /*hasWrittenPrototype=*/true, Dev->getConstexprKind());
  }
  Host->setAccess(Dev->getAccess());
  cloneParams(Ctx, Dev, Host);
  copyHostAttrs(Ctx, Dev, Host);
  Host->setBody(
      clang::CompoundStmt::Create(Ctx, /*Stmts=*/{}, clang::FPOptionsOverride(),
                                  Host->getLocation(), Host->getLocation()));
  annotateExportHandle(Ctx, Host);
  return Host;
}

//===----------------------------------------------------------------------===//
// Pre-pass: discover which __device__ functions need a synthesized host handle.
//
// A device function only needs one if host code references it (which is
// ill-formed and would otherwise be lost as a RecoveryExpr) or it is `used`,
// AND it has no __host__ overload already. The references are recovered from
// the err_ref_bad_target diagnostics the pre-pass provokes; the "no host
// overload" question is answered against the completed AST (so the standard
// library's host malloc/sqrt, declared after their __device__ peers, are seen).
//===----------------------------------------------------------------------===//

/// Wraps the main compilation instance's \c clang::DiagnosticConsumer, and
/// forwards all errors to it besides \c err_ref_bad_target errors caused by
/// \c __device__ functions referenced inside \c __host__ code. The referenced
/// \c __device__ functions are recorded for further processing by stage 2.
class BadRefCollector : public clang::DiagnosticConsumer {
  llvm::StringSet<> &Referenced;
  clang::DiagnosticConsumer &Inner;
  bool DroppingNotes = false;

  static int64_t asInt(const clang::Diagnostic &Info, unsigned I) {
    return Info.getArgKind(I) == clang::DiagnosticsEngine::ak_uint
               ? static_cast<int64_t>(Info.getArgUInt(I))
               : Info.getArgSInt(I);
  }

  static bool isDeviceFromHostBadRef(const clang::Diagnostic &Info) {
    if (Info.getID() != clang::diag::err_ref_bad_target ||
        Info.getNumArgs() < 4 ||
        Info.getArgKind(2) != clang::DiagnosticsEngine::ak_nameddecl)
      return false;
    return asInt(Info, 0) ==
               static_cast<int64_t>(clang::CUDAFunctionTarget::Device) &&
           asInt(Info, 3) ==
               static_cast<int64_t>(clang::CUDAFunctionTarget::Host);
  }

public:
  BadRefCollector(llvm::StringSet<> &Referenced,
                  clang::DiagnosticConsumer &Inner)
      : Referenced(Referenced), Inner(Inner) {}

  // Only HandleDiagnostic forwards to Inner; the base BeginSourceFile /
  // EndSourceFile will be called by the main action
  void HandleDiagnostic(clang::DiagnosticsEngine::Level L,
                        const clang::Diagnostic &Info) override {
    if (isDeviceFromHostBadRef(Info)) {
      auto *ND = reinterpret_cast<clang::NamedDecl *>(Info.getRawArg(2));
      if (auto *FD = llvm::dyn_cast<clang::FunctionDecl>(ND))
        Referenced.insert(keyOf(FD->getASTContext().getSourceManager(), FD));
      DroppingNotes = true; // also drop the bad-ref's "declared here" note(s)
      return;
    }
    if (L == clang::DiagnosticsEngine::Note && DroppingNotes)
      return;
    DroppingNotes = false;
    Inner.HandleDiagnostic(L, Info);
  }
};

/// Collects the location keys of functions whose address is \e taken in host
/// code, as opposed to being merely called — taking a device function's address
/// is what warrants a host handle, a plain call does not. Device-\e only
/// address-takes are ill-formed and surface as err_ref_bad_target (see
/// \c BadRefCollector); this catches the legal case where a host-callable
/// counterpart (a user \c __host__ overload, a \c __host__ \c __device__
/// function, or an external like libm's \c sqrt) already exists.
class AddrTakeCollector : public clang::RecursiveASTVisitor<AddrTakeCollector> {
  clang::Sema &S;
  llvm::StringSet<> &AddressTaken;
  llvm::SmallVector<const clang::FunctionDecl *, 8> Enclosing;
  llvm::SmallPtrSet<const clang::Stmt *, 32> CalleeRefs;

  bool inHostContext() const {
    if (Enclosing.empty())
      return true; // file-scope initializer: runs on host
    clang::CUDAFunctionTarget T = S.CUDA().IdentifyTarget(Enclosing.back());
    return T == clang::CUDAFunctionTarget::Host ||
           T == clang::CUDAFunctionTarget::HostDevice;
  }

public:
  AddrTakeCollector(clang::Sema &S, llvm::StringSet<> &AddressTaken)
      : S(S), AddressTaken(AddressTaken) {}

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool TraverseFunctionDecl(clang::FunctionDecl *FD) {
    Enclosing.push_back(FD);
    bool Ok =
        clang::RecursiveASTVisitor<AddrTakeCollector>::TraverseFunctionDecl(FD);
    Enclosing.pop_back();
    return Ok;
  }
  bool TraverseCXXMethodDecl(clang::CXXMethodDecl *MD) {
    Enclosing.push_back(MD);
    bool Ok =
        clang::RecursiveASTVisitor<AddrTakeCollector>::TraverseCXXMethodDecl(
            MD);
    Enclosing.pop_back();
    return Ok;
  }

  bool VisitCallExpr(clang::CallExpr *CE) {
    // Visited before the callee's DeclRefExpr (pre-order), so the callee is
    // recorded by the time VisitDeclRefExpr sees it.
    if (const clang::Expr *Callee = CE->getCallee())
      CalleeRefs.insert(Callee->IgnoreParenImpCasts());
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *E) {
    if (CalleeRefs.contains(E))
      return true; // a call, not an address-take
    auto *FD = llvm::dyn_cast<clang::FunctionDecl>(E->getDecl());
    if (!FD || !inHostContext())
      return true;
    AddressTaken.insert(keyOf(S.getSourceManager(), FD));
    return true;
  }
};

/// Walks the \e completed AST and decides, for each \c __device__ function, how
/// the real parse should make it host-addressable:
///   - \c Synthesize: device-only with no host overload, taken-or-\c used from
///     host — the real parse will emit a \c __host__ handle.
///   - \c Annotate: a host-callable counterpart already exists (user overload,
///     \c __host__ \c __device__, or external) — the real parse tags it.
class ExportPlanConsumer
    : public clang::SemaConsumer,
      public clang::RecursiveASTVisitor<ExportPlanConsumer> {
  llvm::StringSet<> &Synthesize;
  llvm::StringSet<> &Annotate;
  const llvm::StringSet<> &Referenced;
  llvm::StringSet<> AddressTaken;
  clang::Sema *SemaRef{nullptr};

public:
  ExportPlanConsumer(llvm::StringSet<> &Synthesize, llvm::StringSet<> &Annotate,
                     const llvm::StringSet<> &Referenced)
      : Synthesize(Synthesize), Annotate(Annotate), Referenced(Referenced) {}

  void InitializeSema(clang::Sema &S) override { SemaRef = &S; }
  void ForgetSema() override { SemaRef = nullptr; }

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitFunctionDecl(clang::FunctionDecl *FD) {
    if (!SemaRef)
      return true;
    // Classify at the template level, not per instantiation: a specialization's
    // concrete signature wouldn't match a host *template* overload's dependent
    // one. (A taken specialization is still seen — AddressTaken keys it to the
    // pattern.)
    if (FD->getPrimaryTemplate())
      return true;
    clang::Sema &S = *SemaRef;
    const clang::SourceManager &SM = S.getSourceManager();
    bool DevUsed = FD->hasAttr<clang::UsedAttr>();
    clang::CUDAFunctionTarget T = S.CUDA().IdentifyTarget(FD);

    // __host__ __device__: the function is its own host counterpart.
    if (T == clang::CUDAFunctionTarget::HostDevice) {
      std::string Key = keyOf(SM, FD);
      if (DevUsed || AddressTaken.contains(Key))
        Annotate.insert(Key);
      return true;
    }
    if (T != clang::CUDAFunctionTarget::Device)
      return true;

    // A host-callable counterpart already exists: tag it if it is taken from
    // host (a plain call doesn't warrant a handle) or the device source is
    // used.
    if (clang::FunctionDecl *Host = findHostOverload(S, FD)) {
      std::string HostKey = keyOf(SM, Host);
      if (DevUsed || AddressTaken.contains(HostKey))
        Annotate.insert(HostKey);
      return true;
    }
    // Device-only with no host overload: synthesize a handle if host code took
    // its address (err_ref_bad_target, via Referenced) or it is `used`.
    std::string Key = keyOf(SM, FD);
    if (Referenced.contains(Key) || DevUsed)
      Synthesize.insert(Key);
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    if (!SemaRef)
      return;
    AddrTakeCollector(*SemaRef, AddressTaken)
        .TraverseDecl(Ctx.getTranslationUnitDecl());
    TraverseDecl(Ctx.getTranslationUnitDecl());
  }
};

class ExportPlanAction : public clang::ASTFrontendAction {
  llvm::StringSet<> &Synthesize;
  llvm::StringSet<> &Annotate;
  const llvm::StringSet<> &Referenced;

public:
  ExportPlanAction(llvm::StringSet<> &Synthesize, llvm::StringSet<> &Annotate,
                   const llvm::StringSet<> &Referenced)
      : Synthesize(Synthesize), Annotate(Annotate), Referenced(Referenced) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
    return std::make_unique<ExportPlanConsumer>(Synthesize, Annotate,
                                                Referenced);
  }
};

/// Runs a syntax-only pre-parse of \p MainCI's translation unit to
/// fill \p Synthesize (device-only functions needing a synthesized host handle)
/// and \p Annotate (existing host counterparts to tag). Unlike the streaming
/// real parse, this inspects the \e complete AST, so a host overload declared
/// after its \c __device__ peer (the standard library's \c malloc/\c sqrt) is
/// recognized. Genuine (non-bad-ref) errors are routed through \p MainCI's own
/// diagnostic client, so they are displayed and counted toward the exit status.
void computeExportPlan(clang::CompilerInstance &MainCI,
                       llvm::StringSet<> &Synthesize,
                       llvm::StringSet<> &Annotate) {
  const clang::LangOptions &LO = MainCI.getLangOpts();
  if (!(LO.CUDA && !LO.CUDAIsDevice))
    return;

  // The main compilation runs with the LLVM IO sandbox armed; our pre-pass
  // legitimately re-reads the same input files, so disable it for the duration.
  auto SandboxOff = llvm::sys::sandbox::scopedDisable();

  auto Inv =
      std::make_shared<clang::CompilerInvocation>(MainCI.getInvocation());
  // Drop this plugin from the cloned invocation so the pre-pass does not
  // re-instantiate it and recurse: the plugin is a CmdlineBeforeMainAction,
  // which the frontend only runs when its name is listed in AddPluginActions.
  Inv->getFrontendOpts().AddPluginActions.clear();

  llvm::StringSet<> Referenced;
  clang::CompilerInstance PreCI(Inv, MainCI.getPCHContainerOperations());
  /// Wrap the main diagnositcs handler with the \c BadRefCollector
  PreCI.createDiagnostics(
      new BadRefCollector(Referenced, *MainCI.getDiagnostics().getClient()),
      /*ShouldOwnClient=*/true);
  /// Don't let the provoked errors trip the error limit and stop parsing early
  PreCI.getDiagnostics().setErrorLimit(0);

  ExportPlanAction Action(Synthesize, Annotate, Referenced);
  PreCI.ExecuteAction(Action);
}

} // namespace

EmitHostHandleForDevFuncConsumer::EmitHostHandleForDevFuncConsumer(
    clang::CompilerInstance &CI) {
  /// The pre-pass forwards genuine errors to the main client; a bump in its
  /// error count means a real error was reported, so the real parse must abort.
  clang::DiagnosticConsumer *Client = CI.getDiagnostics().getClient();
  unsigned ErrorsBefore = Client ? Client->getNumErrors() : 0;
  computeExportPlan(CI, Synthesize, Annotate);
  PrePassFailed = Client && Client->getNumErrors() > ErrorsBefore;
}

void EmitHostHandleForDevFuncConsumer::InitializeSema(clang::Sema &S) {
  SemaRef = &S;
  Synthesized.clear();
}

void EmitHostHandleForDevFuncConsumer::ForgetSema() { SemaRef = nullptr; }

bool EmitHostHandleForDevFuncConsumer::HandleTopLevelDecl(
    clang::DeclGroupRef DG) {
  /// The pre-parse already reported a genuine error through the main client;
  /// abort the real parse so it neither re-reports nor emits anything.
  if (PrePassFailed)
    return false;
  if (!SemaRef)
    return true;
  clang::ASTContext &Ctx = SemaRef->Context;
  if (!(Ctx.getLangOpts().CUDA && !Ctx.getLangOpts().CUDAIsDevice))
    return true;

  clang::Sema &S = *SemaRef;
  const clang::SourceManager &SM = S.getSourceManager();

  // Walks one declaration, recursing through namespace / linkage-spec scopes
  // and the tool's own records
  llvm::SmallVector<clang::Decl *, 16> Worklist(DG.begin(), DG.end());
  while (!Worklist.empty()) {
    clang::Decl *D = Worklist.pop_back_val();

    if (llvm::isa<clang::NamespaceDecl, clang::LinkageSpecDecl>(D)) {
      for (clang::Decl *Sub : llvm::cast<clang::DeclContext>(D)->decls())
        Worklist.push_back(Sub);
      continue;
    }
    if (auto *RD = llvm::dyn_cast<clang::CXXRecordDecl>(D)) {
      if (RD->isThisDeclarationADefinition())
        for (clang::Decl *Sub : RD->decls())
          Worklist.push_back(Sub);
      continue;
    }

    clang::FunctionDecl *Dev = nullptr;
    clang::FunctionTemplateDecl *DevTpl = nullptr;
    if (auto *FTD = llvm::dyn_cast<clang::FunctionTemplateDecl>(D)) {
      DevTpl = FTD;
      Dev = FTD->getTemplatedDecl();
    } else {
      Dev = llvm::dyn_cast<clang::FunctionDecl>(D);
    }
    if (!Dev)
      continue;

    std::string Key = keyOf(SM, Dev);

    /// Tag existing host handles or synthesize new one here
    if (Annotate.contains(Key)) {
      annotateExportHandle(Ctx, Dev);
      continue;
    }

    if (S.CUDA().IdentifyTarget(Dev) != clang::CUDAFunctionTarget::Device)
      continue;
    if (!Synthesize.contains(Key))
      continue;
    // A function with separate declaration + definition is seen twice; only the
    // first encounter synthesizes the handle.
    if (!Synthesized.insert(Key).second)
      continue;

    clang::DeclContext *DC = Dev->getDeclContext();
    bool IsMember = DC->isRecord();

    // The pre-pass guarantees no host overload exists, so the handle is
    // finalized immediately: an empty body and the export annotation. For a
    // template, both are placed on the pattern and inherited by every
    // instantiation the host references trigger.
    clang::FunctionDecl *HostPattern = makeHostHandle(S, Dev, DC);

    if (DevTpl) {
      auto *HostTpl = clang::FunctionTemplateDecl::Create(
          Ctx, DC, DevTpl->getLocation(), DevTpl->getDeclName(),
          DevTpl->getTemplateParameters(), HostPattern);
      HostTpl->setAccess(DevTpl->getAccess());
      HostPattern->setDescribedFunctionTemplate(HostTpl);
      DC->addDecl(HostTpl);
      if (!IsMember)
        S.PushOnScopeChains(HostTpl, S.TUScope, /*AddToContext=*/false);
    } else {
      DC->addDecl(HostPattern);
      if (!IsMember)
        S.PushOnScopeChains(HostPattern, S.TUScope, /*AddToContext=*/false);
      S.MarkFunctionReferenced(HostPattern->getLocation(), HostPattern,
                               /*MightBeOdrUse=*/true);
      S.Consumer.HandleTopLevelDecl(clang::DeclGroupRef(HostPattern));
    }
  }
  return true;
}

} // namespace luthier
