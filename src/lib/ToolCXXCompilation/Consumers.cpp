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
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/IOSandbox.h>

namespace luthier {

namespace {

/// A stable, ASTContext-independent key for a function: the canonical
/// declaration location (or, for a template specialization, its primary
/// template's location) printed as \c file:line:col. Identical source parsed
/// twice yields identical keys, so the pre-pass and the real parse agree.
std::string keyOf(const clang::SourceManager &SM,
                  const clang::FunctionDecl *FD) {
  const clang::FunctionDecl *K = FD;
  if (clang::FunctionTemplateDecl *PT = FD->getPrimaryTemplate())
    K = PT->getTemplatedDecl();
  return K->getCanonicalDecl()->getLocation().printToString(SM);
}

/// Returns true if any redeclaration of \p FD already carries an
/// \c annotate(\p Tag) attribute.
bool hasAnnotation(const clang::FunctionDecl *FD, llvm::StringRef Tag) {
  for (const clang::FunctionDecl *ReDecl : FD->redecls())
    for (const auto *A : ReDecl->specific_attrs<clang::AnnotateAttr>())
      if (A->getAnnotation() == Tag)
        return true;
  return false;
}

/// Tags \p FD with the export-handle marker the IR pass harvests from
/// \c @llvm.global.annotations, unless an equal annotation is already present.
void annotateExportHandle(clang::ASTContext &Ctx, clang::FunctionDecl *FD) {
  if (hasAnnotation(FD, ExportFunctionHandleMarker))
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
    // Ignore the exception specification: a __device__ function and its host
    // peer routinely differ there (e.g. glibc's `malloc` is `__THROW` while
    // HIP's __device__ `malloc` is not), yet they are the same overload.
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

/// Deep-clones \p Src's parameters into \p Dst.
void cloneParams(clang::ASTContext &Ctx, const clang::FunctionDecl *Src,
                 clang::FunctionDecl *Dst,
                 llvm::SmallVectorImpl<clang::ParmVarDecl *> &Out) {
  for (const clang::ParmVarDecl *P : Src->parameters()) {
    auto *NP = clang::ParmVarDecl::Create(
        Ctx, Dst, P->getBeginLoc(), P->getLocation(), P->getIdentifier(),
        P->getType(), P->getTypeSourceInfo(), P->getStorageClass(),
        /*DefArg=*/nullptr);
    NP->setScopeInfo(/*scopeDepth=*/0, /*parameterIndex=*/Out.size());
    Out.push_back(NP);
  }
}

/// Copies every non-device-only attribute of \p Src onto \p Dst (this carries
/// \c used, visibility, etc.), then forces \p Dst \c __host__.
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

/// Creates a body-less \c __host__ clone of the \c __device__ function \p Dev
/// in \p DC, mirroring its name, signature and access specifier so the clone
/// merges with any \c __host__ overload the tool defines for itself.
clang::FunctionDecl *cloneHostDecl(clang::Sema &S, clang::FunctionDecl *Dev,
                                   clang::DeclContext *DC) {
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

  llvm::SmallVector<clang::ParmVarDecl *, 4> Params;
  cloneParams(Ctx, Dev, Host, Params);
  Host->setParams(Params);

  copyHostAttrs(Ctx, Dev, Host);
  return Host;
}

/// Gives \p Host an empty body if no definition of it exists yet, then tags it
/// with the export-handle marker and marks it referenced so it is emitted. When
/// the tool defined its own \c __host__ overload, \p Host already merged with
/// it; the existing definition is kept and only annotated.
void finalizeHostHandle(clang::Sema &S, clang::FunctionDecl *Host) {
  clang::ASTContext &Ctx = S.Context;
  clang::FunctionDecl *Def = Host->getDefinition();
  if (!Def) {
    Host->setBody(clang::CompoundStmt::Create(
        Ctx, /*Stmts=*/{}, clang::FPOptionsOverride(), Host->getLocation(),
        Host->getLocation()));
    Def = Host;
  }
  annotateExportHandle(Ctx, Def);
  S.MarkFunctionReferenced(Def->getLocation(), Def, /*MightBeOdrUse=*/true);
  // CodeGen already streamed past this decl during parsing (as a body-less
  // declaration); re-feed it so the freshly attached body and annotation are
  // emitted.
  S.Consumer.HandleTopLevelDecl(clang::DeclGroupRef(Def));
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

/// Collects the \c __device__ callees of host code from the bad-target
/// diagnostics, keyed exactly like \c keyOf. Suppresses all output.
class BadRefCollector : public clang::DiagnosticConsumer {
  llvm::StringSet<> &Referenced;

  static int64_t asInt(const clang::Diagnostic &Info, unsigned I) {
    return Info.getArgKind(I) == clang::DiagnosticsEngine::ak_uint
               ? static_cast<int64_t>(Info.getArgUInt(I))
               : Info.getArgSInt(I);
  }

public:
  explicit BadRefCollector(llvm::StringSet<> &R) : Referenced(R) {}

  void HandleDiagnostic(clang::DiagnosticsEngine::Level,
                        const clang::Diagnostic &Info) override {
    if (Info.getID() != clang::diag::err_ref_bad_target || Info.getNumArgs() < 4)
      return;
    if (Info.getArgKind(2) != clang::DiagnosticsEngine::ak_nameddecl)
      return;
    if (asInt(Info, 0) != static_cast<int64_t>(clang::CUDAFunctionTarget::Device) ||
        asInt(Info, 3) != static_cast<int64_t>(clang::CUDAFunctionTarget::Host))
      return;
    auto *ND = reinterpret_cast<clang::NamedDecl *>(Info.getRawArg(2));
    if (auto *FD = llvm::dyn_cast<clang::FunctionDecl>(ND))
      Referenced.insert(keyOf(FD->getASTContext().getSourceManager(), FD));
  }
};

/// Walks the \e completed AST and records every \c __device__-only function
/// that is referenced from host (\p Referenced) or \c used yet has no
/// \c __host__ overload — the set the real parse must synthesize handles for.
class ExportPlanConsumer
    : public clang::SemaConsumer,
      public clang::RecursiveASTVisitor<ExportPlanConsumer> {
  llvm::StringSet<> &Synthesize;
  const llvm::StringSet<> &Referenced;
  clang::Sema *SemaRef{nullptr};

public:
  ExportPlanConsumer(llvm::StringSet<> &Synthesize,
                     const llvm::StringSet<> &Referenced)
      : Synthesize(Synthesize), Referenced(Referenced) {}

  void InitializeSema(clang::Sema &S) override { SemaRef = &S; }
  void ForgetSema() override { SemaRef = nullptr; }

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitFunctionDecl(clang::FunctionDecl *FD) {
    if (!SemaRef)
      return true;
    if (SemaRef->CUDA().IdentifyTarget(FD) != clang::CUDAFunctionTarget::Device)
      return true;
    std::string Key = keyOf(SemaRef->getSourceManager(), FD);
    if (!Referenced.contains(Key) && !FD->hasAttr<clang::UsedAttr>())
      return true;
    if (findHostOverload(*SemaRef, FD))
      return true;
    Synthesize.insert(Key);
    return true;
  }

  void HandleTranslationUnit(clang::ASTContext &Ctx) override {
    if (SemaRef)
      TraverseDecl(Ctx.getTranslationUnitDecl());
  }
};

class ExportPlanAction : public clang::ASTFrontendAction {
  llvm::StringSet<> &Synthesize;
  const llvm::StringSet<> &Referenced;

public:
  ExportPlanAction(llvm::StringSet<> &Synthesize,
                   const llvm::StringSet<> &Referenced)
      : Synthesize(Synthesize), Referenced(Referenced) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
    return std::make_unique<ExportPlanConsumer>(Synthesize, Referenced);
  }
};

/// Runs a throwaway, syntax-only pre-parse of \p MainCI's translation unit and
/// returns the location keys of \c __device__-only functions that are
/// referenced from host (or \c used) yet lack a \c __host__ overload. Unlike
/// the streaming real parse, this inspects the \e complete AST, so the standard
/// library's host \c malloc/\c sqrt (declared after their \c __device__ peers)
/// are correctly recognized. Returns empty for non-host CUDA/HIP compiles.
llvm::StringSet<> computeSynthesizeKeys(clang::CompilerInstance &MainCI) {
  llvm::StringSet<> Synthesize;
  const clang::LangOptions &LO = MainCI.getLangOpts();
  if (!(LO.CUDA && !LO.CUDAIsDevice))
    return Synthesize;

  // The main compilation runs with the LLVM IO sandbox armed; our pre-pass
  // legitimately re-reads the same input files, so disable it for the duration.
  auto SandboxOff = llvm::sys::sandbox::scopedDisable();

  auto Inv = std::make_shared<clang::CompilerInvocation>(MainCI.getInvocation());
  // Drop this plugin from the cloned invocation so the pre-pass does not
  // re-instantiate it and recurse: the plugin is a CmdlineBeforeMainAction,
  // which the frontend only runs when its name is listed in AddPluginActions.
  Inv->getFrontendOpts().AddPluginActions.clear();

  llvm::StringSet<> Referenced;
  clang::CompilerInstance PreCI(Inv, MainCI.getPCHContainerOperations());
  // The collector records the device callees of host code and swallows all
  // output; the bad-target references we provoke are expected, and any genuine
  // error is reported by the real parse that follows.
  PreCI.createDiagnostics(new BadRefCollector(Referenced),
                          /*ShouldOwnClient=*/true);
  // Don't let the provoked errors trip the error limit and stop parsing early.
  PreCI.getDiagnostics().setErrorLimit(0);

  ExportPlanAction Action(Synthesize, Referenced);
  PreCI.ExecuteAction(Action);
  return Synthesize;
}

} // namespace

EmitHostHandleForDevFuncConsumer::EmitHostHandleForDevFuncConsumer(
    clang::CompilerInstance &CI)
    : Synthesize(computeSynthesizeKeys(CI)) {}

void EmitHostHandleForDevFuncConsumer::InitializeSema(clang::Sema &S) {
  SemaRef = &S;
  Handles.clear();
  TemplateHandles.clear();
  ExistingHosts.clear();
}

void EmitHostHandleForDevFuncConsumer::ForgetSema() { SemaRef = nullptr; }

bool EmitHostHandleForDevFuncConsumer::HandleTopLevelDecl(
    clang::DeclGroupRef DG) {
  if (!SemaRef)
    return true;
  clang::ASTContext &Ctx = SemaRef->Context;
  if (!(Ctx.getLangOpts().CUDA && !Ctx.getLangOpts().CUDAIsDevice))
    return true;

  clang::Sema &S = *SemaRef;
  const clang::SourceManager &SM = S.getSourceManager();

  // Walks one declaration, recursing through namespace / linkage-spec scopes
  // and the tool's own records, and synthesizes a body-less __host__ overload
  // for every __device__-only function the pre-pass flagged as lacking one.
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

    clang::CUDAFunctionTarget Target = S.CUDA().IdentifyTarget(Dev);
    if (Target == clang::CUDAFunctionTarget::HostDevice) {
      // Already host-addressable; annotate later if it turns out exported.
      ExistingHosts.push_back(Dev);
      continue;
    }
    if (Target != clang::CUDAFunctionTarget::Device)
      continue;

    if (!Synthesize.contains(keyOf(SM, Dev))) {
      // The pre-pass saw a __host__ overload for this function (a user sibling
      // or the standard library's host peer); annotate it instead.
      ExistingHosts.push_back(Dev);
      continue;
    }

    clang::DeclContext *DC = Dev->getDeclContext();
    bool IsMember = DC->isRecord();

    if (DevTpl) {
      clang::FunctionTemplateDecl *&HostTpl =
          TemplateHandles[DevTpl->getCanonicalDecl()];
      if (HostTpl)
        continue;
      clang::FunctionDecl *HostPattern = cloneHostDecl(S, Dev, DC);
      HostTpl = clang::FunctionTemplateDecl::Create(
          Ctx, DC, DevTpl->getLocation(), DevTpl->getDeclName(),
          DevTpl->getTemplateParameters(), HostPattern);
      HostTpl->setAccess(DevTpl->getAccess());
      HostPattern->setDescribedFunctionTemplate(HostTpl);
      DC->addDecl(HostTpl);
      if (!IsMember)
        S.PushOnScopeChains(HostTpl, S.TUScope, /*AddToContext=*/false);
    } else {
      clang::FunctionDecl *Host = cloneHostDecl(S, Dev, DC);
      DC->addDecl(Host);
      if (!IsMember)
        S.PushOnScopeChains(Host, S.TUScope, /*AddToContext=*/false);
      Handles.push_back({Dev, Host});
    }
  }
  return true;
}

void EmitHostHandleForDevFuncConsumer::HandleTranslationUnit(
    clang::ASTContext &Ctx) {
  if (!SemaRef)
    return;
  if (!(Ctx.getLangOpts().CUDA && !Ctx.getLangOpts().CUDAIsDevice))
    return;

  clang::Sema &S = *SemaRef;

  /// A handle is exported when host code referenced it, or when its
  /// \c __device__ source is \c used (a hook that must survive even without a
  /// reference).
  auto isExported = [](const clang::FunctionDecl *Dev,
                       const clang::FunctionDecl *Host) {
    return (Dev && Dev->hasAttr<clang::UsedAttr>()) || Host->isReferenced() ||
           Host->isUsed();
  };

  // Non-template host overloads.
  for (const SynthHandle &H : Handles)
    if (isExported(H.Dev, H.Host))
      finalizeHostHandle(S, H.Host);

  // Template host overloads: finalize each specialization that host code
  // instantiated (or all of them when the device template is `used`).
  for (auto &[DevTpl, HostTpl] : TemplateHandles) {
    bool DevUsed = DevTpl->getTemplatedDecl()->hasAttr<clang::UsedAttr>();
    for (clang::FunctionDecl *Spec : HostTpl->specializations())
      if (DevUsed || Spec->isReferenced() || Spec->isUsed())
        finalizeHostHandle(S, Spec);
  }

  // __device__ functions that already had a host-callable counterpart (a user
  // overload, or the function itself when __host__ __device__): annotate that
  // counterpart in place — no body, no synthesis.
  for (clang::FunctionDecl *Dev : ExistingHosts) {
    clang::FunctionDecl *Host =
        S.CUDA().IdentifyTarget(Dev) == clang::CUDAFunctionTarget::HostDevice
            ? Dev
            : findHostOverload(S, Dev);
    if (!Host || !isExported(Dev, Host))
      continue;
    clang::FunctionDecl *Def = Host->getDefinition();
    if (!Def)
      continue;
    annotateExportHandle(Ctx, Def);
    S.Consumer.HandleTopLevelDecl(clang::DeclGroupRef(Def));
  }
}

} // namespace luthier
