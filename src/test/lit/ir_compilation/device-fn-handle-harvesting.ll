; RUN: opt %luthier_tool_ir_compilation_plugin_path -passes="luthier-load-hip-fat-binary-info-pass" %s -S | %tee_out FileCheck %s

; Verifies that LoadHIPFATBinaryInfoPass's device-function-handle
; harvester correctly populates the HipDeviceFunctions slot for the
; three sibling-symbol shapes the CXX plugin can produce, and that
; handles tagged with the synthesized-handle marker have their linkage
; flipped to internal (so per-TU auto-generated handles don't clash at
; static link time).
;
;   * Non-templated C++ sibling (dual-overload): symbol is the
;     Itanium mangling of the original device function's source name;
;     DeviceName recovered as the symbol verbatim.
;
;   * Non-templated C-linkage sibling (extern "C"): symbol is the raw
;     source identifier (no Itanium prefix); DeviceName recovered as
;     the symbol verbatim. This exercises the partialDemangle-fail
;     fallback that uses the IR symbol directly.
;
;   * Templated per-specialization handle (synthesized by the
;     consumer): symbol is the Itanium mangling of
;     <DevFuncHandlePrefix><orig-mangling>; DeviceName recovered by
;     demangling and stripping the prefix.
;
;   * In-place annotated \c __host__ overloads (marker
;     \c luthier.function.export_device_handle) keep their original
;     linkage; auto-generated handles (marker
;     \c luthier.function.synthesized_export_device_handle) are
;     flipped to \c internal by the pass.

target triple = "x86_64-unknown-linux-gnu"

; Stand-in for any host bodies the rest of the loader needs to exist.
@__hip_cuid_aabbccdd = global i8 0
@__hip_gpubin_handle_aabbccdd = internal global ptr null, align 8

; Minimum HIP-side declarations LoadHIPFATBinaryInfoPass touches. The
; pass bails early if __hipRegisterFatBinary is absent.
declare dso_local ptr @__hipRegisterFatBinary(ptr)
declare dso_local void @__hipUnregisterFatBinary(ptr)

@__hip_fatbin = external constant i8, section ".hip_fatbin"
@__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @__hip_fatbin, ptr null }, section ".hipFatBinSegment", align 8

define internal void @__hip_module_ctor() {
entry:
  %fb = call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
  store ptr %fb, ptr @__hip_gpubin_handle_aabbccdd, align 8
  ret void
}

define internal void @__hip_module_dtor() {
entry:
  ret void
}

define internal void @__hip_register_globals(ptr %0) {
entry:
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }
]

;
; ArrayRef placeholders. We just need HipFatBinaries +
; HipDeviceFunctions for this test (the device-function path is the
; one we're exercising); the harvester skips slots with no entries.
;
%"class.llvm::ArrayRef" = type { ptr, i64 }

@HipFatBinaries     = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8
@HipDeviceFunctions = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8

@.str.fb  = private unnamed_addr constant [32 x i8] c"luthier.loader.hip_fat_binaries\00", section "llvm.metadata"
@.str.df  = private unnamed_addr constant [36 x i8] c"luthier.loader.hip_device_functions\00", section "llvm.metadata"
@.str.exp = private unnamed_addr constant [38 x i8] c"luthier.function.export_device_handle\00", section "llvm.metadata"
@.str.syn = private unnamed_addr constant [50 x i8] c"luthier.function.synthesized_export_device_handle\00", section "llvm.metadata"
@.str.src = private unnamed_addr constant [14 x i8] c"/app/test.cpp\00", section "llvm.metadata"

;
; Three host-sibling functions, one per shape we care about.
;

; (1) Non-templated C++ sibling: same name as the original __device__
; function. Itanium-mangled.
define dso_local void @_Z6myHookv() #0 {
entry:
  ret void
}

; (2) Non-templated C-linkage sibling (inside extern "C"). Symbol is
; the raw source identifier.
define dso_local void @myCHook() #0 {
entry:
  ret void
}

; (3) Templated per-specialization handle: the consumer mangles the
; original ("_Z6mySpecIiEvT_" — a hypothetical `mySpec<int>(int)`)
; into the source identifier
; `__luthier_builtin_dev_func_handle__Z6mySpecIiEvT_`, then Clang
; Itanium-mangles the whole thing.
define dso_local void @_Z49__luthier_builtin_dev_func_handle__Z6mySpecIiEvT_i(i32 %0) #0 {
entry:
  ret void
}

; (4) Auto-generated non-templated handle: same shape as (1), but
; carries the synthesized-handle marker (there is no user-provided
; __host__ overload for this device function). The pass should flip
; this handle's linkage from dso_local to internal so per-TU copies
; of the auto-generated handle don't collide at static link time.
define dso_local void @_Z9autoHookAv() #0 {
entry:
  ret void
}

attributes #0 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" }

;
; The export-handle annotations: one entry per host sibling. Shape (4)
; uses the synthesized-handle marker; the other three use the plain
; export marker (they simulate the in-place annotated path).
;
@llvm.global.annotations = appending global [6 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @HipFatBinaries,     ptr @.str.fb,  ptr @.str.src, i32 1, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipDeviceFunctions, ptr @.str.df,  ptr @.str.src, i32 1, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z6myHookv, ptr @.str.exp, ptr @.str.src, i32 2, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @myCHook, ptr @.str.exp, ptr @.str.src, i32 3, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z49__luthier_builtin_dev_func_handle__Z6mySpecIiEvT_i, ptr @.str.exp, ptr @.str.src, i32 4, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z9autoHookAv, ptr @.str.syn, ptr @.str.src, i32 5, ptr null }
], section "llvm.metadata"

; Reannotate the device-functions slot too — it's the receiver.
; The harvester appends *its* annotation entry of @HipDeviceFunctions
; using the same .str.df at index 0 of the slot-detection scan.

; --- After the pass: the device-functions slot points at a constant
; data array of three { HostHandle, DeviceName } records. ---

; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryLoader::HipDeviceFunctionInfo" = type { ptr, ptr }

; CHECK-DAG: @[[DEVFN_DATA:[._a-zA-Z0-9]+]] = private constant [4 x %"struct.luthier::DeviceToolCodeFatBinaryLoader::HipDeviceFunctionInfo"]

; For shape (1) — non-templated C++ sibling: DeviceName is the IR
; symbol verbatim.
; CHECK-DAG: @[[DEV_MYHOOK:[._a-zA-Z0-9]+]] = private constant [11 x i8] c"_Z6myHookv\00"
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryLoader::HipDeviceFunctionInfo" { ptr @_Z6myHookv, ptr @[[DEV_MYHOOK]] }

; For shape (2) — extern "C" sibling: partialDemangle fails, fallback
; uses the IR symbol; the prefix isn't present, so DeviceName is the
; IR symbol verbatim.
; CHECK-DAG: @[[DEV_MYCHOOK:[._a-zA-Z0-9]+]] = private constant [8 x i8] c"myCHook\00"
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryLoader::HipDeviceFunctionInfo" { ptr @myCHook, ptr @[[DEV_MYCHOOK]] }

; For shape (3) — templated handle: demangle succeeds, base name has
; the prefix, strip to recover the original specialization's mangling.
; CHECK-DAG: @[[DEV_SPEC:[._a-zA-Z0-9]+]] = private constant [16 x i8] c"_Z6mySpecIiEvT_\00"
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryLoader::HipDeviceFunctionInfo" { ptr @_Z49__luthier_builtin_dev_func_handle__Z6mySpecIiEvT_i, ptr @[[DEV_SPEC]] }

; For shape (4) — auto-generated non-templated handle: symbol is the
; verbatim mangling of the original __device__ function, DeviceName
; recovered as the symbol verbatim, AND the pass flipped its linkage
; from dso_local to internal.
; CHECK-DAG: @[[DEV_AUTO:[._a-zA-Z0-9]+]] = private constant [14 x i8] c"_Z9autoHookAv\00"
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryLoader::HipDeviceFunctionInfo" { ptr @_Z9autoHookAv, ptr @[[DEV_AUTO]] }
; CHECK-DAG: define internal void @_Z9autoHookAv()

; The in-place annotated shapes keep their original linkage.
; CHECK-DAG: define dso_local void @_Z6myHookv()
; CHECK-DAG: define dso_local void @myCHook()
