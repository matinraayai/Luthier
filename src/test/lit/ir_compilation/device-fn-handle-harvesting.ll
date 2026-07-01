; RUN: opt %luthier_tool_ir_compilation_plugin_path -passes="luthier-tool-device-code-offload-parser-pass" %s -S | %tee_out FileCheck %s

; Verifies that ToolDeviceCodeOffloadParserPass's device-function-handle
; harvester records every host function tagged with the
; luthier.export_function_handle marker (in @llvm.global.annotations) into the
; unified HipHandles trait slot as a { HostHandle, DeviceName } pair.
;
; The CXX plugin gives the host sibling the original __device__ function's exact
; symbol name, so the DeviceName recorded here is simply the host handle's IR
; symbol verbatim — no demangling or prefix stripping. This is checked across
; the shapes the plugin can produce:
;
;   * Non-templated C++ sibling: Itanium-mangled symbol (_Z6myHookv).
;   * extern "C" sibling: raw source identifier (myCHook).
;   * Templated per-specialization handle: the specialization's own Itanium
;     mangling (_Z8tmplHookIiET_S0_).
;
; A decoy function carrying a different annotation must NOT be recorded.

target triple = "x86_64-unknown-linux-gnu"

; Minimum HIP-side machinery the pass touches. The pass bails early if
; __hipRegisterFatBinary is absent.
declare dso_local ptr @__hipRegisterFatBinary(ptr)
declare dso_local void @__hipUnregisterFatBinary(ptr)

@__hip_fatbin = external constant i8, section ".hip_fatbin"
@__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @__hip_fatbin, ptr null }, section ".hipFatBinSegment", align 8
@__hip_gpubin_handle = internal global ptr null, align 8

define internal void @__hip_module_ctor() {
entry:
  %fb = call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
  store ptr %fb, ptr @__hip_gpubin_handle, align 8
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

; The unified HipHandles trait slot: a linkonce_odr static member of
; luthier::ToolDeviceCodeOffloadParserTrait<Derived>, laid out as
; llvm::ArrayRef<HipHandleInfo> = { ptr Data; i64 Length; }. Detected by the
; pass via its demangled name.
%"class.llvm::ArrayRef" = type { ptr, i64 }
@_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE10HipHandlesE = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8

@.str.exp = private unnamed_addr constant [31 x i8] c"luthier.export_function_handle\00", section "llvm.metadata"
@.str.other = private unnamed_addr constant [12 x i8] c"some.marker\00", section "llvm.metadata"
@.str.src = private unnamed_addr constant [14 x i8] c"/app/test.cpp\00", section "llvm.metadata"

; Host-sibling functions, one per shape, plus a decoy.
define dso_local void @_Z6myHookv() {
entry:
  ret void
}
define dso_local void @myCHook() {
entry:
  ret void
}
define dso_local void @_Z8tmplHookIiET_S0_(i32 %0) {
entry:
  ret void
}
define dso_local void @_Z9decoyHookv() {
entry:
  ret void
}

; Three export-handle markers + one decoy annotation (different string).
@llvm.global.annotations = appending global [4 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z6myHookv, ptr @.str.exp, ptr @.str.src, i32 2, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @myCHook, ptr @.str.exp, ptr @.str.src, i32 3, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z8tmplHookIiET_S0_, ptr @.str.exp, ptr @.str.src, i32 4, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z9decoyHookv, ptr @.str.other, ptr @.str.src, i32 5, ptr null }
], section "llvm.metadata"

; --- After the pass: the HipHandles slot points at a constant data array of
; three { HostHandle, DeviceName } records — one per marker, none for the
; decoy. Each DeviceName is the host handle's IR symbol verbatim. ---

; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" = type { ptr, ptr }

; The recovered device-name strings (symbol verbatim).
; CHECK-DAG: @[[DEV_MYHOOK:[._a-zA-Z0-9]+]] = private constant [11 x i8] c"_Z6myHookv\00"
; CHECK-DAG: @[[DEV_MYCHOOK:[._a-zA-Z0-9]+]] = private constant [8 x i8] c"myCHook\00"
; CHECK-DAG: @[[DEV_SPEC:[._a-zA-Z0-9]+]] = private constant [20 x i8] c"_Z8tmplHookIiET_S0_\00"

; The slot ArrayRef is initialized to view the data array, with length 3.
; CHECK-DAG: @_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE10HipHandlesE = dso_local constant %"class.llvm::ArrayRef" { ptr @[[DATA:[._a-zA-Z0-9]+]], i64 3 }

; Exactly three handle records ([3 x ...], so the decoy is excluded), in
; annotation order, each pairing the host handle with its verbatim device name.
; CHECK-DAG: @[[DATA]] = private constant [3 x %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo"] [%"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @_Z6myHookv, ptr @[[DEV_MYHOOK]] }, %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @myCHook, ptr @[[DEV_MYCHOOK]] }, %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @_Z8tmplHookIiET_S0_, ptr @[[DEV_SPEC]] }]
