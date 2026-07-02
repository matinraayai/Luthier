; RUN: opt -load-pass-plugin=%luthier_tool_ir_compilation_plugin_path -passes="luthier-tool-device-code-offload-parser-pass" %s -S | %tee_out FileCheck %s

; Verifies that ToolDeviceCodeOffloadParserPass's device-function-handle
; harvester records every host function tagged with the
; luthier.function.export_device_handle marker (in @llvm.global.annotations)
; into the luthier_hip_handles section as a { HostHandle, DeviceName } array,
; and points the annotated HIP-handle-section begin/end slots at the linker's
; __start_/__stop_ boundary symbols.
;
; The CXX plugin gives the host sibling the original __device__ function's exact
; symbol name, so the DeviceName recorded here is simply the host handle's IR
; name.

target triple = "x86_64-unknown-linux-gnu"

; Minimum HIP machinery so the pass does not bail early (needs
; __hipRegisterFatBinary) and can locate the embedded bundle.
declare dso_local ptr @__hipRegisterFatBinary(ptr)
declare dso_local void @__hipUnregisterFatBinary(ptr)
@__hip_fatbin = internal constant [16 x i8] c"__CLANG_OFFLOAD_", section ".hip_fatbin"
@__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @__hip_fatbin, ptr null }
@__hip_gpubin_handle = internal global ptr null

define internal void @__hip_module_ctor() {
  %fb = call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
  store ptr %fb, ptr @__hip_gpubin_handle
  ret void
}
define internal void @__hip_register_globals(ptr %0) {
  ret void
}
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }
]

; The trait's four annotated section-boundary pointer slots. They are located by
; the annotate attribute (in @llvm.global.annotations), not by name.
@FatBegin = internal global ptr null
@FatEnd   = internal global ptr null
@HipBegin = internal global ptr null
@HipEnd   = internal global ptr null

@.str.ob = private unnamed_addr constant [36 x i8] c"luthier_clang_offload_section_begin\00", section "llvm.metadata"
@.str.oe = private unnamed_addr constant [34 x i8] c"luthier_clang_offload_section_end\00", section "llvm.metadata"
@.str.hb = private unnamed_addr constant [33 x i8] c"luthier_hip_handle_section_begin\00", section "llvm.metadata"
@.str.he = private unnamed_addr constant [31 x i8] c"luthier_hip_handle_section_end\00", section "llvm.metadata"
@.str.exp = private unnamed_addr constant [38 x i8] c"luthier.function.export_device_handle\00", section "llvm.metadata"
@.str.other = private unnamed_addr constant [6 x i8] c"decoy\00", section "llvm.metadata"
@.str.src = private unnamed_addr constant [6 x i8] c"f.cpp\00", section "llvm.metadata"

; Host-sibling functions, one per shape, plus a decoy.
define dso_local void @_Z6myHookv() {
  ret void
}
define dso_local void @myCHook() {
  ret void
}
define dso_local void @_Z8tmplHookIiET_S0_(i32 %0) {
  ret void
}
define dso_local void @_Z9decoyHookv() {
  ret void
}

; Four slot annotations + three export markers + one decoy annotation.
@llvm.global.annotations = appending global [8 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @FatBegin, ptr @.str.ob, ptr @.str.src, i32 1, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @FatEnd,   ptr @.str.oe, ptr @.str.src, i32 2, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipBegin, ptr @.str.hb, ptr @.str.src, i32 3, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipEnd,   ptr @.str.he, ptr @.str.src, i32 4, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z6myHookv, ptr @.str.exp, ptr @.str.src, i32 5, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @myCHook, ptr @.str.exp, ptr @.str.src, i32 6, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z8tmplHookIiET_S0_, ptr @.str.exp, ptr @.str.src, i32 7, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @_Z9decoyHookv, ptr @.str.other, ptr @.str.src, i32 8, ptr null }
], section "llvm.metadata"

; --- After the pass ---

; CHECK-DAG: %"struct.luthier::HipHandleInfo" = type { ptr, ptr }

; The recovered device-name strings (symbol verbatim).
; CHECK-DAG: @[[DEV_MYHOOK:[._a-zA-Z0-9]+]] = private constant [11 x i8] c"_Z6myHookv\00"
; CHECK-DAG: @[[DEV_MYCHOOK:[._a-zA-Z0-9]+]] = private constant [8 x i8] c"myCHook\00"
; CHECK-DAG: @[[DEV_SPEC:[._a-zA-Z0-9]+]] = private constant [20 x i8] c"_Z8tmplHookIiET_S0_\00"

; Exactly three handle records ([3 x ...], so the decoy is excluded), packed
; into the luthier_hip_handles section, each pairing the host handle with its
; verbatim device name.
; CHECK-DAG: @[[HDATA:[._a-zA-Z0-9]+]] = private constant [3 x %"struct.luthier::HipHandleInfo"] [%"struct.luthier::HipHandleInfo" { ptr @_Z6myHookv, ptr @[[DEV_MYHOOK]] }, %"struct.luthier::HipHandleInfo" { ptr @myCHook, ptr @[[DEV_MYCHOOK]] }, %"struct.luthier::HipHandleInfo" { ptr @_Z8tmplHookIiET_S0_, ptr @[[DEV_SPEC]] }], section "luthier_hip_handles"

; The array is retained so it survives --gc-sections.
; CHECK-DAG: @llvm.used = {{.*}}@[[HDATA]]

; The linker boundary symbols and the slots pointed at them.
; CHECK-DAG: @__start_luthier_hip_handles = external constant i8
; CHECK-DAG: @__stop_luthier_hip_handles = external constant i8
; CHECK-DAG: @HipBegin = internal global ptr @__start_luthier_hip_handles
; CHECK-DAG: @HipEnd = internal global ptr @__stop_luthier_hip_handles
