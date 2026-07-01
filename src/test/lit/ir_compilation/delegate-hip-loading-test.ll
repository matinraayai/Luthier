; RUN: opt %luthier_tool_ir_compilation_plugin_path \
; RUN:   -passes="luthier-tool-device-code-offload-parser-pass" %s -S \
; RUN:   | %tee_out FileCheck %s --implicit-check-not=__hipRegister \
; RUN:     --implicit-check-not=__hipUnregisterFatBinary \
; RUN:     --implicit-check-not=__hip_module_ctor \
; RUN:     --implicit-check-not=__hip_module_dtor \
; RUN:     --implicit-check-not=__hip_register_globals

; Verifies the full ToolDeviceCodeOffloadParserPass lowering of a host module:
;   * every __hipRegister* kind (kernels, device var, managed var, textures,
;     surface) is harvested into the single unified HipHandles trait slot as a
;     { HostHandle, DeviceName } pair;
;   * the embedded fat binary is moved into the luthier_fatbin section and
;     retained, and the FatBinaryStart / FatBinaryStop pointer slots are set to
;     the linker's section-boundary symbols;
;   * the host-side HIP registration machinery (__hip_module_ctor/_dtor,
;     __hip_register_globals, and the __hipRegister*/__hipUnregisterFatBinary
;     declarations) is deleted, and __hip_module_ctor is dropped from
;     llvm.global_ctors (the --implicit-check-not flags on the RUN line prove
;     none of it survives).

target triple = "x86_64-unknown-linux-gnu"

@TexName = private unnamed_addr constant [12 x i8] c"TextureName\00", align 1
@TexName2 = private unnamed_addr constant [13 x i8] c"TextureName2\00", align 1
@DevTexName = private unnamed_addr constant [18 x i8] c"DeviceTextureName\00", align 1
@DevTexName2 = private unnamed_addr constant [19 x i8] c"DeviceTextureName2\00", align 1
@SurName = private unnamed_addr constant [12 x i8] c"SurfaceName\00", align 1
@DevSurName = private unnamed_addr constant [18 x i8] c"DeviceSurfaceName\00", align 1
@VarName = private unnamed_addr constant [8 x i8] c"VarName\00", align 1
@DeviceVarName = private unnamed_addr constant [14 x i8] c"DeviceVarName\00", align 1
@VarManaged = global i64 0, align 8
@SurfaceAddr = global i64 0, align 8
@TextureAddr = global i64 0, align 8
@TextureAddr2 = global i64 0, align 8
@__hip_cuid_60997337ce9624a2 = global i8 0
@__hip_gpubin_handle_60997337ce9624a2 = internal global ptr null, align 8
@DummyVar = dso_local global i64 0
@DummyManagedVariable = dso_local global i64 0

declare dso_local i32 @__hipRegisterFunction(ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr)
declare dso_local void @__hipRegisterVar(ptr, ptr, ptr, ptr, i32, i64, i32, i32)
declare dso_local void @__hipRegisterManagedVar(ptr, ptr, ptr, ptr, i64, i32)
declare dso_local void @__hipRegisterSurface(ptr, ptr, ptr, ptr, i32, i32)
declare dso_local void @__hipRegisterTexture(ptr, ptr, ptr, ptr, i32, i32, i32)
declare dso_local ptr @__hipRegisterFatBinary(ptr)
declare dso_local void @__hipUnregisterFatBinary(ptr)
declare dso_local i32 @atexit(ptr)

;
; The three trait slots Clang emits for
; luthier::ToolDeviceCodeOffloadParserTrait<Derived>: the unified HipHandles
; slot is an llvm::ArrayRef<HipHandleInfo> = { ptr Data; i64 Length; }; the
; FatBinaryStart / FatBinaryStop slots are plain pointers. Detected by the pass
; via their demangled names.
;
%"class.llvm::ArrayRef" = type { ptr, i64 }
@_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE10HipHandlesE = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8
@_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE14FatBinaryStartE = dso_local global ptr null, align 8
@_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE13FatBinaryStopE = dso_local global ptr null, align 8

@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_BinomialOption.cpp, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }]

define internal void @__hip_register_globals(ptr %0) {
entry:
  %1 = call i32 @__hipRegisterFunction(ptr %0, ptr @_Z16binomial_optionsiPK15HIP_vector_typeIfLj4EEPS0_, ptr @0, ptr @0, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
  %2 = call i32 @__hipRegisterFunction(ptr %0, ptr @add_numbers_ptr, ptr @1, ptr @1, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
  call void @__hipRegisterVar(ptr %0, ptr @DummyVar, ptr @VarName, ptr @DeviceVarName, i32 0, i64 0, i32 0, i32 0)
  call void @__hipRegisterManagedVar(ptr %0, ptr @VarManaged, ptr @DummyManagedVariable, ptr @VarName, i64 0, i32 0)
  call void @__hipRegisterSurface(ptr %0, ptr @SurfaceAddr, ptr @SurName, ptr @DevSurName, i32 0, i32 0)
  call void @__hipRegisterTexture(ptr %0, ptr @TextureAddr, ptr @TexName, ptr @DevTexName, i32 0, i32 0, i32 0)
  call void @__hipRegisterTexture(ptr %0, ptr @TextureAddr2, ptr @TexName2, ptr @DevTexName2, i32 0, i32 0, i32 0)
  ret void
}

define internal void @__hip_module_dtor() {
entry:
  %0 = load ptr, ptr @__hip_gpubin_handle_60997337ce9624a2, align 8
  %1 = icmp ne ptr %0, null
  br i1 %1, label %if, label %exit

if:
  call void @__hipUnregisterFatBinary(ptr %0)
  store ptr null, ptr @__hip_gpubin_handle_60997337ce9624a2, align 8
  br label %exit

exit:
  ret void
}

define internal void @__hip_module_ctor() {
entry:
  %0 = load ptr, ptr @__hip_gpubin_handle_60997337ce9624a2, align 8
  %1 = icmp eq ptr %0, null
  br i1 %1, label %if, label %exit

if:
  %2 = call ptr @__hipRegisterFatBinary(ptr @__hip_fatbin_wrapper)
  store ptr %2, ptr @__hip_gpubin_handle_60997337ce9624a2, align 8
  br label %exit

exit:
  %3 = load ptr, ptr @__hip_gpubin_handle_60997337ce9624a2, align 8
  call void @__hip_register_globals(ptr %3)
  %4 = call i32 @atexit(ptr @__hip_module_dtor)
  ret void
}

; --- UNTOUCHED CODE ---
@_Z16binomial_optionsiPK15HIP_vector_typeIfLj4EEPS0_ = dso_local constant ptr @_Z31__device_stub__binomial_optionsiPK15HIP_vector_typeIfLj4EEPS0_, align 8
@add_numbers_ptr = dso_local constant ptr @add_numbers, align 8
; Minimum-shape uncompressed Clang offload bundle: 24-byte magic + 8-byte
; NumEntries(=0). Sized as [32 x i8]; the pass reads the GV's array size
; to compute the runtime MemoryBufferRef extent.
@__hip_fatbin = internal constant [32 x i8] c"__CLANG_OFFLOAD_BUNDLE__\00\00\00\00\00\00\00\00"
@__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @__hip_fatbin, ptr null }
@0 = private unnamed_addr constant [52 x i8] c"_Z16binomial_optionsiPK15HIP_vector_typeIfLj4EEPS0_\00"
@1 = private unnamed_addr constant [12 x i8] c"add_numbers\00"

define internal void @_GLOBAL__sub_I_BinomialOption.cpp() {
  ret void
}

define dso_local void @_Z31__device_stub__binomial_optionsiPK15HIP_vector_typeIfLj4EEPS0_(i32 noundef %numSteps, ptr noundef %randArray, ptr noundef %out) #0 {
  ret void
}

define i32 @add_numbers(i32 %0, i32 %1) {
entry:
  %2 = add i32 %0, %1
  ret i32 %2
}

attributes #0 = { "frame-pointer"="all" }

; --- After the pass: the unified handle record struct and the harvested slots ---

; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" = type { ptr, ptr }

; The HipHandles slot views a 7-element data array (2 kernels, 1 device var,
; 2 textures, 1 surface, 1 managed var). The records are emitted in harvest
; order, so each { HostHandle, DeviceName } pair is matched order-independently.
; CHECK-DAG: @_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE10HipHandlesE = dso_local constant %"class.llvm::ArrayRef" { ptr @[[DATA:[._a-zA-Z0-9]+]], i64 7 }
; CHECK-DAG: @[[DATA]] = private constant [7 x %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo"]

; Kernels: HostHandle = arg1 (host stub ptr), DeviceName = arg3 (name string).
; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @_Z16binomial_optionsiPK15HIP_vector_typeIfLj4EEPS0_, ptr @0 }
; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @add_numbers_ptr, ptr @1 }
; Device var / managed var: HostHandle = shadow ptr, DeviceName = name string.
; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @DummyVar, ptr @VarName }
; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @VarManaged, ptr @VarName }
; Surface / textures: HostHandle = arg1, DeviceName = arg2.
; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @SurfaceAddr, ptr @SurName }
; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @TextureAddr, ptr @TexName }
; CHECK-DAG: %"struct.luthier::ToolDeviceCodeOffloadParser::HipHandleInfo" { ptr @TextureAddr2, ptr @TexName2 }

; Fat binary: moved to the luthier_fatbin section, retained via llvm.used, and
; the FatBinaryStart / FatBinaryStop slots point at the linker boundary symbols.
; CHECK-DAG: @__hip_fatbin = internal constant [32 x i8] c"__CLANG_OFFLOAD_BUNDLE__\00\00\00\00\00\00\00\00", section "luthier_fatbin"
; CHECK-DAG: @llvm.used = appending global [1 x ptr] [ptr @__hip_fatbin], section "llvm.metadata"
; CHECK-DAG: @__start_luthier_fatbin = external constant i8
; CHECK-DAG: @__stop_luthier_fatbin = external constant i8
; CHECK-DAG: @_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE14FatBinaryStartE = dso_local global ptr @__start_luthier_fatbin
; CHECK-DAG: @_ZN7luthier32ToolDeviceCodeOffloadParserTraitIiE13FatBinaryStopE = dso_local global ptr @__stop_luthier_fatbin

; The HIP module ctor is dropped from global_ctors, leaving only the tool's.
; CHECK-DAG: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_BinomialOption.cpp, ptr null }]
