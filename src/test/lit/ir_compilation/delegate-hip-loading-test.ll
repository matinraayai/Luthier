; RUN: opt %luthier_tool_ir_compilation_plugin_path -passes="luthier-load-hip-fat-binary-info-pass" %s -S | %tee_out FileCheck %s

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

; --- Verify that hip functions of interest are gone ---
; CHECK-NOT: define internal void @__hip_module_ctor
; CHECK-NOT: define internal void @__hip_register_globals
; CHECK-NOT: define internal void @__hip_module_dtor
; CHECK-NOT: declare dso_local ptr @__hipRegisterFatBinary
; CHECK-NOT: declare dso_local i32 @__hipRegisterFunction
; CHECK-NOT: declare dso_local void @__hipRegisterTexture
; CHECK-NOT: declare dso_local void @__hipRegisterSurface
; CHECK-NOT: declare dso_local void @__hipRegisterVar
; CHECK-NOT: declare dso_local void @__hipRegisterManagedVar
; CHECK-NOT: declare dso_local void @__hipUnregisterFatBinary

declare dso_local i32 @__hipRegisterFunction(ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr)
declare dso_local void @__hipRegisterVar(ptr, ptr, ptr, ptr, i32, i64, i32, i32)
declare dso_local void @__hipRegisterManagedVar(ptr, ptr, ptr, ptr, i64, i32)
declare dso_local void @__hipRegisterSurface(ptr, ptr, ptr, ptr, i32, i32)
declare dso_local void @__hipRegisterTexture(ptr, ptr, ptr, ptr, i32, i32, i32)
declare dso_local ptr @__hipRegisterFatBinary(ptr)
declare dso_local void @__hipUnregisterFatBinary(ptr)
declare dso_local i32 @atexit(ptr)

; --- After the pass: element structs exist, ArrayRef slots are init'd ---
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryParser::HipKernelInfo" = type { ptr, ptr }
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryParser::HipManagedVarInfo" = type { ptr, ptr, ptr, i64, i32 }
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryParser::HipDeviceVarInfo" = type { ptr, ptr }
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryParser::HipTextureInfo" = type { ptr, ptr }
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryParser::HipSurfaceInfo" = type { ptr, ptr }

;
; The placeholders below mirror what Clang emits for the slots on
; DeviceToolCodeFatBinaryParser<Derived>. The five `inline static
; llvm::ArrayRef<T>` slots are each a global of two-element struct type
; `{ ptr Data; i64 Length; }` matching ArrayRef's ABI. HipFatBinaries is the
; exception: it is a lone `HipFatBinaryInfo` (`{ ptr Bundle; i64 Size; }`).
;
%"class.llvm::ArrayRef" = type { ptr, i64 }
%"struct.luthier::DeviceToolCodeFatBinaryParser::HipFatBinaryInfo" = type { ptr, i64 }

@HipFatBinaries = dso_local global %"struct.luthier::DeviceToolCodeFatBinaryParser::HipFatBinaryInfo" zeroinitializer, align 8
@HipFunctions   = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8
@HipDeviceVars  = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8
@HipManagedVars = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8
@HipTextureVars = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8
@HipSurfaceVars = dso_local global %"class.llvm::ArrayRef" zeroinitializer, align 8

@.str.fb  = private unnamed_addr constant [32 x i8] c"luthier.loader.hip_fat_binaries\00", section "llvm.metadata"
@.str.fn  = private unnamed_addr constant [27 x i8] c"luthier.loader.hip_kernels\00", section "llvm.metadata"
@.str.dv  = private unnamed_addr constant [31 x i8] c"luthier.loader.hip_device_vars\00", section "llvm.metadata"
@.str.mv  = private unnamed_addr constant [32 x i8] c"luthier.loader.hip_managed_vars\00", section "llvm.metadata"
@.str.tx  = private unnamed_addr constant [32 x i8] c"luthier.loader.hip_texture_vars\00", section "llvm.metadata"
@.str.sf  = private unnamed_addr constant [32 x i8] c"luthier.loader.hip_surface_vars\00", section "llvm.metadata"
@.str.src = private unnamed_addr constant [17 x i8] c"/app/example.cpp\00", section "llvm.metadata"

@llvm.global.annotations = appending global [6 x { ptr, ptr, ptr, i32, ptr }] [
  { ptr, ptr, ptr, i32, ptr } { ptr @HipFatBinaries, ptr @.str.fb, ptr @.str.src, i32 63, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipFunctions,   ptr @.str.fn, ptr @.str.src, i32 67, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipDeviceVars,  ptr @.str.dv, ptr @.str.src, i32 71, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipManagedVars, ptr @.str.mv, ptr @.str.src, i32 75, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipTextureVars, ptr @.str.tx, ptr @.str.src, i32 79, ptr null },
  { ptr, ptr, ptr, i32, ptr } { ptr @HipSurfaceVars, ptr @.str.sf, ptr @.str.src, i32 83, ptr null }
], section "llvm.metadata"

@llvm.compiler.used = appending global [7 x ptr] [ptr @HipFunctions, ptr @HipDeviceVars, ptr @HipFatBinaries, ptr @HipManagedVars, ptr @HipSurfaceVars, ptr @HipTextureVars, ptr @__hip_cuid_60997337ce9624a2], section "llvm.metadata"

; --- VERIFY PROMOTED SLOTS ---
;
; Each ArrayRef slot is set in-place to a `{ ptr to data, i64 N }` initializer
; that views a private constant data array. HipFatBinaries is the exception:
; its lone HipFatBinaryInfo slot is set directly to `{ ptr Bundle, i64 Size }`.
;
; CHECK-DAG: %"struct.luthier::DeviceToolCodeFatBinaryParser::HipFatBinaryInfo" = type { ptr, i64 }
; CHECK-DAG: @HipFatBinaries = dso_local constant %"struct.luthier::DeviceToolCodeFatBinaryParser::HipFatBinaryInfo" { ptr @__hip_fatbin, i64 32 }
;
; CHECK-DAG: @[[FN_DATA:[._a-zA-Z0-9]+]] = private constant [2 x %"struct.luthier::DeviceToolCodeFatBinaryParser::HipKernelInfo"]
; CHECK-DAG: @HipFunctions = dso_local constant %"class.llvm::ArrayRef" { ptr @[[FN_DATA]], i64 2 }
;
; CHECK-DAG: @[[MV_DATA:[._a-zA-Z0-9]+]] = private constant [1 x %"struct.luthier::DeviceToolCodeFatBinaryParser::HipManagedVarInfo"] [%"struct.luthier::DeviceToolCodeFatBinaryParser::HipManagedVarInfo" { ptr @VarManaged, ptr @DummyManagedVariable, ptr @VarName, i64 0, i32 0 }]
; CHECK-DAG: @HipManagedVars = dso_local constant %"class.llvm::ArrayRef" { ptr @[[MV_DATA]], i64 1 }
;
; CHECK-DAG: @[[DV_DATA:[._a-zA-Z0-9]+]] = private constant [1 x %"struct.luthier::DeviceToolCodeFatBinaryParser::HipDeviceVarInfo"] [%"struct.luthier::DeviceToolCodeFatBinaryParser::HipDeviceVarInfo" { ptr @DummyVar, ptr @VarName }]
; CHECK-DAG: @HipDeviceVars = dso_local constant %"class.llvm::ArrayRef" { ptr @[[DV_DATA]], i64 1 }
;
; CHECK-DAG: @[[TX_DATA:[._a-zA-Z0-9]+]] = private constant [2 x %"struct.luthier::DeviceToolCodeFatBinaryParser::HipTextureInfo"]
; CHECK-DAG: @HipTextureVars = dso_local constant %"class.llvm::ArrayRef" { ptr @[[TX_DATA]], i64 2 }
;
; CHECK-DAG: @[[SF_DATA:[._a-zA-Z0-9]+]] = private constant [1 x %"struct.luthier::DeviceToolCodeFatBinaryParser::HipSurfaceInfo"] [%"struct.luthier::DeviceToolCodeFatBinaryParser::HipSurfaceInfo" { ptr @SurfaceAddr, ptr @SurName }]
; CHECK-DAG: @HipSurfaceVars = dso_local constant %"class.llvm::ArrayRef" { ptr @[[SF_DATA]], i64 1 }

; --- VERIFY CONSTRUCTOR MODIFICATION ---
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_BinomialOption.cpp, ptr null }]
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
