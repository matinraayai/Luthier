; RUN: opt -load-pass-plugin=%luthier_tool_ir_compilation_plugin_path -passes=luthier-substitute-amdgcn-intrinsics -S %s | %tee_out FileCheck %s
; Verifies that lower-hip-device-intrinsics rewrites each amdgcn intrinsic
; to a call to luthier::readSVA with the matching ScalarValueArgument slot:
;   - llvm.amdgcn.workgroup.id.x -> luthier::readSVA(i8 WORKGROUP_ID_X=10)
;   - llvm.amdgcn.workgroup.id.y -> luthier::readSVA(i8 WORKGROUP_ID_Y=11)
;   - llvm.amdgcn.workgroup.id.z -> luthier::readSVA(i8 WORKGROUP_ID_Z=12)
;   - llvm.amdgcn.implicitarg.ptr -> luthier::readSVA(i8 IMPLICIT_ARG_BUFFER=9)
;                                    followed by inttoptr to ptr addrspace(4)
; and removes the original intrinsic declarations.

target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workgroup.id.z()
declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()

define i32 @uses_all() {
  %x = call i32 @llvm.amdgcn.workgroup.id.x()
  %y = call i32 @llvm.amdgcn.workgroup.id.y()
  %z = call i32 @llvm.amdgcn.workgroup.id.z()
  %p = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %xy = add i32 %x, %y
  %xyz = add i32 %xy, %z
  ret i32 %xyz
}

; CHECK-NOT: llvm.amdgcn.workgroup.id.x
; CHECK-NOT: llvm.amdgcn.workgroup.id.y
; CHECK-NOT: llvm.amdgcn.workgroup.id.z
; CHECK-NOT: llvm.amdgcn.implicitarg.ptr

; CHECK: call i32 @"luthier::readSVA.i32.i8"(i8 10)
; CHECK: call i32 @"luthier::readSVA.i32.i8"(i8 11)
; CHECK: call i32 @"luthier::readSVA.i32.i8"(i8 12)
; CHECK: call i64 @"luthier::readSVA.i64.i8"(i8 9)
; CHECK: inttoptr i64 %{{.*}} to ptr addrspace(4)
