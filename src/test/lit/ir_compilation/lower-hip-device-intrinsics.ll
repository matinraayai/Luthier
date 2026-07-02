; RUN: opt -load-pass-plugin=%luthier_tool_ir_compilation_plugin_path -passes=luthier-substitute-amdgcn-intrinsics -S %s | %tee_out FileCheck %s
; Verifies that lower-hip-device-intrinsics rewrites:
;   - llvm.amdgcn.workgroup.id.x -> luthier::workgroupIdX.i32
;   - llvm.amdgcn.workgroup.id.y -> luthier::workgroupIdY.i32
;   - llvm.amdgcn.workgroup.id.z -> luthier::workgroupIdZ.i32
;   - llvm.amdgcn.implicitarg.ptr -> luthier::implicitArgPtr.<ptr-type>
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

; CHECK: call i32 @"luthier::workgroupIdX.i32"()
; CHECK: call i32 @"luthier::workgroupIdY.i32"()
; CHECK: call i32 @"luthier::workgroupIdZ.i32"()
; CHECK: call ptr addrspace(4) @"luthier::implicitArgPtr.{{.*}}"()
