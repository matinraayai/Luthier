; RUN: opt %luthier_tool_ir_compilation_plugin_path -passes=luthier-substitute-amdgcn-intrinsics -S %s | %tee_out FileCheck %s
; Verifies that lower-hip-device-intrinsics rewrites:
;   - llvm.amdgcn.workgroup.id.x -> luthier::workgroupIdX.i32
;   - llvm.amdgcn.workgroup.id.y -> luthier::workgroupIdY.i32
;   - llvm.amdgcn.workgroup.id.z -> luthier::workgroupIdZ.i32
;   - llvm.amdgcn.implicitarg.ptr -> luthier::implicitArgPtr.<ptr-type>
;   - llvm.amdgcn.workitem.id.x  -> threadIdx recompute (readSVA of the
;     entry-captured packed lane-0 work-item id + mbcnt lane + blockDim from the
;     implicit args, decomposed by urem/udiv)
; and removes the original intrinsic declarations.

target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workgroup.id.z()
declare ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
declare i32 @llvm.amdgcn.workitem.id.x()

define i32 @uses_all() {
  %x = call i32 @llvm.amdgcn.workgroup.id.x()
  %y = call i32 @llvm.amdgcn.workgroup.id.y()
  %z = call i32 @llvm.amdgcn.workgroup.id.z()
  %p = call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %xy = add i32 %x, %y
  %xyz = add i32 %xy, %z
  ret i32 %xyz
}

define i32 @uses_tid() #0 {
  %tx = call i32 @llvm.amdgcn.workitem.id.x()
  ret i32 %tx
}

attributes #0 = { "target-features"="+wavefrontsize64" }

; CHECK-NOT: llvm.amdgcn.workgroup.id.x
; CHECK-NOT: llvm.amdgcn.workgroup.id.y
; CHECK-NOT: llvm.amdgcn.workgroup.id.z
; CHECK-NOT: llvm.amdgcn.implicitarg.ptr
; CHECK-NOT: llvm.amdgcn.workitem.id.x

; CHECK: call i32 @"luthier::workgroupIdX.i32"()
; CHECK: call i32 @"luthier::workgroupIdY.i32"()
; CHECK: call i32 @"luthier::workgroupIdZ.i32"()
; CHECK: call ptr addrspace(4) @"luthier::implicitArgPtr.{{.*}}"()

; threadIdx.x recompute: read the packed lane-0 work-item id (SA 13) from the
; SVA, add the lane index (mbcnt), and decompose by blockDim.x (urem).
; CHECK-LABEL: define i32 @uses_tid
; CHECK-DAG: call i32 @"luthier::readSVA{{.*}}"(i8 13)
; CHECK-DAG: call ptr addrspace(4) @"luthier::implicitArgPtr.{{.*}}"()
; CHECK-DAG: call i32 @llvm.amdgcn.mbcnt.lo
; CHECK-DAG: call i32 @llvm.amdgcn.mbcnt.hi
; CHECK: urem i32
