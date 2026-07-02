; RUN: opt -load-pass-plugin=%luthier_tool_ir_compilation_plugin_path -passes=luthier-finalize-intrinsics -S %s | %tee_out FileCheck %s
; Verifies that finalize-intrinsics:
;   - deletes the body of intrinsic functions
;   - sets external linkage
;   - renames the function to <demangled>.<retty>.<argty>... form
;   - sets the luthier.intrinsic fn-attr to the demangled base name
;   - leaves non-intrinsic functions untouched

target triple = "amdgcn-amd-amdhsa"

; Mangled: luthier::readReg<unsigned int>(llvm::MCRegister) -> _ZN7luthier7readRegIjEET_N4llvm10MCRegisterE
define internal i32 @_ZN7luthier7readRegIjEET_N4llvm10MCRegisterE(i32 %r) #0 {
  ret i32 0
}

define void @host_helper() {
  ret void
}

attributes #0 = { noinline "luthier.intrinsic" }

; The body must be gone (no `entry:` block on this function after the pass).
; CHECK: declare i32 @"luthier::readReg.i32.i32"(i32) #[[INTR:[0-9]+]]

; Untouched function preserved.
; CHECK: define void @host_helper()

; CHECK: attributes #[[INTR]] = { noinline "luthier.intrinsic"="luthier::readReg" }
